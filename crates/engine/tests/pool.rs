//! Integration tests for the inference pool.
//!
//! These tests load the real model and therefore need a few hundred megabytes
//! of memory. They are serialised with [`model_lock`] because Cargo runs the
//! tests of a target in parallel, and every pool keeps its own model and KV
//! caches resident.
//!
//! The model is read from `models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf`. When
//! the file is absent (a checkout without the download script) the tests
//! report that and pass: they are integration tests of a native component, not
//! a reason for `cargo test` to fail on a machine without the weights.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use synapse_core::{TokenEvent, TokenStreamProvider};
use synapse_engine::pool::{FORCE_PANIC_AFTER_GENERATION, FORCE_RETIRE_AFTER_GENERATION};
use synapse_engine::InferencePool;

/// Relative to the workspace root.
const MODEL_FILE: &str = "models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

/// Serialises the tests that load a model.
static MODEL_LOCK: Mutex<()> = Mutex::new(());

/// Resolves the model path, or reports that the test is skipped.
fn model_path() -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(MODEL_FILE);

    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipping: the model file '{}' was not found",
            path.display()
        );
        None
    }
}

/// Serialises the model tests, even after one of them panicked.
fn model_lock() -> MutexGuard<'static, ()> {
    match MODEL_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn load(path: &Path, capacity: u32, context_length: u32) -> InferencePool {
    load_with(path, capacity, context_length, true)
}

fn load_with(
    path: &Path,
    capacity: u32,
    context_length: u32,
    enable_prefix_caching: bool,
) -> InferencePool {
    load_with_options(path, capacity, context_length, enable_prefix_caching, true)
}

fn load_with_options(
    path: &Path,
    capacity: u32,
    context_length: u32,
    enable_prefix_caching: bool,
    enable_affinity_scheduling: bool,
) -> InferencePool {
    match InferencePool::load(
        &path.to_string_lossy(),
        capacity,
        context_length,
        enable_prefix_caching,
        enable_affinity_scheduling,
    ) {
        Ok(pool) => pool,
        Err(err) => panic!("failed to load the inference pool: {err}"),
    }
}

fn start(pool: &InferencePool, prompt: &str, max_tokens: usize) -> flume::Receiver<TokenEvent> {
    match pool.generate_streaming(prompt.to_owned(), max_tokens) {
        Ok(receiver) => receiver,
        Err(err) => panic!("failed to start a generation: {err}"),
    }
}

/// A collected generation.
struct Collected {
    text: String,
    done: bool,
}

/// Blocks until the generation is complete and the slot has been returned.
fn collect(receiver: flume::Receiver<TokenEvent>) -> Collected {
    let mut pieces = Vec::new();
    let mut done = false;

    while let Ok(event) = receiver.recv() {
        match event {
            TokenEvent::Token { text, .. } => pieces.push(text),
            TokenEvent::Done { .. } => done = true,
            TokenEvent::Error { message } => panic!("the generation failed: {message}"),
        }
    }

    Collected {
        text: pieces.concat(),
        done,
    }
}

/// Like [`collect`], but fails instead of hanging when a slot is never
/// released. Used by the concurrency stress tests, where a deadlock would
/// otherwise stall the whole test binary.
fn collect_with_timeout(receiver: flume::Receiver<TokenEvent>, timeout: Duration) -> Collected {
    let mut pieces = Vec::new();
    let mut done = false;

    loop {
        match receiver.recv_timeout(timeout) {
            Ok(TokenEvent::Token { text, .. }) => pieces.push(text),
            Ok(TokenEvent::Done { .. }) => done = true,
            Ok(TokenEvent::Error { message }) => panic!("the generation failed: {message}"),
            Err(flume::RecvTimeoutError::Timeout) => {
                panic!("the generation did not finish within {timeout:?}")
            }
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }
    }

    Collected {
        text: pieces.concat(),
        done,
    }
}

/// Waits until every slot is free again.
///
/// A generation is over when its terminal event was sent, but the slot is only
/// returned to the free queue after the KV cache was trimmed, which happens
/// just after the terminal event. The pool is idle when `active()` reaches
/// zero and the slot is back in the registry (the guard returns the slot
/// before it decrements the counter).
fn wait_until_idle(pool: &InferencePool) {
    let deadline = Instant::now() + Duration::from_secs(60);

    while pool.active() > 0 {
        if Instant::now() > deadline {
            panic!("the pool did not release its slots within 60 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Runs one generation to completion and waits until the slot is reusable.
fn generate(pool: &InferencePool, prompt: &str, max_tokens: usize) -> Collected {
    let collected = collect(start(pool, prompt, max_tokens));
    wait_until_idle(pool);
    collected
}

fn join<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    match handle.join() {
        Ok(value) => value,
        Err(_) => panic!("a collector thread panicked"),
    }
}

/// The one metrics recorder of the test process.
///
/// It is installed lazily by the first test that inspects the counters. A
/// snapshot *drains* counters and histograms, so the values it returns are the
/// deltas since the previous snapshot.
fn snapshotter() -> &'static Snapshotter {
    static SNAPSHOTTER: OnceLock<Snapshotter> = OnceLock::new();

    SNAPSHOTTER.get_or_init(|| {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        if recorder.install().is_err() {
            panic!("a metrics recorder was already installed");
        }
        snapshotter
    })
}

fn take_metrics() -> HashMap<String, DebugValue> {
    snapshotter()
        .snapshot()
        .into_hashmap()
        .into_iter()
        .map(|(key, (_, _, value))| (key.key().name().to_owned(), value))
        .collect()
}

fn counter(metrics: &HashMap<String, DebugValue>, name: &str) -> u64 {
    match metrics.get(name) {
        Some(DebugValue::Counter(value)) => *value,
        _ => 0,
    }
}

fn gauge(metrics: &HashMap<String, DebugValue>, name: &str) -> f64 {
    match metrics.get(name) {
        Some(DebugValue::Gauge(value)) => value.into_inner(),
        _ => f64::NAN,
    }
}

fn histogram(metrics: &HashMap<String, DebugValue>, name: &str) -> Vec<f64> {
    match metrics.get(name) {
        Some(DebugValue::Histogram(values)) => {
            values.iter().map(|value| value.into_inner()).collect()
        }
        _ => Vec::new(),
    }
}

/// Runs a generation and returns when the first token arrived as well.
fn generate_timed(pool: &InferencePool, prompt: &str, max_tokens: usize) -> (Collected, Duration) {
    let started = Instant::now();
    let receiver = start(pool, prompt, max_tokens);
    let mut first_token_at = None;
    let mut pieces = Vec::new();
    let mut done = false;

    loop {
        match receiver.recv_timeout(Duration::from_secs(300)) {
            Ok(TokenEvent::Token { text, .. }) => {
                if first_token_at.is_none() {
                    first_token_at = Some(started.elapsed());
                }
                pieces.push(text);
            }
            Ok(TokenEvent::Done { .. }) => done = true,
            Ok(TokenEvent::Error { message }) => {
                panic!("the generation failed: {message}")
            }
            Err(flume::RecvTimeoutError::Timeout) => panic!("the generation stalled for 300s"),
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }
    }

    wait_until_idle(pool);

    (
        Collected {
            text: pieces.concat(),
            done,
        },
        first_token_at.unwrap_or_else(|| started.elapsed()),
    )
}

#[test]
fn two_slots_really_run_concurrently() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let pool = load(&path, 2, 512);
    assert_eq!(pool.capacity(), 2);
    assert_eq!(pool.active(), 0);

    let first = start(&pool, "Write one sentence about the sea.", 16);
    let second = start(&pool, "Write one sentence about the sky.", 16);

    // Both commands were accepted before either could finish: the two slots
    // are busy at the same time. With the old serial actor this was 1.
    assert_eq!(
        pool.active(),
        2,
        "both slots should be busy after two adjacent requests"
    );

    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(move || collect(first));
        let second = scope.spawn(move || collect(second));
        (join(first), join(second))
    });

    assert!(first.done, "the first generation did not finish");
    assert!(second.done, "the second generation did not finish");

    wait_until_idle(&pool);
    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
}

#[test]
fn concurrent_affinity_requests_do_not_deadlock() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let pool = load(&path, 3, 512);
    let prompt = user_prompt("What is the capital of France?");

    // Three requests taken back to back occupy three slots at once. If the
    // registry lock were held across a generation, the second acquisition
    // would block until the first finished and this would never reach 3.
    let receivers = (0..3).map(|_| start(&pool, &prompt, 8)).collect::<Vec<_>>();
    assert_eq!(
        pool.active(),
        3,
        "the registry lock is held across generation"
    );

    std::thread::scope(|scope| {
        let handles = receivers
            .into_iter()
            .map(|receiver| {
                scope.spawn(move || collect_with_timeout(receiver, Duration::from_secs(120)))
            })
            .collect::<Vec<_>>();

        for handle in handles {
            assert!(join(handle).done, "a concurrent generation did not finish");
        }
    });

    wait_until_idle(&pool);
    assert_eq!(pool.active(), 0);

    // More requests than slots: the excess is rejected immediately and every
    // accepted request completes. No request waits for a busy slot.
    let mut accepted = 0;
    let mut rejected = 0;

    std::thread::scope(|scope| {
        let handles = (0..8)
            .map(|index| {
                let prompt = format!("{prompt}\nVariant {index}");
                let pool = &pool;
                scope.spawn(move || match pool.generate_streaming(prompt, 8) {
                    Ok(receiver) => {
                        let collected = collect_with_timeout(receiver, Duration::from_secs(120));
                        assert!(collected.done, "an accepted request did not finish");
                        true
                    }
                    Err(err) => {
                        assert!(
                            err.to_string().contains("at capacity"),
                            "a burst request was rejected for the wrong reason: {err}"
                        );
                        false
                    }
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            if join(handle) {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
    });

    assert!(accepted >= 1, "every request in the burst was rejected");
    assert_eq!(accepted + rejected, 8, "a burst request was lost");
    assert!(
        rejected >= 1,
        "the burst never exercised the capacity limit"
    );

    wait_until_idle(&pool);
    assert_eq!(pool.active(), 0, "the burst leaked a busy slot");
    assert_eq!(pool.capacity(), 3, "a healthy slot was retired");

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
}

#[test]
fn a_busy_pool_rejects_a_second_request() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let pool = load(&path, 1, 512);

    // The first request keeps the only slot busy for a while.
    let first = start(&pool, "Write ten words about distributed systems.", 24);
    assert_eq!(pool.active(), 1);

    match pool.generate_streaming("This must be rejected.".to_owned(), 8) {
        Ok(_) => panic!("a second request was accepted by a one-slot pool"),
        Err(err) => {
            let message = err.to_string();
            assert!(
                message.contains("at capacity"),
                "the rejection is not clear: {message}"
            );
        }
    }

    // The rejected request must not have disturbed the running one.
    let first = collect(first);
    assert!(first.done, "the in-flight generation was corrupted");

    wait_until_idle(&pool);

    // ... and the slot is still usable.
    let second = generate(&pool, "Write one word about the sky.", 4);
    assert!(second.done);

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
}

#[test]
fn dropping_the_receiver_returns_the_slot() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let pool = load(&path, 1, 512);

    // A long generation that is aborted after the first token, which is what
    // a client disconnect looks like to the engine.
    let receiver = start(&pool, "Write a long story about the sea.", 256);
    match receiver.recv_timeout(Duration::from_secs(60)) {
        Ok(TokenEvent::Token { .. }) => {}
        Ok(other) => panic!("expected a token first, got {other:?}"),
        Err(err) => panic!("the generation never produced a token: {err}"),
    }

    drop(receiver);

    // The abort has to release the slot on its own: no restart, no timeout.
    wait_until_idle(&pool);

    let after = generate(&pool, "Write one word about the sky.", 4);
    assert!(after.done, "the slot was not reused after the abort");

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
}

#[test]
fn a_reused_slot_does_not_inherit_the_previous_request() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    // Same prompt and same seed, but the first run happens on a slot that was
    // already used for a different generation. If the KV cache were not
    // cleared, the model could attend to the warm-up tokens and the outputs
    // would drift apart. The pools are created one after the other so that
    // only one model (and one backend) is resident at a time.
    const WARM_UP: &str = "The history of the Roman empire is a story about";
    const TARGET: &str = "Question: what is the capital of France? Answer:";

    let reused = {
        let pool = load(&path, 1, 512);
        let warm_up = generate(&pool, WARM_UP, 16);
        assert!(warm_up.done);

        let target = generate(&pool, TARGET, 16);
        pool.shutdown()
            .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
        target
    };

    let fresh = {
        let pool = load(&path, 1, 512);
        let target = generate(&pool, TARGET, 16);
        pool.shutdown()
            .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
        target
    };

    assert!(reused.done && fresh.done, "a generation did not finish");
    assert!(
        !reused.text.is_empty() && !fresh.text.is_empty(),
        "the model produced no output"
    );
    assert_eq!(
        reused.text, fresh.text,
        "the reused slot produced a different continuation than a fresh context"
    );
}

/// A system prompt long enough that reusing it is measurable.
const SYSTEM_PROMPT: &str = "You are a careful and concise assistant. You answer questions about \
geography, science and history in a single short sentence. You never add a preamble, you never \
repeat the question, and you never explain your reasoning. If a question is ambiguous, you pick \
the most common interpretation and answer it directly. The user expects a factual answer with no \
filler words, no lists and no follow-up questions.";

fn user_prompt(question: &str) -> String {
    format!("{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:")
}

/// Runs A, A, B, A on a two-slot pool and returns the prefix-cache counters.
///
/// The sequence is deliberately serialised: [`generate`] waits until the pool
/// is idle, so the only variable is which free slot the pool selects. With
/// affinity disabled this is the Phase 8 FIFO; with affinity enabled the
/// second and the last A must return to the slot that already holds A.
fn run_affinity_sequence(
    path: &Path,
    prompts: &[String],
    enable_affinity_scheduling: bool,
) -> (u64, u64, u64, u64) {
    let pool = load_with_options(path, 2, 512, true, enable_affinity_scheduling);
    let _ = take_metrics();

    for prompt in prompts {
        let generated = generate(&pool, prompt, 16);
        assert!(
            generated.done,
            "a generation in the affinity sequence did not finish"
        );
    }

    let metrics = take_metrics();
    let result = (
        counter(&metrics, "synapse_prefix_cache_requests_total"),
        counter(&metrics, "synapse_prefix_cache_hits_total"),
        counter(&metrics, "synapse_prefill_tokens_saved_total"),
        counter(&metrics, "synapse_slot_affinity_matches_total"),
    );

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));

    result
}

#[test]
fn affinity_scheduling_increases_prefix_cache_hits() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let prompt_a = user_prompt("What is the capital of France?");
    let prompt_b = user_prompt("What is the capital of Germany?");
    let prompts = [prompt_a.clone(), prompt_a.clone(), prompt_b, prompt_a];

    let (off_requests, off_hits, off_saved, off_matches) =
        run_affinity_sequence(&path, &prompts, false);
    let (on_requests, on_hits, on_saved, on_matches) = run_affinity_sequence(&path, &prompts, true);

    println!("affinity scheduling: A, A, B, A at capacity 2");
    println!(
        "  disabled: requests={off_requests} hits={off_hits} saved_tokens={off_saved} \
         affinity_matches={off_matches}"
    );
    println!(
        "  enabled:  requests={on_requests} hits={on_hits} saved_tokens={on_saved} \
         affinity_matches={on_matches}"
    );

    assert_eq!(off_requests, prompts.len() as u64);
    assert_eq!(on_requests, prompts.len() as u64);
    assert_eq!(
        off_matches, 0,
        "affinity scored slots even though it is disabled"
    );
    assert!(
        on_matches > 0,
        "affinity never selected a slot with a cached prefix"
    );

    // Opportunistic affinity can only add hits: the prompts, the model and
    // the order of the requests are identical, only the slot choice differs.
    assert!(
        on_hits >= off_hits,
        "affinity reduced the prefix-cache hit count: off={off_hits}, on={on_hits}"
    );
    assert!(
        on_saved >= off_saved,
        "affinity reduced the saved prefill tokens: off={off_saved}, on={on_saved}"
    );
    assert!(
        on_hits > off_hits,
        "the A, A, B, A sequence did not benefit from affinity: off={off_hits}, on={on_hits}"
    );
}

#[test]
fn a_prefix_cache_hit_is_byte_identical_to_a_fresh_context() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    // Request 1 is a cold slot (K = 0). Request 2 shares the system prompt
    // (partial K), request 3 repeats request 2 (K = the whole prompt) and
    // request 4 shares only the chat-template prefix with the previous one.
    // Two slots are used on purpose: affinity scheduling must pick a slot
    // whose cached prefix matches, and the result must still be identical to
    // a fresh context.
    let prompts = [
        user_prompt("What is the capital of France?"),
        user_prompt("What is the capital of Germany?"),
        user_prompt("What is the capital of Germany?"),
        user_prompt("What is the capital of France?"),
    ];

    let mut cached_outputs = Vec::with_capacity(prompts.len());
    let mut later_requests = 0;
    let mut later_hits = 0;
    let mut later_saved = 0;
    let mut later_matches = 0;
    let mut later_prefills = Vec::new();

    let pool = load(&path, 2, 512);

    // Install the recorder and drain anything recorded before this test.
    let _ = take_metrics();

    for (index, prompt) in prompts.iter().enumerate() {
        cached_outputs.push(generate(&pool, prompt, 16));
        let metrics = take_metrics();

        if index == 0 {
            assert_eq!(
                counter(&metrics, "synapse_prefix_cache_requests_total"),
                1,
                "the cold request was not counted"
            );
            assert_eq!(
                counter(&metrics, "synapse_prefix_cache_hits_total"),
                0,
                "a cold slot must not report a cache hit"
            );
            assert_eq!(
                counter(&metrics, "synapse_prefill_tokens_saved_total"),
                0,
                "a cold slot cannot save prefill tokens"
            );
        } else {
            later_requests += counter(&metrics, "synapse_prefix_cache_requests_total");
            later_hits += counter(&metrics, "synapse_prefix_cache_hits_total");
            later_saved += counter(&metrics, "synapse_prefill_tokens_saved_total");
            later_matches += counter(&metrics, "synapse_slot_affinity_matches_total");
            later_prefills.extend(histogram(&metrics, "synapse_prefill_ms"));
        }
    }

    assert_eq!(later_requests, 3);
    assert!(
        later_hits >= 2,
        "the partial and the full hit were not reported: {later_hits}"
    );
    assert!(
        later_saved > 0,
        "no prefill tokens were saved on repeated prefixes"
    );
    assert!(
        later_matches > 0,
        "affinity scheduling never picked a slot with a cached prefix"
    );
    assert_eq!(later_prefills.len(), 3, "one prefill per warm request");

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));

    // The same prompts on contexts that are cleared before every generation
    // are the reference: a cache hit must not change a single byte.
    let fresh_outputs = {
        let pool = load_with(&path, 2, 512, false);
        let outputs = prompts
            .iter()
            .map(|prompt| generate(&pool, prompt, 16))
            .collect::<Vec<_>>();
        pool.shutdown()
            .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
        outputs
    };

    // The Phase 8 path (prefix caching on, affinity off) has to produce the
    // same bytes as well: only the choice of slot differs, never the result.
    let phase8_outputs = {
        let pool = load_with_options(&path, 2, 512, true, false);
        let outputs = prompts
            .iter()
            .map(|prompt| generate(&pool, prompt, 16))
            .collect::<Vec<_>>();
        pool.shutdown()
            .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
        outputs
    };

    for (index, ((cached, fresh), phase8)) in cached_outputs
        .iter()
        .zip(&fresh_outputs)
        .zip(&phase8_outputs)
        .enumerate()
    {
        assert!(cached.done && fresh.done, "prompt {index} did not finish");
        assert!(
            !cached.text.is_empty() && !fresh.text.is_empty(),
            "prompt {index} produced no output"
        );
        assert_eq!(
            cached.text, fresh.text,
            "prompt {index}: the prefix cache changed the continuation"
        );
        assert_eq!(
            cached.text, phase8.text,
            "prompt {index}: disabling affinity changed the continuation"
        );
    }
}

#[test]
fn a_panicked_generation_leaves_a_clean_slot() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let target = user_prompt("What is the capital of France?");
    let pool = load(&path, 1, 512);

    // The first generation fills the slot's prefix cache with the prompt.
    let warm = generate(&pool, &target, 12);
    assert!(warm.done, "the warm-up generation did not finish");

    // The next request is a full prefix hit. The test hook panics after its
    // completion tokens were decoded and before the slot is trimmed, so the
    // KV cache holds a prompt and a completion that must never be reused.
    FORCE_PANIC_AFTER_GENERATION.store(true, Ordering::SeqCst);
    let panicked = generate(&pool, &target, 12);
    assert!(
        panicked.done,
        "the terminal event is sent before the slot is trimmed"
    );
    assert!(
        !panicked.text.is_empty(),
        "the panicked run produced no text"
    );
    assert!(
        !FORCE_PANIC_AFTER_GENERATION.load(Ordering::SeqCst),
        "the test hook was not consumed"
    );

    wait_until_idle(&pool);
    assert_eq!(pool.active(), 0, "the panic leaked a busy slot");

    // If the panic cleanup had failed, the slot would have been retired and
    // this request would be rejected as capacity. The output also has to be
    // exactly the output of the clean warm-up run.
    let after = generate(&pool, &target, 12);
    assert!(after.done, "the slot did not survive the panic");

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));

    assert_eq!(
        after.text, warm.text,
        "the request after the panic inherited state from the panicked generation"
    );
}

#[test]
fn retiring_a_slot_drops_live_capacity_and_telemetry() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let prompt = user_prompt("What is the capital of France?");
    let pool = load(&path, 2, 512);
    let _ = take_metrics();

    assert_eq!(pool.capacity(), 2, "the pool did not start with two slots");
    assert_eq!(
        TokenStreamProvider::max_capacity(&pool),
        2,
        "telemetry does not advertise the live capacity"
    );

    // Retire one healthy slot through the test hook. This is the same path the
    // node takes when a context cannot be cleared after a failure.
    FORCE_RETIRE_AFTER_GENERATION.store(true, Ordering::SeqCst);
    let warmed = generate(&pool, &prompt, 8);
    assert!(
        warmed.done,
        "the generation before retirement did not finish"
    );
    assert!(
        !FORCE_RETIRE_AFTER_GENERATION.load(Ordering::SeqCst),
        "the retirement hook was not consumed"
    );

    wait_until_idle(&pool);
    assert_eq!(pool.active(), 0, "retirement leaked the busy counter");

    // `TokenStreamProvider::max_capacity` is exactly what the network layer
    // copies into `NodeTelemetry.max_capacity`, so this assertion proves that
    // the mesh sees the lost slot.
    assert_eq!(pool.capacity(), 1, "the retired slot still counts as live");
    assert_eq!(
        TokenStreamProvider::max_capacity(&pool),
        1,
        "telemetry still advertises the capacity the node lost"
    );

    let metrics = take_metrics();
    assert_eq!(
        counter(&metrics, "synapse_slots_retired_total"),
        1,
        "the retired slot was not counted"
    );
    assert_eq!(
        gauge(&metrics, "synapse_capacity_live"),
        1.0,
        "the live capacity gauge does not reflect the retirement"
    );

    // The remaining slot still serves, and the retired one is never selected.
    let after = generate(&pool, &prompt, 8);
    assert!(after.done, "the surviving slot stopped serving");

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
}

#[test]
fn prefix_caching_can_be_disabled() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let target = user_prompt("What is the capital of France?");
    let pool = load_with(&path, 1, 512, false);

    let _ = take_metrics();

    for _ in 0..3 {
        let output = generate(&pool, &target, 8);
        assert!(output.done);
    }

    let metrics = take_metrics();

    assert_eq!(
        counter(&metrics, "synapse_prefix_cache_requests_total"),
        0,
        "the disabled cache still counted requests"
    );
    assert_eq!(
        counter(&metrics, "synapse_prefix_cache_hits_total"),
        0,
        "the disabled cache reported a hit"
    );
    assert_eq!(
        counter(&metrics, "synapse_prefill_tokens_saved_total"),
        0,
        "the disabled cache saved prefill tokens"
    );
    assert_eq!(
        counter(&metrics, "synapse_slot_affinity_matches_total"),
        0,
        "affinity matched prefixes even though the cache is disabled"
    );
    assert_eq!(
        histogram(&metrics, "synapse_prefill_ms").len(),
        3,
        "prefill latency is still measured when the cache is off"
    );

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));
}

#[test]
fn a_repeated_prompt_skips_prefill() {
    let Some(path) = model_path() else {
        return;
    };
    let _lock = model_lock();

    let prompt =
        user_prompt("In one sentence, explain why the sky is blue and why sunsets are red.");
    let pool = load(&path, 1, 512);

    // The calibration reference for `tokens_saved`: a full cache hit reuses
    // every prompt token except the last one, whose logits have to be
    // recomputed for the sampler.
    let prompt_tokens = match pool.tokenize_prompt(&prompt) {
        Ok(tokens) => tokens.len(),
        Err(err) => panic!("failed to tokenize the test prompt: {err}"),
    };
    assert!(prompt_tokens > 1, "the test prompt is too short");

    let _ = take_metrics();

    let repetitions = 4;
    let mut prefill_ms = Vec::with_capacity(repetitions);
    let mut tokens_saved = Vec::with_capacity(repetitions);
    let mut ttft_ms = Vec::with_capacity(repetitions);

    for _ in 0..repetitions {
        let (generated, ttft) = generate_timed(&pool, &prompt, 8);
        assert!(generated.done, "a repeated generation did not finish");

        let metrics = take_metrics();
        let values = histogram(&metrics, "synapse_prefill_ms");
        assert_eq!(values.len(), 1, "exactly one prefill per request");

        prefill_ms.push(values[0]);
        tokens_saved.push(counter(&metrics, "synapse_prefill_tokens_saved_total"));
        ttft_ms.push(ttft.as_secs_f64() * 1000.0);
    }

    pool.shutdown()
        .unwrap_or_else(|err| panic!("failed to stop the pool: {err}"));

    let cumulative_saved: u64 = tokens_saved.iter().sum();

    println!("prefix cache: {repetitions} identical requests, capacity 1");
    println!("  prompt_tokens: {prompt_tokens}");
    println!("  prefill_ms per request: {prefill_ms:.1?}");
    println!("  tokens_saved per request: {tokens_saved:?}");
    println!("  time_to_first_token_ms per request: {ttft_ms:.1?}");
    println!("  cumulative prefill tokens saved: {cumulative_saved}");

    // The first request fills the cache and cannot save anything.
    assert_eq!(tokens_saved[0], 0, "the cold request reported a saving");

    // Every later request is a full hit, so it reuses all but the last prompt
    // token (whose logits have to be recomputed). This is the calibration of
    // the count: with a 65-token prompt the saving has to be exactly 64.
    assert_eq!(
        tokens_saved[1],
        (prompt_tokens - 1) as u64,
        "a full hit saved a different number of tokens than prompt_tokens - 1"
    );
    assert!(
        tokens_saved[1..].iter().all(|&saved| saved > 0),
        "a repeated prompt did not reuse its prefix: {tokens_saved:?}"
    );
    assert!(
        tokens_saved[1..]
            .iter()
            .all(|&saved| saved == tokens_saved[1]),
        "identical requests saved different amounts: {tokens_saved:?}"
    );
    assert_eq!(
        cumulative_saved,
        tokens_saved[1] * (repetitions as u64 - 1),
        "the cumulative saving does not add up"
    );

    let warm_prefill = prefill_ms[1..].iter().copied().fold(f64::MIN, f64::max);

    assert!(
        prefill_ms[0] > 2.0 * warm_prefill,
        "the cold prefill ({:.3} ms) was not significantly more expensive than a cached one \
         ({warm_prefill:.3} ms)",
        prefill_ms[0]
    );
}
