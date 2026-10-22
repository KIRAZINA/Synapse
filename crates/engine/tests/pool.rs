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

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use synapse_core::{TokenEvent, TokenStreamProvider};
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
    match InferencePool::load(&path.to_string_lossy(), capacity, context_length) {
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

/// Waits until every slot is free again.
///
/// A generation is over when its terminal event was sent, but the slot is only
/// returned to the free list after the KV cache was cleared, which happens
/// just after the terminal event. The pool is idle when `active()` reaches
/// zero and the slot is back in the channel (the guard returns the slot before
/// it decrements the counter).
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
