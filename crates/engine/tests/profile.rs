//! Real-model profiling (Phase 13).
//!
//! These tests load a real GGUF and are `#[ignore]`d: they take minutes and
//! need gigabytes of RAM. They are the measurement harness behind
//! `docs/GUARANTEES.md`, not part of the fast tier.
//!
//! Every test prints one `PROFILE key=value ...` line per observation. The
//! lines are the source of truth for the numbers in the documentation; nothing
//! is estimated in prose.
//!
//! # Selecting the model
//!
//! `SYNAPSE_PROFILE_MODEL` names the GGUF. When it is unset the 7B profile
//! model is used if it is present, otherwise the bundled TinyLlama.
//!
//! # Running
//!
//! ```text
//! powershell -File scripts/profile.ps1
//! ```
//!
//! or directly:
//!
//! ```text
//! cargo test -p synapse-engine --test profile -- --ignored --exact --nocapture \
//!     profile_single_request
//! ```
//!
//! The tests are serialised by cargo's default test harness when run together;
//! `scripts/profile.ps1` runs each in its own process so the resident set of one
//! scenario cannot inflate the next.

use std::path::{Path, PathBuf};
use std::sync::Barrier;
use std::time::{Duration, Instant};

use synapse_core::{TokenEvent, TokenStreamProvider};
use synapse_engine::{InferencePool, PoolDiagnostics};

const SEVEN_B: &str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
const SMALL: &str = "models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

/// A prompt that produces a deterministic, bounded generation.
const PROMPT: &str = "Explain in two sentences why a bounded queue is useful in an inference server.";

fn model_path() -> PathBuf {
    if let Ok(path) = std::env::var("SYNAPSE_PROFILE_MODEL") {
        return PathBuf::from(path);
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let seven = root.join(SEVEN_B);
    if seven.is_file() {
        seven
    } else {
        root.join(SMALL)
    }
}

fn tokens() -> usize {
    std::env::var("SYNAPSE_PROFILE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64)
}

fn n_ctx() -> u32 {
    std::env::var("SYNAPSE_PROFILE_N_CTX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2048)
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn load(capacity: u32, context_length: u32) -> InferencePool {
    let path = model_path();
    match InferencePool::load(
        &path.to_string_lossy(),
        capacity,
        context_length,
        true,
        true,
        0,
        5_000,
    ) {
        Ok(pool) => pool,
        Err(err) => panic!("failed to load {}: {err}", path.display()),
    }
}

// ---------------------------------------------------------------------------
// Process resource sampling (Windows)
// ---------------------------------------------------------------------------

#[cfg(windows)]
fn memory() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..PROCESS_MEMORY_COUNTERS::default()
    };

    // SAFETY: the handle is the current process and the struct is fully
    // initialised with the size the API expects.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then_some((counters.WorkingSetSize as u64, counters.PeakWorkingSetSize as u64))
}

#[cfg(not(windows))]
fn memory() -> Option<(u64, u64)> {
    None
}

fn rss_mib() -> f64 {
    memory()
        .map(|(current, _)| current as f64 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

fn peak_rss_mib() -> f64 {
    memory()
        .map(|(_, peak)| peak as f64 / (1024.0 * 1024.0))
        .unwrap_or(f64::NAN)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    if values.is_empty() {
        return f64::NAN;
    }
    values[values.len() / 2]
}

fn print_diagnostics(pool: &InferencePool, load_ms: u128, rss_before: f64) {
    let diagnostics: PoolDiagnostics = pool.diagnostics();
    println!(
        "PROFILE scenario=memory model_mib={:.1} parameters={} n_ctx={} slots={} \
         kv_est_mib_per_slot={:.1} kv_est_mib_total={:.1} rss_before_mib={:.1} \
         rss_after_load_mib={:.1} peak_rss_mib={:.1} load_ms={}",
        mib(diagnostics.model_bytes),
        diagnostics.model_parameters,
        diagnostics.context_length,
        diagnostics.slots,
        mib(diagnostics.kv_bytes_per_slot),
        mib(diagnostics.kv_bytes_total),
        rss_before,
        rss_mib(),
        peak_rss_mib(),
        load_ms,
    );
}

// ---------------------------------------------------------------------------
// Generation measurement
// ---------------------------------------------------------------------------

/// The wall-clock facts of one generation.
struct Run {
    ttft_ms: f64,
    decode_ms: f64,
    tokens: usize,
    decode_tok_s: f64,
    e2e_tok_s: f64,
}

fn run_one(pool: &InferencePool, prompt: &str, max_tokens: usize) -> Run {
    let started = Instant::now();
    let receiver = match pool.generate_streaming(prompt.to_owned(), max_tokens) {
        Ok(receiver) => receiver,
        Err(err) => panic!("failed to start a generation: {err}"),
    };

    let mut ttft_ms = f64::NAN;
    let mut done = None;

    loop {
        match receiver.recv() {
            Ok(TokenEvent::Token { .. }) => {
                if ttft_ms.is_nan() {
                    ttft_ms = started.elapsed().as_secs_f64() * 1000.0;
                }
            }
            Ok(TokenEvent::Done {
                total_tokens,
                elapsed_ms,
                ..
            }) => {
                done = Some((total_tokens, elapsed_ms));
                break;
            }
            Ok(TokenEvent::Error { message }) => panic!("the generation failed: {message}"),
            Err(_) => break,
        }
    }

    let (tokens, decode_ms) = done.unwrap_or((0, 0));
    let e2e_ms = started.elapsed().as_secs_f64() * 1000.0;

    Run {
        ttft_ms,
        decode_ms: decode_ms as f64,
        tokens,
        decode_tok_s: if decode_ms > 0 {
            tokens as f64 / (decode_ms as f64 / 1000.0)
        } else {
            f64::NAN
        },
        e2e_tok_s: if e2e_ms > 0.0 {
            tokens as f64 / (e2e_ms / 1000.0)
        } else {
            f64::NAN
        },
    }
}

fn print_run(label: &str, run: &Run) {
    println!(
        "PROFILE scenario={label} ttft_ms={:.1} decode_ms={:.1} tokens={} \
         decode_tok_s={:.2} e2e_tok_s={:.2}",
        run.ttft_ms, run.decode_ms, run.tokens, run.decode_tok_s, run.e2e_tok_s,
    );
}

fn wait_until_idle(pool: &InferencePool) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while pool.active() > 0 {
        assert!(Instant::now() < deadline, "the pool did not go idle");
        std::thread::sleep(Duration::from_millis(20));
    }
}

// ---------------------------------------------------------------------------
// 1. Memory: one process per n_ctx (scripts/profile.ps1 drives the loop)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "loads a real model and measures the resident set"]
fn profile_memory() {
    let context_length = n_ctx();
    let capacity = u32::try_from(env_usize("SYNAPSE_PROFILE_SLOTS", 1)).unwrap_or(1);
    let rss_before = rss_mib();

    let started = Instant::now();
    let pool = load(capacity, context_length);
    let load_ms = started.elapsed().as_millis();

    print_diagnostics(&pool, load_ms, rss_before);

    // Force the first decode so the compute buffers are touched as well; the
    // KV cache itself is allocated when the context is created.
    let run = run_one(&pool, PROMPT, 8);
    println!(
        "PROFILE scenario=memory_touch ttft_ms={:.1} decode_tok_s={:.2} rss_mib={:.1} peak_rss_mib={:.1}",
        run.ttft_ms,
        run.decode_tok_s,
        rss_mib(),
        peak_rss_mib(),
    );

    wait_until_idle(&pool);
    match pool.shutdown() {
        Ok(()) => {}
        Err(err) => panic!("failed to stop the pool: {err}"),
    }
}

// ---------------------------------------------------------------------------
// 2. Single request and concurrent load
// ---------------------------------------------------------------------------

#[test]
#[ignore = "loads a real model and generates tokens"]
fn profile_single_request() {
    let context_length = n_ctx();
    let capacity = u32::try_from(env_usize("SYNAPSE_PROFILE_SLOTS", 1)).unwrap_or(1);
    let max_tokens = tokens();
    let rss_before = rss_mib();

    let started = Instant::now();
    let pool = load(capacity, context_length);
    print_diagnostics(&pool, started.elapsed().as_millis(), rss_before);

    // Warm up: the first generation pays for lazy kernel setup and page faults
    // that the steady state does not. The prompt is unrelated to the measured
    // ones so it cannot make them cache hits.
    let _ = run_one(&pool, PROMPT, 8);
    wait_until_idle(&pool);

    let runs = env_usize("SYNAPSE_PROFILE_RUNS", 3);
    let mut cold_ttfts = Vec::new();
    let mut cold_rates = Vec::new();
    let mut e2e_rates = Vec::new();

    // Cold: a unique first token per run, so the prefix cache cannot serve the
    // prompt and every run pays the full prefill. This is the TTFT a request
    // with no shared prefix sees.
    for index in 0..runs {
        let prompt = format!("Request {index}: {PROMPT}");
        let run = run_one(&pool, &prompt, max_tokens);
        print_run("single_cold", &run);
        cold_ttfts.push(run.ttft_ms);
        cold_rates.push(run.decode_tok_s);
        e2e_rates.push(run.e2e_tok_s);
        wait_until_idle(&pool);
    }

    // Warm: the same prompt twice. The first run populates the slot's KV cache,
    // the second reuses it, so its TTFT is the cache-hit latency.
    let _ = run_one(&pool, PROMPT, max_tokens);
    wait_until_idle(&pool);
    let mut warm_ttfts = Vec::new();
    for _ in 0..2 {
        let run = run_one(&pool, PROMPT, max_tokens);
        print_run("single_warm", &run);
        warm_ttfts.push(run.ttft_ms);
        wait_until_idle(&pool);
    }

    println!(
        "PROFILE scenario=single_summary cold_runs={} median_cold_ttft_ms={:.1} \
         median_decode_tok_s={:.2} median_e2e_tok_s={:.2} \
         warm_runs={} median_warm_ttft_ms={:.1} peak_rss_mib={:.1}",
        runs,
        median(&mut cold_ttfts),
        median(&mut cold_rates),
        median(&mut e2e_rates),
        warm_ttfts.len(),
        median(&mut warm_ttfts),
        peak_rss_mib(),
    );

    match pool.shutdown() {
        Ok(()) => {}
        Err(err) => panic!("failed to stop the pool: {err}"),
    }
}

#[test]
#[ignore = "loads a real model and runs concurrent generations"]
fn profile_concurrent_load() {
    let context_length = n_ctx();
    let capacity = u32::try_from(env_usize("SYNAPSE_PROFILE_SLOTS", 2)).unwrap_or(2);
    let requests = env_usize("SYNAPSE_PROFILE_REQUESTS", capacity as usize);
    let max_tokens = tokens();
    let rss_before = rss_mib();

    let started = Instant::now();
    let pool = load(capacity, context_length);
    print_diagnostics(&pool, started.elapsed().as_millis(), rss_before);

    // Warm every slot before measuring. Identical prompts would all be sent to
    // the slot with the longest cached prefix (affinity), leaving the other
    // slots' compute buffers untouched; distinct suffixes force the FIFO tie
    // breaker to cycle through the pool.
    for index in 0..capacity {
        let warmup = format!("{PROMPT} Warm-up request {index}.");
        let _ = run_one(&pool, &warmup, 8);
    }
    wait_until_idle(&pool);

    let barrier = Barrier::new(requests);
    let wall_start = Instant::now();
    let prompts: Vec<String> = (0..requests)
        .map(|index| format!("Request {index}: {PROMPT}"))
        .collect();
    let pool_ref = &pool;
    let barrier_ref = &barrier;
    let prompts_ref = &prompts;

    let mut results: Vec<Run> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..requests)
            .map(|index| {
                scope.spawn(move || {
                    barrier_ref.wait();
                    // A unique first token per request: a shared prompt would be
                    // a prefix-cache hit and would measure the cache, not the
                    // concurrent cold path.
                    run_one(pool_ref, &prompts_ref[index], max_tokens)
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(run) => run,
                Err(_) => panic!("a generation thread panicked"),
            })
            .collect()
    });

    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let total_tokens: usize = results.iter().map(|run| run.tokens).sum();
    let aggregate_tok_s = total_tokens as f64 / (wall_ms / 1000.0);

    let mut ttfts: Vec<f64> = results.iter().map(|run| run.ttft_ms).collect();
    let mut decode_rates: Vec<f64> = results.iter().map(|run| run.decode_tok_s).collect();

    for run in &results {
        print_run("concurrent", run);
    }
    println!(
        "PROFILE scenario=concurrent_summary slots={} requests={} wall_ms={:.1} \
         total_tokens={} aggregate_tok_s={:.2} median_ttft_ms={:.1} \
         min_ttft_ms={:.1} max_ttft_ms={:.1} median_decode_tok_s={:.2} peak_rss_mib={:.1}",
        capacity,
        requests,
        wall_ms,
        total_tokens,
        aggregate_tok_s,
        median(&mut ttfts),
        results
            .iter()
            .map(|run| run.ttft_ms)
            .fold(f64::INFINITY, f64::min),
        results
            .iter()
            .map(|run| run.ttft_ms)
            .fold(f64::NEG_INFINITY, f64::max),
        median(&mut decode_rates),
        peak_rss_mib(),
    );

    results.clear();
    match pool.shutdown() {
        Ok(()) => {}
        Err(err) => panic!("failed to stop the pool: {err}"),
    }
}

// ---------------------------------------------------------------------------
// 3. Prefix cache: a long shared system prompt, then the same prefix again
// ---------------------------------------------------------------------------

/// Builds a prompt whose shared prefix is large and whose suffix is unique.
fn prefix_prompt(suffix: &str, sentences: usize) -> String {
    let sentence = "You are a careful, concise assistant for a distributed inference network. \
                    Always explain your reasoning in plain language and never invent facts. ";
    let mut prompt = sentence.repeat(sentences);
    prompt.push_str("\n\nQuestion: ");
    prompt.push_str(suffix);
    prompt
}

#[test]
#[ignore = "loads a real model and measures the prefix cache"]
fn profile_prefix_cache() {
    let context_length = n_ctx();
    let max_tokens = tokens().min(32);
    let sentences = env_usize("SYNAPSE_PROFILE_PREFIX_SENTENCES", 12);
    let rss_before = rss_mib();

    let started = Instant::now();
    let pool = load(1, context_length);
    print_diagnostics(&pool, started.elapsed().as_millis(), rss_before);

    let cold_prompt = prefix_prompt("Name the capital of France.", sentences);
    let warm_prompt = prefix_prompt("Name the capital of Germany.", sentences);
    let warm_prompt_2 = prefix_prompt("Name the capital of Italy.", sentences);

    let prompt_tokens = match pool.tokenize_prompt(&cold_prompt) {
        Ok(tokens) => tokens.len(),
        Err(err) => panic!("failed to tokenize the profiling prompt: {err}"),
    };

    // First-touch warm-up with an unrelated prompt. Without this the "cold"
    // run below would also pay the one-off compute-buffer allocation and the
    // page faults of the mapped weights, which would be reported as prefill
    // cost and would exaggerate the cache saving.
    let _ = run_one(&pool, PROMPT, 8);
    wait_until_idle(&pool);

    // Cold: an unrelated prompt, so nothing of this 317-token prefix is in the
    // KV cache and the whole prompt is prefilled.
    let _ = take_metrics(); // drain anything recorded while loading and warming
    let cold = run_one(&pool, &cold_prompt, max_tokens);
    print_run("prefix_cold", &cold);
    wait_until_idle(&pool);
    let cold_metrics = take_metrics();

    // Warm: the next prompt shares the long prefix, so only the suffix is
    // prefilled. Repeat to show the steady state rather than a one-off.
    let mut warm_ttfts = Vec::new();
    let mut warm_rates = Vec::new();
    let warm_runs = env_usize("SYNAPSE_PROFILE_RUNS", 3);
    for index in 0..warm_runs {
        let prompt = if index % 2 == 0 {
            &warm_prompt
        } else {
            &warm_prompt_2
        };
        let run = run_one(&pool, prompt, max_tokens);
        print_run("prefix_warm", &run);
        warm_ttfts.push(run.ttft_ms);
        warm_rates.push(run.decode_tok_s);
        wait_until_idle(&pool);
    }

    // The engine's own counters are authoritative for the hit rate and for how
    // many prompt tokens were not prefilled.
    let warm_metrics = take_metrics();
    let mut cold_prefill = cold_metrics.prefill_ms.clone();
    let mut warm_prefill = warm_metrics.prefill_ms.clone();
    let warm_runs_f = warm_runs.max(1) as f64;
    let saved_per_request = warm_metrics.saved_tokens as f64 / warm_runs_f;
    let cold_prefill_median = median(&mut cold_prefill);
    let warm_prefill_median = median(&mut warm_prefill);

    println!(
        "PROFILE scenario=prefix_summary prompt_tokens={} cold_ttft_ms={:.1} \
         warm_median_ttft_ms={:.1} warm_median_decode_tok_s={:.2} \
         cold_prefill_ms={:.1} warm_prefill_median_ms={:.1} \
         prefill_saved_pct={:.1} cache_requests={} cache_hits={} hit_rate={:.3} \
         prefill_tokens_saved_per_request={:.1} saved_pct_of_prompt={:.1} \
         peak_rss_mib={:.1}",
        prompt_tokens,
        cold.ttft_ms,
        median(&mut warm_ttfts),
        median(&mut warm_rates),
        cold_prefill_median,
        warm_prefill_median,
        if cold_prefill_median > 0.0 {
            100.0 * (cold_prefill_median - warm_prefill_median) / cold_prefill_median
        } else {
            f64::NAN
        },
        warm_metrics.requests,
        warm_metrics.hits,
        if warm_metrics.requests > 0 {
            warm_metrics.hits as f64 / warm_metrics.requests as f64
        } else {
            f64::NAN
        },
        saved_per_request,
        if prompt_tokens > 0 {
            100.0 * saved_per_request / prompt_tokens as f64
        } else {
            f64::NAN
        },
        peak_rss_mib(),
    );

    match pool.shutdown() {
        Ok(()) => {}
        Err(err) => panic!("failed to stop the pool: {err}"),
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
struct MetricSnapshot {
    requests: u64,
    hits: u64,
    saved_tokens: u64,
    prefill_ms: Vec<f64>,
}

/// The one metrics recorder of this test process.
///
/// `snapshot()` drains counters and histograms, so each call returns the delta
/// since the previous one. That is exactly what per-phase measurement needs.
fn snapshotter() -> &'static metrics_util::debugging::Snapshotter {
    use metrics_util::debugging::DebuggingRecorder;

    static SNAPSHOTTER: std::sync::OnceLock<metrics_util::debugging::Snapshotter> =
        std::sync::OnceLock::new();

    SNAPSHOTTER.get_or_init(|| {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let _ = recorder.install();
        snapshotter
    })
}

fn take_metrics() -> MetricSnapshot {
    use metrics_util::debugging::DebugValue;

    let mut snapshot = MetricSnapshot::default();

    for (key, (_, _, value)) in snapshotter().snapshot().into_hashmap() {
        match key.key().name() {
            "synapse_prefix_cache_requests_total" => {
                if let DebugValue::Counter(value) = value {
                    snapshot.requests = value;
                }
            }
            "synapse_prefix_cache_hits_total" => {
                if let DebugValue::Counter(value) = value {
                    snapshot.hits = value;
                }
            }
            "synapse_prefill_tokens_saved_total" => {
                if let DebugValue::Counter(value) = value {
                    snapshot.saved_tokens = value;
                }
            }
            "synapse_prefill_ms" => {
                if let DebugValue::Histogram(values) = value {
                    snapshot.prefill_ms = values
                        .iter()
                        .map(|value| value.into_inner())
                        .collect();
                }
            }
            _ => {}
        }
    }

    snapshot
}

