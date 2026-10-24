//! Real-model cost of redundant verification (Phase 13).
//!
//! This measures the *compute* cost of `[verification].enabled = true` with
//! `K = 3` against the disabled single-node path, on a real GGUF. It uses the
//! production orchestrator ([`run_verification`] through [`VerificationClient`])
//! with a real [`InferencePool`] behind every participant.
//!
//! # What is and is not measured
//!
//! The fan-out is in-process: the three participants are three slots of one
//! pool on one host, not three hosts over QUIC. That is deliberate. The number
//! this test answers is "what does K-fold compute cost", which is independent
//! of placement. It does **not** include the QUIC/auth round trip; on three
//! separate hosts the wall-clock multiplier is `max(node latency) + RTT`
//! instead of the single-host contention measured here.
//!
//! Run with:
//!
//! ```text
//! powershell -File scripts/profile.ps1
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libp2p::PeerId;
use synapse_core::{
    Determinism, NodeTelemetry, TokenEvent, TokenStreamProvider, VerificationConfig,
};
use synapse_engine::InferencePool;
use synapse_net::orchestrator::VerificationClient;
use synapse_net::routing::RoutingTable;

const SEVEN_B: &str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf";
const SMALL: &str = "models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

const PROMPT: &str =
    "Explain in two sentences why a bounded queue is useful in an inference server.";

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
        .unwrap_or(48)
}

fn runs() -> usize {
    std::env::var("SYNAPSE_PROFILE_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3)
}

fn n_ctx() -> u32 {
    std::env::var("SYNAPSE_PROFILE_N_CTX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2048)
}

/// The per-participant fan-out timeout.
///
/// `DEFAULT_PARTICIPANT_TIMEOUT` is 30 s, which is right for a GPU or a warm
/// server but shorter than a 7B decode on a 4-core CPU: three concurrent
/// participants at 2 threads each decode below 1 token/s. The profiling run
/// therefore raises the timeout so that the measurement records a completed
/// fan-out rather than a timeout. The production default is untouched.
fn verify_timeout() -> Duration {
    std::env::var("SYNAPSE_PROFILE_VERIFY_TIMEOUT_S")
        .ok()
        .and_then(|value| value.parse().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(600))
}

fn model_name() -> String {
    std::env::var("SYNAPSE_PROFILE_MODEL_NAME").unwrap_or_else(|_| {
        if model_path().to_string_lossy().contains("Qwen") {
            "qwen2.5-7b-instruct".to_owned()
        } else {
            "tinyllama-1.1b".to_owned()
        }
    })
}

#[cfg(windows)]
fn cpu_ms() -> f64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    fn ticks(time: FILETIME) -> u64 {
        ((time.dwHighDateTime as u64) << 32) | time.dwLowDateTime as u64
    }

    let mut creation = FILETIME::default();
    let mut exit = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();

    // SAFETY: the handle is the current process and every out-pointer is a
    // valid, initialised FILETIME.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };

    if ok == 0 {
        return f64::NAN;
    }

    // FILETIME counts 100 ns intervals.
    (ticks(kernel) + ticks(user)) as f64 / 10_000.0
}

#[cfg(not(windows))]
fn cpu_ms() -> f64 {
    f64::NAN
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

/// The facts of one observed generation.
#[derive(Debug, Clone, Copy)]
struct Observation {
    wall_ms: f64,
    cpu_ms: f64,
    tokens: usize,
}

impl Observation {
    fn cpu_ms_per_token(&self) -> f64 {
        if self.tokens == 0 {
            f64::NAN
        } else {
            self.cpu_ms / self.tokens as f64
        }
    }
}

fn wait_until_idle(pool: &InferencePool) {
    let deadline = Instant::now() + Duration::from_secs(300);
    while pool.active() > 0 {
        assert!(Instant::now() < deadline, "the pool did not go idle");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Runs one greedy generation directly on the pool and records wall and CPU.
fn baseline(pool: &InferencePool, max_tokens: usize) -> Observation {
    baseline_with_prompt(pool, PROMPT, max_tokens)
}

fn baseline_with_prompt(pool: &InferencePool, prompt: &str, max_tokens: usize) -> Observation {
    wait_until_idle(pool);
    let cpu_before = cpu_ms();
    let started = Instant::now();

    let receiver = match pool.generate_streaming_greedy(prompt.to_owned(), max_tokens) {
        Ok(receiver) => receiver,
        Err(err) => panic!("failed to start the baseline generation: {err}"),
    };

    let total_tokens = drain(receiver);
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cpu = cpu_ms() - cpu_before;

    Observation {
        wall_ms,
        cpu_ms: cpu,
        tokens: total_tokens,
    }
}

fn drain(receiver: flume::Receiver<TokenEvent>) -> usize {
    loop {
        match receiver.recv() {
            Ok(TokenEvent::Token { .. }) => {}
            Ok(TokenEvent::Done { total_tokens, .. }) => return total_tokens,
            Ok(TokenEvent::Error { message }) => panic!("the generation failed: {message}"),
            Err(_) => return 0,
        }
    }
}

fn telemetry(peer: &PeerId, model: &str) -> NodeTelemetry {
    NodeTelemetry {
        peer_id: peer.to_base58(),
        quic_port: 0,
        addresses: vec!["/ip4/127.0.0.1/tcp/4001".to_owned()],
        active_requests: 0,
        max_capacity: 1,
        supported_models: vec![model.to_owned()],
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "loads a real model and runs K=3 verification"]
async fn profile_verification_cost() {
    let context_length = n_ctx();
    let max_tokens = tokens();
    let model = model_name();
    let path = model_path();

    // One pool with three slots: the three participants are three slots, each
    // with `available_parallelism / 3` threads, exactly as three single-slot
    // pools would be on one host.
    let pool = match InferencePool::load(
        &path.to_string_lossy(),
        3,
        context_length,
        true,
        true,
        0,
        5_000,
    ) {
        Ok(pool) => pool,
        Err(err) => panic!("failed to load {}: {err}", path.display()),
    };
    let diagnostics = pool.diagnostics();
    let threads_per_slot = std::thread::available_parallelism()
        .map(|value| (value.get() / 3).max(1))
        .unwrap_or(2) as u32;

    println!(
        "PROFILE scenario=verification_setup model_mib={:.1} parameters={} n_ctx={} slots={} \
         kv_est_mib_per_slot={:.1} kv_est_mib_total={:.1} threads_per_slot={}",
        mib(diagnostics.model_bytes),
        diagnostics.model_parameters,
        diagnostics.context_length,
        diagnostics.slots,
        mib(diagnostics.kv_bytes_per_slot),
        mib(diagnostics.kv_bytes_total),
        threads_per_slot,
    );

    let pinned = Determinism::greedy(model.clone(), "Q4_K_M", 1234, threads_per_slot, "cpu");

    // Warm all three slots so the first measured request does not pay for lazy
    // context setup. Distinct prompts make affinity's FIFO tie breaker cycle
    // through all three slots instead of reusing slot 0.
    for index in 0..3 {
        let warmup = format!("{PROMPT} Warm-up request {index}.");
        let _ = baseline_with_prompt(&pool, &warmup, 8);
    }
    wait_until_idle(&pool);

    let pool = Arc::new(pool);

    // The production entry point. The routing table supplies the ledger; the
    // three providers are distinct peers backed by this pool's three slots.
    let config = VerificationConfig {
        enabled: true,
        redundancy: 3,
        ..VerificationConfig::default()
    };
    let peers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();
    let routing = Arc::new(RoutingTable::with_policy(Duration::from_secs(300), &config));
    for peer in &peers {
        routing.update(*peer, telemetry(peer, &model));
    }
    let client = VerificationClient::new(Arc::clone(&routing), config.clone())
        .with_timeout(verify_timeout());

    println!(
        "PROFILE scenario=verification_policy redundancy={} sample_every={} participant_timeout_s={} \
         quorum=all_must_answer",
        config.redundancy,
        config.sample_every,
        verify_timeout().as_secs(),
    );

    let mut baseline_walls = Vec::new();
    let mut baseline_cpus = Vec::new();
    let mut baseline_rates = Vec::new();
    let mut verify_walls = Vec::new();
    let mut verify_cpus = Vec::new();
    let mut verify_rates = Vec::new();
    let mut verify_answers = Vec::new();
    let mut accepted = 0usize;

    for index in 0..runs() {
        let observation = baseline(&pool, max_tokens);
        println!(
            "PROFILE scenario=verification_baseline run={} wall_ms={:.1} cpu_ms={:.1} \
             tokens={} cpu_ms_per_token={:.2}",
            index,
            observation.wall_ms,
            observation.cpu_ms,
            observation.tokens,
            observation.cpu_ms_per_token(),
        );
        baseline_walls.push(observation.wall_ms);
        baseline_cpus.push(observation.cpu_ms);
        baseline_rates.push(observation.cpu_ms_per_token());
        wait_until_idle(&pool);

        let providers: Vec<(String, Determinism, Arc<dyn TokenStreamProvider>)> = peers
            .iter()
            .map(|peer| {
                (
                    peer.to_base58(),
                    pinned.clone(),
                    Arc::clone(&pool) as Arc<dyn TokenStreamProvider>,
                )
            })
            .collect();

        let cpu_before = cpu_ms();
        let started = Instant::now();
        let outcome = client
            .verify(PROMPT, max_tokens as u32, pinned.clone(), providers)
            .await
            .unwrap_or_else(|err| panic!("the verified request should run: {err}"));
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cpu = cpu_ms() - cpu_before;

        if outcome.is_accepted() {
            accepted += 1;
        }
        let answered_tokens = max_tokens * outcome.answered;
        println!(
            "PROFILE scenario=verification_enabled run={} wall_ms={:.1} cpu_ms={:.1} \
             generated_tokens_upper={} cpu_ms_per_token_upper={:.2} requested={} answered={} accepted={}",
            index,
            wall_ms,
            cpu,
            answered_tokens,
            if answered_tokens > 0 {
                cpu / answered_tokens as f64
            } else {
                f64::NAN
            },
            outcome.requested,
            outcome.answered,
            outcome.is_accepted(),
        );
        verify_walls.push(wall_ms);
        verify_cpus.push(cpu);
        verify_rates.push(if answered_tokens > 0 {
            cpu / answered_tokens as f64
        } else {
            f64::NAN
        });
        verify_answers.push(outcome.answered);
        wait_until_idle(&pool);
    }

    let baseline_wall = median(&mut baseline_walls);
    let baseline_cpu = median(&mut baseline_cpus);
    let verify_wall = median(&mut verify_walls);
    let verify_cpu = median(&mut verify_cpus);

    let total_answered: usize = verify_answers.iter().sum();
    let baseline_rate = median(&mut baseline_rates);
    let verify_rate = median(&mut verify_rates);

    println!(
        "PROFILE scenario=verification_summary runs={} max_tokens={} \
         baseline_wall_ms={:.1} baseline_cpu_ms={:.1} baseline_cpu_ms_per_token={:.2} \
         verify_wall_ms={:.1} verify_cpu_ms={:.1} verify_cpu_ms_per_token_upper={:.2} \
         wall_multiplier={:.2} cpu_multiplier={:.2} answered_per_run={:.1}/3 \
         accepted_runs={}/{} note=in-process_fanout_quic_excluded",
        runs(),
        max_tokens,
        baseline_wall,
        baseline_cpu,
        baseline_rate,
        verify_wall,
        verify_cpu,
        verify_rate,
        verify_wall / baseline_wall,
        verify_cpu / baseline_cpu,
        total_answered as f64 / runs() as f64,
        accepted,
        runs(),
    );
}
