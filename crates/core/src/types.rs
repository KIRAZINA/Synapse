//! Fundamental data types and the shared error taxonomy.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors surfaced across the whole Synapse workspace.
///
/// The variants intentionally map to the three architectural pillars so that a
/// failure can always be attributed to the layer that produced it.
#[derive(Debug, Error)]
pub enum SynapseError {
    /// The inference engine (or its FFI boundary) failed.
    #[error("engine error: {0}")]
    Engine(String),

    /// The networking layer failed.
    #[error("network error: {0}")]
    Network(String),

    /// Configuration was missing or invalid.
    #[error("configuration error: {0}")]
    Config(String),
}

/// A single event produced by a streaming generation.
///
/// The engine emits one `Token` event per sampled token, followed by exactly
/// one terminal event (`Done` or `Error`).
#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// A single generated token (detokenized text piece + raw token ID)
    Token { text: String, token_id: i32 },
    /// Generation completed successfully
    Done {
        total_tokens: usize,
        elapsed_ms: u64,
        stop_reason: StopReason,
    },
    /// Generation failed
    Error { message: String },
}

/// Why a generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// End-of-sequence token sampled
    Eos,
    /// Hit max_tokens limit
    MaxTokens,
    /// Receiver dropped (client disconnected)
    ClientAbort,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eos => f.write_str("eos"),
            Self::MaxTokens => f.write_str("max_tokens"),
            Self::ClientAbort => f.write_str("client_abort"),
        }
    }
}

/// The telemetry a node broadcasts about itself.
///
/// The payload is gossiped over the control plane, so that the clients can
/// pick a node that is neither busy nor dead.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NodeTelemetry {
    /// The libp2p peer id of the node.
    pub peer_id: String,
    /// The UDP port of the QUIC endpoint of the node.
    pub quic_port: u16,
    /// The libp2p addresses of the node, as multiaddr strings.
    ///
    /// They are part of the telemetry because a client has to be able to
    /// reach the node it selected, and the gossip does not carry the
    /// multiaddrs of the other peers.
    pub addresses: Vec<String>,
    /// The number of generations that are currently running.
    pub active_requests: u32,
    /// The number of generations the node accepts in parallel.
    pub max_capacity: u32,
    /// The models the node is able to serve.
    pub supported_models: Vec<String>,
    /// The time the telemetry was created, in unix milliseconds.
    pub timestamp_ms: u64,
}

/// A provider of streaming inference.
///
/// The networking layer depends on this trait instead of on the inference
/// engine, so the control plane, the data plane and the engine stay
/// decoupled: the engine implements the trait, the data plane only knows the
/// trait.
pub trait TokenStreamProvider: Send + Sync + 'static {
    /// Starts a streaming generation and returns the receiver of its events.
    ///
    /// The call may return before a slot is assigned: when the provider has a
    /// bounded admission queue, the request waits there and its events begin
    /// once a slot frees. A rejection before enqueue is returned as `Err`; a
    /// timeout after enqueue is a terminal [`TokenEvent::Error`]. Dropping the
    /// receiver aborts the generation, including a request that is still
    /// waiting in the queue.
    fn generate_streaming(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError>;

    /// Whether this provider can decode with strict argmax.
    ///
    /// Redundant verification requires it: a distribution sampler is only
    /// reproducible on the exact build that produced it, so two honest nodes
    /// would disagree and one would be punished for it. A provider that cannot
    /// decode greedily is not eligible for verification, and the orchestrator
    /// refuses to compare it rather than accusing it.
    fn supports_greedy(&self) -> bool {
        false
    }

    /// Starts a greedy (strict argmax) generation.
    ///
    /// The default implementation refuses. A provider that returns
    /// [`supports_greedy`](Self::supports_greedy) `true` must override this, so
    /// a provider can never claim greedy support and then silently sample.
    fn generate_streaming_greedy(
        &self,
        _prompt: String,
        _max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        Err(SynapseError::Engine(
            "this provider cannot decode greedily, so it cannot take part in verification"
                .to_owned(),
        ))
    }

    /// The number of generations that are currently running.
    fn active_requests(&self) -> u32;

    /// The number of generations the node accepts in parallel.
    fn max_capacity(&self) -> u32;
}

/// The decoding configuration that a comparison of outputs depends on.
///
/// # Why this type exists
///
/// Redundant verification compares the token sequences that several nodes
/// produced for the same prompt. That comparison is only meaningful when the
/// nodes were *supposed* to produce the same sequence. Two honest nodes will
/// still disagree if any of the following differ:
///
/// * **temperature** — anything above zero samples, so the same prompt yields
///   different tokens on the same node, let alone on two nodes.
/// * **model** — a different model has different weights.
/// * **quantization** — `Q4_K_M` and `Q8_0` of the same model are different
///   functions; they diverge quickly on long generations.
/// * **seed** — a seeded sampler is only reproducible when the seed matches.
/// * **threads** — floating point addition is not associative, so a different
///   thread count changes the reduction order and can flip a near-tie.
/// * **backend** — a CPU and a GPU kernel accumulate in a different order for
///   the same reason, and a different instruction set rounds differently.
///
/// A request whose [`Determinism`] is not [`is_comparable`](Self::is_comparable)
/// is therefore **refused** by the verifier instead of being compared: flagging
/// an honest node for a difference that the configuration guaranteed is worse
/// than not verifying at all.
///
/// # Residual risk (documented, not hidden)
///
/// Pinning threads and backend removes the *predictable* sources of numerical
/// divergence. It does not remove all of them: two machines with the same
/// thread count and the same backend can still differ if the underlying CPU
/// supports different vector widths (AVX2 vs AVX-512), if the OS schedules
/// differently, or if a different `llama.cpp` build was used. Verification
/// therefore detects *gross* misbehaviour (garbage output, a cheaper model, a
/// truncated generation), which is what an authenticated-but-untrusted peer can
/// actually get away with. It is not a bit-exact proof of the computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Determinism {
    /// The sampling temperature. Comparison requires exactly `0.0`.
    pub temperature: f32,
    /// The sampler seed. Comparison requires the participants to agree.
    pub seed: u64,
    /// The advertised name of the model.
    pub model: String,
    /// The quantization of the weights, for example `Q4_K_M`.
    pub quantization: String,
    /// The number of threads the generation runs on.
    ///
    /// Part of the numerical identity: a different thread count changes the
    /// order in which partial sums are reduced.
    pub n_threads: u32,
    /// The compute backend, for example `cpu`, `cuda` or `metal`.
    ///
    /// Part of the numerical identity for the same reason as `n_threads`.
    pub backend: String,
}

impl Determinism {
    /// Builds a fully pinned, greedy decoding configuration.
    ///
    /// Every field is required: a comparison against an unspecified thread
    /// count or backend is not a comparison, it is a guess.
    pub fn greedy(
        model: impl Into<String>,
        quantization: impl Into<String>,
        seed: u64,
        n_threads: u32,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            temperature: 0.0,
            seed,
            model: model.into(),
            quantization: quantization.into(),
            n_threads,
            backend: backend.into(),
        }
    }

    /// Whether outputs produced under this configuration may be compared.
    ///
    /// Every property has to be pinned: a comparison against an unspecified
    /// model, quantization, thread count or backend is not a comparison.
    pub fn is_comparable(&self) -> bool {
        self.temperature == 0.0
            && !self.model.trim().is_empty()
            && !self.quantization.trim().is_empty()
            && self.n_threads > 0
            && !self.backend.trim().is_empty()
    }

    /// Whether two configurations are close enough to compare outputs.
    ///
    /// This is stricter than [`is_comparable`](Self::is_comparable): both sides
    /// have to be comparable *and* identical, because a difference in any
    /// field makes the outputs legitimately different.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.is_comparable()
            && other.is_comparable()
            && self.temperature == other.temperature
            && self.seed == other.seed
            && self.model == other.model
            && self.quantization == other.quantization
            && self.n_threads == other.n_threads
            && self.backend == other.backend
    }
}

/// How a generation picks its next token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    /// Strict argmax: the most likely token, every time.
    ///
    /// This is the only mode a verified request may use. A distribution
    /// sampler, even with a fixed seed, is reproducible only on the exact
    /// build that produced it; argmax is reproducible across builds that agree
    /// on the logits.
    Greedy,
    /// A seeded distribution sampler, used by the normal (unverified) path.
    Seeded {
        /// The sampler seed.
        seed: u32,
    },
}

impl Sampling {
    /// Whether this mode is strict argmax.
    pub fn is_greedy(&self) -> bool {
        matches!(self, Self::Greedy)
    }
}

/// A single inference request handed to the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The raw user prompt text.
    pub text: String,
    /// Identifier of the session this prompt belongs to.
    pub session_id: String,
}

impl Prompt {
    /// Builds a prompt with a freshly generated session identifier.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            session_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Builds a prompt with an explicit session identifier.
    pub fn with_session(text: impl Into<String>, session_id: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            session_id: session_id.into(),
        }
    }
}
