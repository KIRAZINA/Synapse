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
///
/// A request whose [`Determinism`] is not [`is_comparable`](Self::is_comparable)
/// is therefore **refused** by the verifier instead of being compared: flagging
/// an honest node for a difference that the configuration guaranteed is worse
/// than not verifying at all.
///
/// # Residual risk (documented, not hidden)
///
/// Even with temperature 0, an identical model, an identical quantization and
/// an identical seed, two nodes can diverge: floating point addition is not
/// associative, so a different thread count or a different CPU/GPU backend can
/// change the reduction order and flip a near-tie in the logits. Verification
/// detects *gross* misbehaviour (garbage output, a cheaper model, a truncated
/// generation), which is what an authenticated-but-untrusted peer can actually
/// get away with. It is not a bit-exact proof of the computation.
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
}

impl Determinism {
    /// Builds a deterministic (greedy) decoding configuration.
    pub fn greedy(model: impl Into<String>, quantization: impl Into<String>, seed: u64) -> Self {
        Self {
            temperature: 0.0,
            seed,
            model: model.into(),
            quantization: quantization.into(),
        }
    }

    /// Whether outputs produced under this configuration may be compared.
    ///
    /// All four properties have to be pinned: a comparison against an
    /// unspecified model or quantization is not a comparison, it is a guess.
    pub fn is_comparable(&self) -> bool {
        self.temperature == 0.0
            && !self.model.trim().is_empty()
            && !self.quantization.trim().is_empty()
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
