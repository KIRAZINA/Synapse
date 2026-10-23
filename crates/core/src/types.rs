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
