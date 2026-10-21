//! Fundamental data types and the shared error taxonomy.

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
