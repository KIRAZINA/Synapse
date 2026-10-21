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
