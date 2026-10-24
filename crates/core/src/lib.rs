//! Core shared types for the Synapse network.
//!
//! This crate is dependency-light on purpose: it holds the error taxonomy and
//! the data structures that cross crate boundaries so that the networking
//! layer and the inference engine never need to depend on each other.

pub mod config;
pub mod types;

pub use config::{
    BootstrapConfig, Config, ConfigError, MetricsConfig, ModelConfig, NodeConfig, QuicConfig,
    SecurityConfig, VerificationConfig, DEFAULT_CONFIG_FILE,
};
pub use types::{
    Determinism, NodeTelemetry, Prompt, Sampling, StopReason, SynapseError, TokenEvent,
    TokenStreamProvider,
};
