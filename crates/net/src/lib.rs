//! Networking layer for Synapse.
//!
//! ## Future responsibilities
//!
//! * Own the `libp2p` / QUIC transport and peer discovery.
//! * Expose a transport-agnostic request/response API so that the inference
//!   engine never has to know about networking primitives.
//! * Handle peer scoring, routing and request forwarding across the swarm.
//!
//! The networking layer is strictly decoupled from the inference engine: it
//! depends on `synapse-core` for shared types only, never the other way
//! around.

use synapse_core::SynapseError;

/// Returns a human readable description of the configured transport.
///
/// The real transport is not wired up yet, so this is a placeholder that
/// allows callers to log a stable description of the layer's intent.
pub fn transport_description() -> &'static str {
    "libp2p/QUIC (not yet wired up)"
}

/// Placeholder that verifies the transport layer can be initialised.
///
/// It always returns an error because the real transport is not implemented.
pub fn init() -> Result<(), SynapseError> {
    Err(SynapseError::Network(
        "networking transport is not implemented yet".to_string(),
    ))
}
