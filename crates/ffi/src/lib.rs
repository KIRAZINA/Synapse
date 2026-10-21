//! Safe boundary around the `llama.cpp` C++ bindings.
//!
//! ## Future responsibilities
//!
//! * Declare the `extern "C"` surface of `llama.cpp` (model loading, context
//!   creation, token generation, teardown).
//! * Translate raw C return codes into [`synapse_core::SynapseError`].
//! * Guarantee that the FFI is *only* ever entered from a dedicated OS thread;
//!   the C++ runtime is not guaranteed to be thread safe.
//! * Own the lifetime of every pointer handed out by the C++ side and release
//!   it exactly once.
//!
//! No `unsafe` code and no actual bindings exist yet. This crate is a
//! placeholder so the rest of the workspace can already depend on it.

use synapse_core::SynapseError;

/// Placeholder for the FFI capability set that will be exposed to the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfiCapability {
    /// Model file loading and lifecycle.
    Model,
    /// Context creation and token generation.
    Inference,
}

impl FfiCapability {
    /// Returns a human readable name for the capability.
    pub fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Inference => "inference",
        }
    }
}

/// Reports whether the native `llama.cpp` bindings are compiled in yet.
///
/// The bindings are not implemented, so this always reports `false`.
pub fn bindings_available() -> bool {
    false
}

/// Placeholder entry point that will eventually forward to `llama.cpp`.
///
/// It always returns an error because the native layer is not implemented yet.
pub fn probe(capability: FfiCapability) -> Result<(), SynapseError> {
    Err(SynapseError::Engine(format!(
        "llama.cpp bindings are not implemented yet (capability: {})",
        capability.name()
    )))
}
