//! Safe wrapper around the `llama.cpp` C++ bindings.
//!
//! This crate is the **quarantine zone** for `unsafe` code: it is the only
//! place in the Synapse workspace that is allowed to touch raw pointers, and
//! it never lets a pointer escape. Everything the rest of the workspace sees
//! is a plain Rust type:
//!
//! * [`LlamaModel`] owns a loaded GGUF model.
//! * [`LlamaContext`] owns an inference context (KV cache + context window)
//!   that borrows from the model it was created from.
//!
//! ## Threading
//!
//! `llama.cpp` is not thread safe. Every type in this crate must therefore be
//! used from a single dedicated OS thread; the `synapse-engine` crate owns
//! that thread (see `InferenceActor`).

pub mod context;
pub mod model;

use llama_cpp_2::LogOptions;
use synapse_core::SynapseError;
use tracing::info;

pub use context::LlamaContext;
pub use model::LlamaModel;

// Re-exported so that the rest of the workspace never has to depend on the
// raw bindings crate directly.
pub use llama_cpp_2::context::params::LlamaContextParams;
pub use llama_cpp_2::llama_backend::LlamaBackend;

// Declares the `ggml` static archives that back the `llama.cpp` bindings.
//
// `llama-cpp-sys-2` compiles the `llama` translation units into its rlib, but
// the `ggml` translation units are built by CMake into separate archives.
// Rust only picks up native libraries that are declared with a `#[link]`
// attribute, so declaring them here is what puts them on the link line of
// every binary that depends on this crate. No build script logic has to be
// repeated in the downstream crates.
//
// `+whole-archive` is required: the archives are pulled into the link before
// the Rust crate that references the `ggml` symbols, so the linker cannot
// resolve them on demand.
//
// The archives are mirrored under their conventional `lib` prefixed names by
// `build.rs`, because `llama.cpp` removes that prefix on MinGW.
//
// Every archive needs its own extern block: `#[link]` may only appear once
// per block.
#[link(name = "ggml", kind = "static", modifiers = "+whole-archive")]
extern "C" {}

#[link(name = "ggml-base", kind = "static", modifiers = "+whole-archive")]
extern "C" {}

#[link(name = "ggml-cpu", kind = "static", modifiers = "+whole-archive")]
extern "C" {}

/// Initialises the `llama.cpp` backend.
///
/// The backend can only be initialised once per process; the returned handle
/// must be kept alive for as long as models and contexts exist.
pub fn init_backend() -> Result<LlamaBackend, SynapseError> {
    // Native log lines are forwarded into `tracing` so that they end up in the
    // same log stream as the rest of the node. They are extremely verbose, so
    // they are muted unless `SYNAPSE_LLAMA_LOGS` is set to a value other than
    // `0`.
    let logs_enabled = std::env::var("SYNAPSE_LLAMA_LOGS")
        .map(|value| value != "0")
        .unwrap_or(false);
    llama_cpp_2::send_logs_to_tracing(LogOptions::default().with_logs_enabled(logs_enabled));

    let backend = LlamaBackend::init()
        .map_err(|err| SynapseError::Engine(format!("failed to initialise llama.cpp: {err}")))?;

    info!("llama.cpp backend initialised");
    Ok(backend)
}
