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
//! used from a single dedicated OS thread; the `synapse-engine` crate gives
//! every pool slot its own thread. The backend itself is process wide and is
//! initialised exactly once, no matter how many models or pools are created.

pub mod context;
pub mod model;

use std::sync::{Mutex, OnceLock};

use llama_cpp_2::LogOptions;
use synapse_core::SynapseError;
use tracing::info;

pub use context::LlamaContext;
pub use model::LlamaModel;

// Re-exported so that the rest of the workspace never has to depend on the
// raw bindings crate directly.
pub use llama_cpp_2::context::params::LlamaContextParams;
pub use llama_cpp_2::llama_backend::LlamaBackend;

/// The one and only backend of the process.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Serialises the call that actually initialises the backend, so that two
/// threads (or two pools) cannot race on the one-shot initialisation of
/// `llama.cpp`.
static BACKEND_INIT: Mutex<()> = Mutex::new(());

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
/// `llama.cpp` can only be initialised once per process, but a process may
/// create several models and pools (tests do). The first call initialises the
/// backend, every later call returns the same handle. The returned reference
/// lives for the rest of the process.
pub fn init_backend() -> Result<&'static LlamaBackend, SynapseError> {
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }

    let _guard = BACKEND_INIT.lock().map_err(|_| {
        SynapseError::Engine("the llama.cpp backend initialisation lock is poisoned".to_owned())
    })?;

    // A different thread may have initialised the backend while this one was
    // waiting for the lock.
    if let Some(backend) = BACKEND.get() {
        return Ok(backend);
    }

    // Native log lines are forwarded into `tracing` so that they end up in the
    // same log stream as the rest of the node. They are extremely verbose, so
    // they are muted unless `SYNAPSE_LLAMA_LOGS` is set to a value other than
    // `0`.
    let logs_enabled = std::env::var("SYNAPSE_LLAMA_LOGS")
        .map(|value| value != "0")
        .unwrap_or(false);
    llama_cpp_2::send_logs_to_tracing(LogOptions::default().with_logs_enabled(logs_enabled));

    // A prefix-cache hit has to produce the same output as a full prefill of
    // the same prompt. `llama.cpp`'s tiled K-quant matmul picks its K-chunking
    // from the number of tokens in the batch, so a token evaluated in a large
    // prefill batch and the same token evaluated in the shorter suffix of a
    // cache hit can differ in the last bits, which is enough to flip a sample.
    // The reference path is batch-shape invariant, and on the CPUs measured
    // here it costs nothing measurable for prefill, while the prefix cache
    // still makes a repeated prompt roughly 30x cheaper.
    //
    // The flag is read once by `ggml` when the first matrix multiplication
    // runs, so it has to be set before the backend is initialised.
    disable_tiled_matmul();

    let backend = LlamaBackend::init()
        .map_err(|err| SynapseError::Engine(format!("failed to initialise llama.cpp: {err}")))?;

    if BACKEND.set(backend).is_err() {
        return Err(SynapseError::Engine(
            "the llama.cpp backend was initialised twice".to_owned(),
        ));
    }

    info!("llama.cpp backend initialised");

    BACKEND.get().ok_or_else(|| {
        SynapseError::Engine(
            "the llama.cpp backend is not initialised after initialising it".to_owned(),
        )
    })
}

/// Disables the tiled K-quant matmul of `llama.cpp` (`GGML_CPU_TILED_MM=0`).
///
/// The tiled kernel chooses its K-chunking from the number of rows in the
/// batch, so the same token can be evaluated with a slightly different
/// floating-point accumulation order in a large prefill batch and in the
/// shorter suffix of a prefix-cache hit. The reference kernel is invariant to
/// the batch shape, which is what makes a cache hit byte-identical to a full
/// prefill. `llama.cpp` reads the variable once, on its first matrix
/// multiplication, so this has to run before the backend is initialised.
///
/// `std::env::set_var` is not enough on Windows: the C runtime keeps its own
/// copy of the environment and `getenv` in `llama.cpp` does not see the Rust
/// side, so the value is written through the CRT there.
fn disable_tiled_matmul() {
    #[cfg(windows)]
    {
        use std::os::raw::c_char;

        extern "C" {
            /// `_putenv` of the C runtime, which updates the environment that
            /// `getenv` (and therefore `llama.cpp`) reads.
            fn _putenv(setting: *const c_char) -> std::os::raw::c_int;
        }

        // A static, NUL terminated string: `_putenv` may keep the pointer.
        const SETTING: &[u8] = b"GGML_CPU_TILED_MM=0\0";

        // The pointer is valid for the life of the process.
        let result = unsafe { _putenv(SETTING.as_ptr().cast()) };
        if result != 0 {
            tracing::warn!(
                "failed to disable the tiled matmul of llama.cpp; prefix-cache hits may not be \
                 byte-identical to a full prefill"
            );
        }
    }

    #[cfg(not(windows))]
    std::env::set_var("GGML_CPU_TILED_MM", "0");
}
