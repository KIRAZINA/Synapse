//! Inference engine for Synapse.
//!
//! The engine owns the *orchestration* of inference work. The actual native
//! calls live behind [`synapse_ffi`], and they are executed on a dedicated OS
//! thread (the actor) so that no `unsafe` FFI call ever happens inside an
//! async Tokio task.

pub mod actor;

pub use actor::{EngineCommand, InferenceActor};
