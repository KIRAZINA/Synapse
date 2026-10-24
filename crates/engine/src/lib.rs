//! Inference engine for Synapse.
//!
//! The engine owns the *orchestration* of inference work. The actual native
//! calls live behind [`synapse_ffi`], and they are executed on dedicated OS
//! threads (one per inference slot) so that no `unsafe` FFI call ever happens
//! inside an async Tokio task.
//!
//! The entry point is [`InferencePool`]: it loads the model once, pre-creates
//! one context per configured slot and hands those slots out to generations.
//! `node.max_capacity` is therefore the real number of concurrent generations a
//! node can serve, not an advertisement.

pub mod pool;

pub use pool::{AtCapacity, InferencePool, PoolDiagnostics, SlotGuard};
