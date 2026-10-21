//! Core shared types for the Synapse network.
//!
//! This crate is dependency-light on purpose: it holds the error taxonomy and
//! the data structures that cross crate boundaries so that the networking
//! layer and the inference engine never need to depend on each other.

pub mod types;

pub use types::{Prompt, SynapseError};
