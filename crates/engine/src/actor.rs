//! Actor model scaffolding for the inference engine.
//!
//! The actor owns a single dedicated OS thread. All native (`llama.cpp`) work
//! will eventually be performed on that thread, which keeps the FFI boundary
//! strictly isolated from the async runtime and from the networking layer.

use std::thread;

use flume::{Receiver, Sender};
use synapse_core::SynapseError;
use tracing::{error, info};

/// Commands accepted by the inference actor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineCommand {
    /// Stop the actor thread and release its resources.
    Shutdown,
}

/// Handle to the running inference actor.
#[derive(Debug)]
pub struct InferenceActor {
    sender: Sender<EngineCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl InferenceActor {
    /// Spawns the actor on a dedicated OS thread.
    ///
    /// If the thread cannot be created the failure is logged and the returned
    /// handle is inert; the actor never panics the caller.
    pub fn spawn() -> Self {
        let (sender, receiver) = flume::unbounded::<EngineCommand>();

        let handle = thread::Builder::new()
            .name("synapse-inference-actor".to_string())
            .spawn(move || run_actor(receiver));

        match handle {
            Ok(handle) => Self {
                sender,
                handle: Some(handle),
            },
            Err(err) => {
                error!("failed to spawn inference actor thread: {err}");
                Self {
                    sender,
                    handle: None,
                }
            }
        }
    }

    /// Sends a command to the actor thread.
    pub fn send(&self, command: EngineCommand) -> Result<(), SynapseError> {
        self.sender.send(command).map_err(|err| {
            SynapseError::Engine(format!("inference actor channel is closed: {err}"))
        })
    }

    /// Reports whether the actor thread has already finished.
    pub fn is_finished(&self) -> bool {
        self.handle
            .as_ref()
            .map_or(true, thread::JoinHandle::is_finished)
    }

    /// Requests a graceful shutdown and waits for the actor thread to exit.
    pub fn shutdown(self) -> Result<(), SynapseError> {
        let Self { sender, handle } = self;

        sender.send(EngineCommand::Shutdown).map_err(|err| {
            SynapseError::Engine(format!("inference actor channel is closed: {err}"))
        })?;

        if let Some(handle) = handle {
            handle
                .join()
                .map_err(|_| SynapseError::Engine("inference actor thread panicked".to_string()))?;
        }

        Ok(())
    }
}

/// Body of the actor thread: a simple command loop.
fn run_actor(receiver: Receiver<EngineCommand>) {
    info!("inference actor thread started");

    // The loop stays alive until a shutdown command is received or the command
    // channel is closed (which happens when every sender is dropped).
    let mut running = true;
    while running {
        match receiver.recv() {
            Ok(EngineCommand::Shutdown) => {
                info!("inference actor received shutdown command");
                running = false;
            }
            Err(_) => {
                info!("inference actor command channel closed");
                running = false;
            }
        }
    }

    info!("inference actor thread exiting");
}
