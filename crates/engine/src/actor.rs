//! Actor model scaffolding for the inference engine.
//!
//! The actor owns a single dedicated OS thread. All native (`llama.cpp`) work
//! is performed on that thread, which keeps the FFI boundary strictly
//! isolated from the async runtime and from the networking layer. Commands are
//! sent over a channel and answered over a per-request response channel, so
//! the async side never blocks on the inference engine.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use flume::{Receiver, Sender};
use synapse_core::{SynapseError, TokenEvent, TokenStreamProvider};
use synapse_ffi::context::LlamaContext;
use synapse_ffi::model::LlamaModel;
use synapse_ffi::{LlamaBackend, LlamaContextParams};
use tracing::{error, info, warn};

/// Fallback number of threads when the parallelism of the host is unknown.
const DEFAULT_THREADS: i32 = 4;

/// Commands accepted by the inference actor.
#[derive(Debug)]
pub enum EngineCommand {
    /// Load a GGUF model.
    LoadModel {
        /// Path of the GGUF file to load.
        path: String,
        /// Channel the result is sent back on.
        response: Sender<Result<(), SynapseError>>,
    },
    /// Generate text for a prompt and stream it token by token.
    Generate {
        /// The prompt to complete.
        prompt: String,
        /// Maximum number of tokens to generate.
        max_tokens: usize,
        /// Channel the generated tokens are sent on.
        token_sender: Sender<TokenEvent>,
    },
    /// Stop the actor thread and release its resources.
    Shutdown,
}

/// Handle to the running inference actor.
#[derive(Debug)]
pub struct InferenceActor {
    sender: Sender<EngineCommand>,
    handle: Option<thread::JoinHandle<()>>,
    /// The number of generations that are currently running.
    active: Arc<AtomicU32>,
    /// The number of generations the node advertises that it can run.
    max_capacity: u32,
}

impl InferenceActor {
    /// Spawns the actor on a dedicated OS thread.
    ///
    /// `max_capacity` is what the telemetry of the node advertises, and
    /// `context_length` is the context window of every inference context. Both
    /// come from the node configuration.
    ///
    /// If the thread cannot be created the failure is logged and the returned
    /// handle is inert; the actor never panics the caller.
    pub fn spawn(max_capacity: u32, context_length: u32) -> Self {
        let (sender, receiver) = flume::unbounded::<EngineCommand>();

        // The counter of the running generations is shared with the telemetry
        // of the node, which is published from another thread.
        let active = Arc::new(AtomicU32::new(0));
        let thread_active = Arc::clone(&active);

        let handle = thread::Builder::new()
            .name("synapse-inference-actor".to_string())
            .spawn(move || run_actor(receiver, thread_active, context_length));

        match handle {
            Ok(handle) => Self {
                sender,
                handle: Some(handle),
                active,
                max_capacity,
            },
            Err(err) => {
                error!("failed to spawn inference actor thread: {err}");
                Self {
                    sender,
                    handle: None,
                    active,
                    max_capacity,
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

    /// Loads a model.
    ///
    /// The work happens on the actor thread; this call only waits for the
    /// result.
    pub async fn load_model(&self, path: String) -> Result<(), SynapseError> {
        let (response_sender, response_receiver) = flume::unbounded();

        self.send(EngineCommand::LoadModel {
            path,
            response: response_sender,
        })?;

        let result = response_receiver.recv_async().await.map_err(|err| {
            SynapseError::Engine(format!(
                "the inference actor dropped the model loading response: {err}"
            ))
        })?;

        result
    }

    /// Starts a streaming generation and returns the receiver of its events.
    ///
    /// The caller receives one [`TokenEvent::Token`] per generated token,
    /// followed by a terminal event. Dropping the receiver aborts the
    /// generation on the actor thread.
    pub async fn generate_streaming(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        let (token_sender, token_receiver) = flume::unbounded();

        self.sender
            .send_async(EngineCommand::Generate {
                prompt,
                max_tokens,
                token_sender,
            })
            .await
            .map_err(|err| {
                SynapseError::Engine(format!("inference actor channel is closed: {err}"))
            })?;

        Ok(token_receiver)
    }

    /// Reports whether the actor thread has already finished.
    pub fn is_finished(&self) -> bool {
        self.handle
            .as_ref()
            .map_or(true, thread::JoinHandle::is_finished)
    }

    /// Requests a graceful shutdown and waits for the actor thread to exit.
    pub fn shutdown(self) -> Result<(), SynapseError> {
        let Self {
            sender,
            handle,
            active: _,
            max_capacity: _,
        } = self;

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

impl TokenStreamProvider for InferenceActor {
    fn generate_streaming(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        let (token_sender, token_receiver) = flume::unbounded();

        self.send(EngineCommand::Generate {
            prompt,
            max_tokens,
            token_sender,
        })?;

        Ok(token_receiver)
    }

    fn active_requests(&self) -> u32 {
        self.active.load(Ordering::Relaxed)
    }

    fn max_capacity(&self) -> u32 {
        self.max_capacity
    }
}

/// Body of the actor thread: the command loop.
fn run_actor(
    receiver: Receiver<EngineCommand>,
    active_generations: Arc<AtomicU32>,
    context_length: u32,
) {
    info!("inference actor thread started");

    let backend = match synapse_ffi::init_backend() {
        Ok(backend) => Some(backend),
        Err(err) => {
            error!("failed to initialise the llama.cpp backend: {err}");
            None
        }
    };

    // The model stays resident for the whole life of the actor. The inference
    // context is created per request: it borrows the model, which keeps the
    // ownership relation between the two checkable by the compiler and gives
    // every request its own KV cache.
    let mut model: Option<Arc<LlamaModel>> = None;
    let context_params = context_params(context_length);

    let mut running = true;
    while running {
        match receiver.recv() {
            Ok(EngineCommand::LoadModel { path, response }) => {
                let result = load_model(backend.as_ref(), &path, &mut model);
                if let Err(err) = &result {
                    error!("model loading failed: {err}");
                }
                if response.send(result).is_err() {
                    warn!("the model loading request was dropped before it was answered");
                }
            }
            Ok(EngineCommand::Generate {
                prompt,
                max_tokens,
                token_sender,
            }) => {
                // The generation is counted while it runs, so that the
                // telemetry of the node tells the clients how busy it is.
                active_generations.fetch_add(1, Ordering::Relaxed);

                let result = generate_streaming(
                    model.as_ref(),
                    backend.as_ref(),
                    &context_params,
                    &prompt,
                    max_tokens,
                    &token_sender,
                );
                if let Err(err) = &result {
                    error!("generation failed: {err}");
                }

                active_generations.fetch_sub(1, Ordering::Relaxed);
            }
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

/// Parameters used for every inference context.
fn context_params(context_length: u32) -> LlamaContextParams {
    let n_threads = std::thread::available_parallelism()
        .map(|threads| i32::try_from(threads.get()).unwrap_or(i32::MAX))
        .unwrap_or(DEFAULT_THREADS);
    let n_ctx = NonZeroU32::new(context_length).unwrap_or(NonZeroU32::MIN);

    LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
}

/// Loads a model and replaces the previously loaded one.
fn load_model(
    backend: Option<&LlamaBackend>,
    path: &str,
    model: &mut Option<Arc<LlamaModel>>,
) -> Result<(), SynapseError> {
    let backend = backend
        .ok_or_else(|| SynapseError::Engine("the llama.cpp backend is not available".to_owned()))?;

    let loaded = Arc::new(LlamaModel::load_from_file(backend, path)?);
    *model = Some(loaded);

    info!(path, "model is ready");
    Ok(())
}

/// Runs a streaming generation request against the loaded model.
fn generate_streaming(
    model: Option<&Arc<LlamaModel>>,
    backend: Option<&LlamaBackend>,
    context_params: &LlamaContextParams,
    prompt: &str,
    max_tokens: usize,
    token_sender: &Sender<TokenEvent>,
) -> Result<(), SynapseError> {
    let (model, backend) = match (model, backend) {
        (Some(model), Some(backend)) => (model, backend),
        _ => {
            let message = "no model has been loaded yet".to_owned();
            let _ = token_sender.send(TokenEvent::Error {
                message: message.clone(),
            });
            return Err(SynapseError::Engine(message));
        }
    };

    // The context borrows the model for the duration of the request.
    let mut context: LlamaContext<'_> = model.new_context(backend, context_params.clone())?;

    context.generate_streaming(prompt, max_tokens, token_sender.clone())
}
