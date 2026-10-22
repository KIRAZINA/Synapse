//! Bounded pool of inference slots.
//!
//! The engine used to own a single context on a single thread, so
//! `node.max_capacity` was an advertisement rather than a capability. The pool
//! replaces that actor with a fixed set of *slots*: every slot owns one
//! `llama.cpp` context, runs on its own dedicated OS thread and serves one
//! generation at a time. The model itself is loaded once and shared, so a slot
//! only costs its KV cache.
//!
//! # Acquiring a slot
//!
//! A slot is a value in a `flume` channel. Acquiring is a non-blocking
//! [`flume::Receiver::try_recv`]; when the channel is empty the request is
//! rejected immediately with [`AtCapacity`]. There is no request queue: a node
//! that cannot serve a request says so instead of making the client wait
//! behind generations it did not ask for.
//!
//! The acquired slot is handed to the slot thread inside the command. The
//! slot thread drops the [`SlotGuard`] after the generation finishes, after the
//! client aborts and after a failure, and dropping the guard is what puts the
//! slot back into the free channel and decrements the busy counter. Because
//! the guard is owned by the thread that runs the generation, a panic unwinds
//! through the same `Drop` and cannot leak the slot either.
//!
//! # Context hygiene
//!
//! A context is reused by many requests. Every generation is followed by
//! [`LlamaContext::clear_cache`], so the next request starts with an empty KV
//! cache and never attends to tokens that belong to a previous client.

use std::num::{NonZeroU32, NonZeroUsize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use flume::{Receiver, Sender};
use synapse_core::{SynapseError, TokenEvent, TokenStreamProvider};
use synapse_ffi::model::LlamaModel;
use synapse_ffi::{LlamaBackend, LlamaContextParams};
use tracing::{error, info, warn};

/// Fallback number of threads when the parallelism of the host is unknown.
const DEFAULT_THREADS: usize = 4;

/// One mebibyte, used for the KV cache log lines.
const MIB: u64 = 1024 * 1024;

/// A slot that was checked out of the pool while it serves a request.
///
/// Dropping the guard releases the slot: it is put back into the free channel
/// and the busy counter is decremented. The guard lives on the slot thread
/// while a generation runs, so every exit path (success, abort, engine error,
/// panic) goes through this `Drop`.
#[derive(Debug)]
pub struct SlotGuard {
    /// The slot, taken by `Drop`. `None` means the guard already released it.
    slot: Option<Slot>,
    /// The free channel the slot is returned to.
    release: Sender<Slot>,
    /// The number of busy slots, shared with the telemetry of the pool.
    active: Arc<AtomicUsize>,
}

impl SlotGuard {
    /// Wraps a checked out slot.
    fn new(slot: Slot, release: Sender<Slot>, active: Arc<AtomicUsize>) -> Self {
        Self {
            slot: Some(slot),
            release,
            active,
        }
    }

    /// The index of the slot that is held, if it was not released yet.
    ///
    /// The index is stable for the life of the pool and only exists for logs
    /// and tests.
    pub fn index(&self) -> Option<usize> {
        self.slot.as_ref().map(|slot| slot.index)
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let Some(slot) = self.slot.take() else {
            return;
        };

        // The slot is returned before the busy counter is decremented: once
        // `active()` reaches zero, the slot is guaranteed to be in the free
        // channel and a caller that observed an idle pool is never rejected
        // because of a slot that is still on its way back.
        if self.release.send(slot).is_err() {
            // The pool is shutting down, so there is no free list left to
            // return the slot to. The thread is about to exit anyway.
            warn!("the inference pool is gone, the slot was not returned");
        }

        let busy = self
            .active
            .fetch_sub(1, Ordering::AcqRel)
            .saturating_sub(1);
        metrics::gauge!("synapse_active_generations").set(busy as f64);
    }
}

/// The error returned by [`InferencePool::try_acquire`] when every slot is
/// busy.
///
/// This is a normal condition, not a fault: the caller is supposed to report
/// it to the client instead of waiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtCapacity {
    /// The number of slots the pool has.
    pub capacity: usize,
}

impl std::fmt::Display for AtCapacity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the node is at capacity: all {} inference slots are busy",
            self.capacity
        )
    }
}

impl std::error::Error for AtCapacity {}

/// A handle to a single inference slot thread.
#[derive(Debug, Clone)]
struct Slot {
    /// Stable index of the slot, used in logs.
    index: usize,
    /// Channel of the slot thread.
    commands: Sender<SlotCommand>,
}

/// A command accepted by a slot thread.
#[derive(Debug)]
enum SlotCommand {
    /// Run one generation and stream its events.
    Generate {
        /// The prompt to complete.
        prompt: String,
        /// Maximum number of tokens to generate.
        max_tokens: usize,
        /// Channel the generated events are sent on.
        token_sender: Sender<TokenEvent>,
        /// The slot that serves this request. The guard is moved into the
        /// command so the slot thread owns it for exactly the duration of the
        /// generation and returns the slot when the command is done.
        guard: SlotGuard,
    },
    /// Stop the slot thread and release its context.
    Shutdown,
}

/// A fixed set of slots that share one model.
///
/// The pool implements [`TokenStreamProvider`], so the networking layer is
/// unchanged: it asks for a stream and gets one, or gets an error when the
/// node is at capacity.
#[derive(Debug)]
pub struct InferencePool {
    /// Slots that are ready to serve a request.
    free: Receiver<Slot>,
    /// The other end of `free`; used to return slots from the guards.
    release: Sender<Slot>,
    /// The number of slots that are currently serving a request.
    active: Arc<AtomicUsize>,
    /// The number of slots in the pool.
    capacity: usize,
    /// One command channel per slot, used to stop the threads at shutdown.
    slot_commands: Vec<Sender<SlotCommand>>,
    /// The threads of the slot loops.
    handles: Vec<thread::JoinHandle<()>>,
}

impl InferencePool {
    /// Loads a model and creates `max_capacity` slots around it.
    ///
    /// The call blocks until every slot has created its context: a node that
    /// cannot create all of its slots fails at startup instead of advertising
    /// a capacity it cannot serve. The work happens on the calling thread; the
    /// node calls this before it starts the network.
    pub fn load(
        path: &str,
        max_capacity: u32,
        context_length: u32,
    ) -> Result<Self, SynapseError> {
        let capacity = usize::try_from(max_capacity).map_err(|_| {
            SynapseError::Engine("node.max_capacity does not fit into a usize".to_owned())
        })?;

        if capacity == 0 {
            return Err(SynapseError::Engine(
                "node.max_capacity has to be at least 1".to_owned(),
            ));
        }

        let backend = synapse_ffi::init_backend()?;
        let model = Arc::new(LlamaModel::load_from_file(backend, path)?);

        let kv_bytes_per_slot = model.estimated_kv_cache_bytes(context_length);
        let kv_mib_per_slot = kv_bytes_per_slot / MIB;
        let kv_mib_total = kv_bytes_per_slot.saturating_mul(capacity as u64) / MIB;

        info!(
            slots = capacity,
            n_ctx = context_length,
            kv_cache_mib_per_slot = kv_mib_per_slot,
            kv_cache_mib_total = kv_mib_total,
            "starting the inference pool"
        );

        let (release, free) = flume::unbounded::<Slot>();
        let (ready_sender, ready_receiver) = flume::unbounded::<Result<(), SynapseError>>();

        // The busy counter is shared between the guards (which are owned by
        // the slot threads) and the telemetry of the pool.
        let active = Arc::new(AtomicUsize::new(0));

        let params = context_params(context_length, capacity);
        let mut slot_commands = Vec::with_capacity(capacity);
        let mut handles = Vec::with_capacity(capacity);

        for index in 0..capacity {
            let (command_sender, command_receiver) = flume::unbounded::<SlotCommand>();
            slot_commands.push(command_sender.clone());

            let startup = SlotStartup {
                index,
                model: Arc::clone(&model),
                backend,
                params: params.clone(),
                commands: command_receiver,
                free: release.clone(),
                ready: ready_sender.clone(),
                own_sender: command_sender,
            };

            let builder = thread::Builder::new().name(format!("synapse-inference-slot-{index}"));

            match builder.spawn(move || startup.run()) {
                Ok(handle) => handles.push(handle),
                Err(err) => {
                    stop_slots(&slot_commands, handles);
                    return Err(SynapseError::Engine(format!(
                        "failed to spawn inference slot {index}: {err}"
                    )));
                }
            }
        }

        // Every thread owns a clone of the sender; dropping this one makes the
        // readiness loop observe a disconnect if a thread dies before it
        // reports in.
        drop(ready_sender);

        let mut failure = None;
        for _ in 0..capacity {
            match ready_receiver.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    failure = Some(err.to_string());
                    break;
                }
                Err(err) => {
                    failure = Some(format!(
                        "an inference slot stopped before it was ready: {err}"
                    ));
                    break;
                }
            }
        }

        if let Some(message) = failure {
            stop_slots(&slot_commands, handles);
            return Err(SynapseError::Engine(format!(
                "failed to start the inference pool: {message}"
            )));
        }

        metrics::gauge!("synapse_active_generations").set(0.0);
        metrics::gauge!("synapse_capacity_total").set(capacity as f64);

        info!(slots = capacity, "inference pool is ready");

        Ok(Self {
            free,
            release,
            active,
            capacity,
            slot_commands,
            handles,
        })
    }

    /// The number of slots in the pool.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of slots that are currently serving a request.
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Takes a free slot without waiting.
    ///
    /// Returns [`AtCapacity`] when every slot is busy. The request is never
    /// queued: the caller decides whether to reject it or to try another node.
    pub fn try_acquire(&self) -> Result<SlotGuard, AtCapacity> {
        match self.free.try_recv() {
            Ok(slot) => {
                let busy = self.active.fetch_add(1, Ordering::AcqRel) + 1;
                metrics::gauge!("synapse_active_generations").set(busy as f64);
                Ok(SlotGuard::new(
                    slot,
                    self.release.clone(),
                    Arc::clone(&self.active),
                ))
            }
            Err(_) => Err(AtCapacity {
                capacity: self.capacity,
            }),
        }
    }

    /// Starts a generation on a free slot.
    ///
    /// The slot guard travels with the command: the slot thread releases the
    /// slot when the generation ends, whatever the reason.
    fn start_generation(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<Receiver<TokenEvent>, SynapseError> {
        let guard = self
            .try_acquire()
            .map_err(|err| SynapseError::Engine(err.to_string()))?;

        let command_sender = guard
            .slot
            .as_ref()
            .map(|slot| slot.commands.clone())
            .ok_or_else(|| {
                SynapseError::Engine("the acquired inference slot was already released".to_owned())
            })?;

        let (token_sender, token_receiver) = flume::unbounded::<TokenEvent>();

        let command = SlotCommand::Generate {
            prompt,
            max_tokens,
            token_sender,
            guard,
        };

        if command_sender.send(command).is_err() {
            // Dropping the failed command drops the guard, which returns the
            // slot to the free list.
            return Err(SynapseError::Engine(
                "the inference slot thread is gone".to_owned(),
            ));
        }

        Ok(token_receiver)
    }

    /// Stops every slot thread and waits for them to exit.
    ///
    /// A slot that is serving a request finishes that request first; dropping
    /// the receiver of the request aborts it instead.
    pub fn shutdown(self) -> Result<(), SynapseError> {
        let Self {
            slot_commands,
            handles,
            ..
        } = self;

        for sender in &slot_commands {
            let _ = sender.send(SlotCommand::Shutdown);
        }

        let mut panicked = false;
        for handle in handles {
            if handle.join().is_err() {
                panicked = true;
            }
        }

        if panicked {
            return Err(SynapseError::Engine(
                "an inference slot thread panicked".to_owned(),
            ));
        }

        Ok(())
    }
}

impl TokenStreamProvider for InferencePool {
    fn generate_streaming(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<Receiver<TokenEvent>, SynapseError> {
        self.start_generation(prompt, max_tokens)
    }

    fn active_requests(&self) -> u32 {
        u32::try_from(self.active()).unwrap_or(u32::MAX)
    }

    fn max_capacity(&self) -> u32 {
        u32::try_from(self.capacity).unwrap_or(u32::MAX)
    }
}

/// Sends the shutdown command to the slots that were created and joins them.
///
/// Used on the error paths of [`InferencePool::load`], where partially
/// initialised slots must not leak threads.
fn stop_slots(senders: &[Sender<SlotCommand>], handles: Vec<thread::JoinHandle<()>>) {
    for sender in senders {
        let _ = sender.send(SlotCommand::Shutdown);
    }

    for handle in handles {
        let _ = handle.join();
    }
}

/// Everything a slot thread needs to start.
struct SlotStartup {
    /// Stable index of the slot.
    index: usize,
    /// The shared model.
    model: Arc<LlamaModel>,
    /// The process wide backend.
    backend: &'static LlamaBackend,
    /// Parameters every context of the pool is created with.
    params: LlamaContextParams,
    /// The commands this slot accepts.
    commands: Receiver<SlotCommand>,
    /// Where the slot announces that it is ready.
    free: Sender<Slot>,
    /// Where the thread reports the outcome of its context creation.
    ready: Sender<Result<(), SynapseError>>,
    /// A sender to its own command channel, stored in the ready [`Slot`].
    own_sender: Sender<SlotCommand>,
}

impl SlotStartup {
    /// Body of a slot thread: create the context, then serve commands.
    fn run(self) {
        let Self {
            index,
            model,
            backend,
            params,
            commands,
            free,
            ready,
            own_sender,
        } = self;

        // The context borrows the model, so both live on this thread. The
        // context is declared after the model and is therefore dropped first,
        // which is the order `llama.cpp` requires.
        let mut context = match model.new_context(backend, params) {
            Ok(context) => context,
            Err(err) => {
                error!(
                    slot = index,
                    "failed to create the inference context: {err}"
                );
                let _ = ready.send(Err(err));
                return;
            }
        };

        if free
            .send(Slot {
                index,
                commands: own_sender,
            })
            .is_err()
        {
            error!(slot = index, "the inference pool is gone, stopping the slot");
            return;
        }

        if ready.send(Ok(())).is_err() {
            // The pool is gone; the slot is already in the free list, which is
            // dropped with the pool.
            return;
        }

        info!(slot = index, n_ctx = context.n_ctx(), "inference slot is ready");

        while let Ok(command) = commands.recv() {
            match command {
                SlotCommand::Generate {
                    prompt,
                    max_tokens,
                    token_sender,
                    guard,
                } => {
                    // A panic in the generation must not take the slot out of
                    // the pool: the guard is dropped by the unwind, and the
                    // cache is cleared below before the guard is released.
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        context.generate_streaming(&prompt, max_tokens, token_sender)
                    }));

                    // Clear on every path, before the slot can be handed to
                    // the next request.
                    context.clear_cache();

                    match outcome {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            error!(slot = index, "generation failed: {err}");
                        }
                        Err(_) => {
                            error!(slot = index, "the generation panicked");
                        }
                    }

                    // Releases the slot and decrements the busy counter.
                    drop(guard);
                }
                SlotCommand::Shutdown => {
                    info!(slot = index, "inference slot received shutdown command");
                    break;
                }
            }
        }

        info!(slot = index, "inference slot is exiting");
    }
}

/// Parameters used for every inference context of the pool.
///
/// The threads of the host are divided between the slots: two contexts that
/// each use every core would oversubscribe the machine and run slower than one
/// context at a time. Every slot gets at least one thread.
fn context_params(context_length: u32, slots: usize) -> LlamaContextParams {
    let available = std::thread::available_parallelism()
        .map(NonZeroUsize::get)
        .unwrap_or(DEFAULT_THREADS);
    let threads_per_slot = (available / slots.max(1)).max(1);
    let n_threads = i32::try_from(threads_per_slot).unwrap_or(i32::MAX);
    let n_ctx = NonZeroU32::new(context_length).unwrap_or(NonZeroU32::MIN);

    LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropping_a_guard_returns_the_slot() {
        let (release, free) = flume::unbounded::<Slot>();
        let (commands, _command_receiver) = flume::unbounded::<SlotCommand>();
        let active = Arc::new(AtomicUsize::new(0));

        if release
            .send(Slot {
                index: 7,
                commands,
            })
            .is_err()
        {
            panic!("the slot could not be queued");
        }

        let slot = match free.try_recv() {
            Ok(slot) => slot,
            Err(err) => panic!("the slot was not queued: {err}"),
        };

        let guard = SlotGuard::new(slot, release, Arc::clone(&active));
        // `try_acquire` counts the slot as busy before it hands the guard out.
        active.fetch_add(1, Ordering::SeqCst);
        assert_eq!(guard.index(), Some(7));
        assert_eq!(active.load(Ordering::SeqCst), 1);

        drop(guard);

        assert_eq!(active.load(Ordering::SeqCst), 0);
        match free.try_recv() {
            Ok(slot) => assert_eq!(slot.index, 7),
            Err(err) => panic!("the slot was not returned to the pool: {err}"),
        }
    }

    #[test]
    fn releasing_a_guard_twice_is_harmless() {
        let (release, free) = flume::unbounded::<Slot>();
        let (commands, _command_receiver) = flume::unbounded::<SlotCommand>();
        let active = Arc::new(AtomicUsize::new(0));

        let guard = SlotGuard::new(
            Slot {
                index: 0,
                commands,
            },
            release,
            Arc::clone(&active),
        );
        // `try_acquire` counts the slot as busy before it hands the guard out.
        active.fetch_add(1, Ordering::SeqCst);

        assert_eq!(active.load(Ordering::SeqCst), 1);
        drop(guard);
        assert_eq!(active.load(Ordering::SeqCst), 0);

        // Only one slot may have been returned.
        assert!(free.try_recv().is_ok());
        assert!(free.try_recv().is_err());
    }
}
