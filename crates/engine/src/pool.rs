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
//! The free slots live in an inspectable registry guarded by a [`Mutex`], not
//! in a channel. Acquiring is a non-blocking selection under that lock; when no
//! slot is free the request is rejected immediately with [`AtCapacity`]. There
//! is no request queue: a node that cannot serve a request says so instead of
//! making the client wait behind generations it did not ask for.
//!
//! The lock is held only while the registry is read or updated: selection,
//! release and retirement. It is never held while a model runs. A generation
//! may take minutes; the registry critical section takes microseconds.
//!
//! The acquired slot is represented by a [`SlotGuard`] that travels to the slot
//! thread inside the command. The slot thread drops the guard after the
//! generation finishes, after the client aborts and after a failure, and
//! dropping the guard is what marks the slot free again and decrements the busy
//! counter. Because the guard is owned by the thread that runs the generation,
//! a panic unwinds through the same `Drop` and cannot leak the slot either.
//!
//! # Cache affinity
//!
//! With `cache.enable_affinity_scheduling` enabled (the default), the pool
//! tokenizes the prompt on the request path and scores every *free* slot by the
//! longest common prefix (LCP) of its cached prompt tokens and the incoming
//! prompt. It takes the best free slot, breaking ties by free-list order. A
//! busy slot is never waited for: affinity is opportunistic, and a full pool is
//! still a rejection. When the option is disabled, selection is the Phase 8
//! FIFO of free slots and no scoring happens.
//!
//! # Context hygiene
//!
//! A context is reused by many requests. Before the KV cache was valuable the
//! pool simply cleared it after every generation. Phase 8 keeps the prompt
//! tokens instead: after a generation the completion tokens are trimmed away,
//! and the slot remembers the token IDs that are still resident in its cache.
//! The next request reuses the longest common prefix of its own prompt with
//! that cache, so a shared system prompt is only prefilled once per slot.
//!
//! The cache is strictly per slot: slots never share KV state, there is no
//! radix tree and no eviction. Correctness always wins over a hit: any error
//! while retaining the prefix, any failed trim and every panic falls back to a
//! full [`LlamaContext::clear_cache`] and a prompt prefill from zero. A panic
//! is caught around the whole generation; the slot's cache is cleared before
//! the guard is released, and a slot whose context cannot even be cleared is
//! retired instead of being handed to the next client.

use std::collections::VecDeque;
use std::num::{NonZeroU32, NonZeroUsize};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;

use flume::{Receiver, Sender};
use synapse_core::{SynapseError, TokenEvent, TokenStreamProvider};
use synapse_ffi::model::LlamaModel;
use synapse_ffi::{LlamaBackend, LlamaContext, LlamaContextParams};
use tracing::{error, info, warn};

/// Test hook: forces the next generation to panic after its completion tokens
/// were decoded and before the slot is trimmed.
///
/// The integration tests set this to prove that a panic cannot leak a dirty KV
/// cache into the next request. The node never sets it.
#[doc(hidden)]
pub static FORCE_PANIC_AFTER_GENERATION: AtomicBool = AtomicBool::new(false);

/// Test hook: retires the slot that served the next generation, after that
/// generation completed successfully.
///
/// A healthy context never retires in production, so the capacity accounting
/// cannot be exercised through the public API otherwise. The integration tests
/// use this to prove that a retired slot leaves the live capacity and the
/// advertised telemetry. The node never sets it.
#[doc(hidden)]
pub static FORCE_RETIRE_AFTER_GENERATION: AtomicBool = AtomicBool::new(false);

/// Fallback number of threads when the parallelism of the host is unknown.
const DEFAULT_THREADS: usize = 4;

/// One mebibyte, used for the KV cache log lines.
const MIB: u64 = 1024 * 1024;

/// Whether a slot can serve a request right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotStatus {
    /// Ready to be selected.
    Free,
    /// Serving a request. The guard owns it until the generation ends.
    Busy,
    /// Permanently out of service because its context could not be cleaned.
    Retired,
}

/// The per-slot bookkeeping that selection and release need.
///
/// The cached prompt lives here, not on the slot thread, so selection can read
/// it under the registry lock without reaching into another thread. The slot
/// thread updates it after a successful trim, while the slot is still busy.
#[derive(Debug)]
struct SlotEntry {
    /// The command channel of the slot thread.
    commands: Sender<SlotCommand>,
    /// Current state of the slot.
    status: SlotStatus,
    /// Prompt token IDs that are resident in the slot's KV cache.
    cached_prefix: Vec<i32>,
}

/// The inspectable slot registry.
///
/// `free` is a FIFO of free slot indices. Keeping it as a queue rather than
/// scanning the vector preserves the Phase 8 order when affinity is disabled:
/// the slot that was released longest ago is selected first. With affinity
/// enabled the queue is scanned and the best match wins; on a tie the earlier
/// queue entry wins, which is deterministic.
#[derive(Debug)]
struct SlotRegistry {
    /// One entry per slot, indexed by the stable slot index.
    slots: Vec<SlotEntry>,
    /// Indices of the slots that are currently free, oldest first.
    free: VecDeque<usize>,
    /// Number of slots that were not retired. This is the live capacity.
    live: usize,
}

impl SlotRegistry {
    /// Selects and removes a free slot.
    ///
    /// With `affinity` disabled this is the Phase 8 behaviour: the oldest free
    /// slot (`free.pop_front()`), still reporting that slot's real LCP so the
    /// generation can reuse its cache. With affinity enabled every free slot
    /// is scored by [`longest_common_prefix`] and the highest score wins; ties
    /// keep the earliest queue entry. `shared` is the raw shared prefix length,
    /// which can be longer than the reusable prefix because the last prompt
    /// token always has to be evaluated again.
    ///
    /// The registry lock is held by the caller. The returned slot is still
    /// marked `Free`; the caller changes it to `Busy` after checking it.
    fn select(&mut self, tokens: &[i32], affinity: bool) -> Option<(usize, usize)> {
        // Defensive: a retired or busy slot must never be in the free queue.
        // A stale entry would otherwise hand out the same slot twice. The
        // invariant is maintained by `Drop` and `retire`; this sweep only
        // protects against a future bug, it is not part of the normal path.
        let slots = &self.slots;
        self.free.retain(|index| {
            slots
                .get(*index)
                .is_some_and(|entry| entry.status == SlotStatus::Free)
        });

        if !affinity {
            let index = self.free.pop_front()?;
            // Phase 8 still reuses whatever prefix the slot happens to hold;
            // only the *choice* of slot is FIFO. Reporting the LCP here keeps
            // the disabled path identical to Phase 8 instead of silently
            // turning every request into a cold prefill.
            let shared = self.slots.get(index).map_or(0, |entry| {
                longest_common_prefix(&entry.cached_prefix, tokens)
            });
            return Some((index, shared));
        }

        let mut best: Option<(usize, usize)> = None;
        for (position, index) in self.free.iter().copied().enumerate() {
            let Some(entry) = self.slots.get(index) else {
                continue;
            };
            let shared = longest_common_prefix(&entry.cached_prefix, tokens);
            if best.map_or(true, |(_, best_shared)| shared > best_shared) {
                best = Some((position, shared));
            }
        }

        let (position, shared) = best?;
        let index = self.free.remove(position)?;
        Some((index, shared))
    }
}

/// Locks the registry, recovering from a poisoned mutex.
///
/// The registry only contains vectors and channels; no user code runs under
/// the lock, so a panic while it is held would be a bug in the pool itself.
/// Recovering keeps telemetry and recovery paths working instead of turning
/// that bug into a second panic on every request. No `unwrap` or `expect` is
/// involved.
fn lock_registry(registry: &Mutex<SlotRegistry>) -> MutexGuard<'_, SlotRegistry> {
    match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A slot that was checked out of the pool while it serves a request.
///
/// Dropping the guard releases the slot: it is pushed back onto the free queue
/// and the busy counter is decremented. The guard lives on the slot thread
/// while a generation runs, so every exit path (success, abort, engine error,
/// panic) goes through this `Drop`.
#[derive(Debug)]
pub struct SlotGuard {
    /// Stable index of the slot.
    index: usize,
    /// The command channel of the slot, cloned so the request path can send to
    /// it before the guard is moved into the command.
    commands: Sender<SlotCommand>,
    /// The registry the slot is returned to.
    registry: Arc<Mutex<SlotRegistry>>,
    /// The number of busy slots, shared with the telemetry of the pool.
    active: Arc<AtomicUsize>,
    /// `true` once the slot was released or retired.
    released: bool,
}

impl SlotGuard {
    /// The index of the slot that is held, if it was not released yet.
    ///
    /// The index is stable for the life of the pool and only exists for logs
    /// and tests.
    pub fn index(&self) -> Option<usize> {
        if self.released {
            None
        } else {
            Some(self.index)
        }
    }

    /// The command channel of the slot.
    ///
    /// Only called before the guard is moved into a [`SlotCommand`].
    fn commands(&self) -> &Sender<SlotCommand> {
        &self.commands
    }

    /// Releases the slot without returning it to the free queue.
    ///
    /// Used when the context of the slot could not be proven clean (the panic
    /// cleanup itself failed). Such a context is never reused: the slot is
    /// retired instead of being recycled, the busy counter is decremented, the
    /// live capacity drops, and the pool keeps serving with the remaining
    /// slots. The retirement is observable as
    /// `synapse_slots_retired_total`.
    pub fn retire(mut self) {
        if self.released {
            return;
        }
        self.released = true;

        let retired = {
            let mut registry = lock_registry(&self.registry);
            let mut retired = false;

            if let Some(entry) = registry.slots.get_mut(self.index) {
                if entry.status != SlotStatus::Retired {
                    entry.status = SlotStatus::Retired;
                    entry.cached_prefix.clear();
                    registry.free.retain(|queued| *queued != self.index);
                    registry.live = registry.live.saturating_sub(1);
                    retired = true;
                }
            }

            retired
        };

        if retired {
            metrics::counter!("synapse_slots_retired_total").increment(1);
            let live = lock_registry(&self.registry).live;
            // `synapse_capacity_total` stays at the number the pool was
            // created with; the live gauge is what the node actually serves.
            metrics::gauge!("synapse_capacity_live").set(live as f64);
            error!(
                slot = self.index,
                live_capacity = live,
                "an inference slot was retired because its context could not be cleaned"
            );
        }

        let busy = self.active.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
        metrics::gauge!("synapse_active_generations").set(busy as f64);
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.released = true;

        // The slot is returned before the busy counter is decremented: once
        // `active()` reaches zero, the slot is guaranteed to be in the free
        // queue and a caller that observed an idle pool is never rejected
        // because of a slot that is still on its way back.
        {
            let mut registry = lock_registry(&self.registry);
            if let Some(entry) = registry.slots.get_mut(self.index) {
                if entry.status == SlotStatus::Busy {
                    entry.status = SlotStatus::Free;
                    registry.free.push_back(self.index);
                }
            }
        }

        let busy = self.active.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
        metrics::gauge!("synapse_active_generations").set(busy as f64);
    }
}

/// The error returned by [`InferencePool::try_acquire`] when every live slot is
/// busy.
///
/// This is a normal condition, not a fault: the caller is supposed to report
/// it to the client instead of waiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtCapacity {
    /// The number of live (not retired) slots the pool has.
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

/// A command accepted by a slot thread.
#[derive(Debug)]
enum SlotCommand {
    /// Run one generation and stream its events.
    Generate {
        /// The tokenized prompt. Tokenization happens on the request path so
        /// that slot selection can score the prompt against every free slot.
        tokens: Vec<i32>,
        /// Number of leading prompt tokens that are already resident in the
        /// KV cache of this slot. Zero means a full prefill.
        prefill_from: usize,
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
    /// The inspectable state of every slot.
    registry: Arc<Mutex<SlotRegistry>>,
    /// The number of slots that are currently serving a request.
    active: Arc<AtomicUsize>,
    /// The shared model. It is only used to tokenize prompts on the request
    /// path; the generation itself runs on the slot threads.
    model: Arc<LlamaModel>,
    /// Whether free slots are scored by their cached prefix.
    enable_affinity: bool,
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
    ///
    /// When `enable_prefix_caching` is `false`, every generation starts from
    /// an empty KV cache: the pool behaves exactly like it did before the
    /// prefix cache existed. When `enable_affinity_scheduling` is `false`,
    /// slot selection is the Phase 8 FIFO order and no prompt is scored.
    pub fn load(
        path: &str,
        max_capacity: u32,
        context_length: u32,
        enable_prefix_caching: bool,
        enable_affinity_scheduling: bool,
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

        // Affinity is meaningless without a cached prefix. Disabling it in the
        // pool as well keeps the disabled-cache path bit-for-bit the Phase 7
        // path and avoids counting matches that can never help.
        let enable_affinity = enable_affinity_scheduling && enable_prefix_caching;

        info!(
            slots = capacity,
            n_ctx = context_length,
            kv_cache_mib_per_slot = kv_mib_per_slot,
            kv_cache_mib_total = kv_mib_total,
            prefix_caching = enable_prefix_caching,
            affinity_scheduling = enable_affinity,
            "starting the inference pool"
        );

        let (ready_sender, ready_receiver) = flume::unbounded::<Result<(), SynapseError>>();

        // The busy counter is shared between the guards (which are owned by
        // the slot threads) and the telemetry of the pool.
        let active = Arc::new(AtomicUsize::new(0));

        let params = context_params(context_length, capacity);
        let mut slot_commands = Vec::with_capacity(capacity);

        // Every slot is registered before its thread is spawned. The pool is
        // not handed to a caller until all threads report ready, so a request
        // can never observe a slot whose context is still being created.
        let registry = Arc::new(Mutex::new(SlotRegistry {
            slots: Vec::with_capacity(capacity),
            free: (0..capacity).collect(),
            live: capacity,
        }));

        let mut handles = Vec::with_capacity(capacity);

        for index in 0..capacity {
            let (command_sender, command_receiver) = flume::unbounded::<SlotCommand>();
            slot_commands.push(command_sender.clone());

            {
                let mut registry = lock_registry(&registry);
                registry.slots.push(SlotEntry {
                    commands: command_sender,
                    status: SlotStatus::Free,
                    cached_prefix: Vec::new(),
                });
            }

            let startup = SlotStartup {
                index,
                model: Arc::clone(&model),
                backend,
                params: params.clone(),
                commands: command_receiver,
                ready: ready_sender.clone(),
                enable_prefix_caching,
                registry: Arc::clone(&registry),
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
        metrics::gauge!("synapse_capacity_live").set(capacity as f64);

        info!(
            slots = capacity,
            prefix_caching = enable_prefix_caching,
            affinity_scheduling = enable_affinity,
            "inference pool is ready"
        );

        Ok(Self {
            registry,
            active,
            model,
            enable_affinity,
            slot_commands,
            handles,
        })
    }

    /// The number of live slots in the pool.
    ///
    /// Retired slots are not included: this is the capacity the node can still
    /// serve, and the number [`TokenStreamProvider::max_capacity`] advertises
    /// to the mesh.
    pub fn capacity(&self) -> usize {
        lock_registry(&self.registry).live
    }

    /// The number of slots that are currently serving a request.
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Relaxed)
    }

    /// Tokenizes a prompt with the chat template of the loaded model.
    ///
    /// Exposed for diagnostics and tests. Tokenization only reads the
    /// vocabulary, so it is safe to call while slots are generating.
    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<i32>, SynapseError> {
        self.model.tokenize_prompt(prompt)
    }

    /// Takes a free slot without waiting.
    ///
    /// Returns [`AtCapacity`] when every live slot is busy. The request is
    /// never queued: the caller decides whether to reject it or to try another
    /// node. This variant does not score prefixes; it is used by tests and by
    /// callers that do not have tokenized prompt yet.
    pub fn try_acquire(&self) -> Result<SlotGuard, AtCapacity> {
        self.acquire(None).map(|(guard, _)| guard)
    }

    /// Takes the free slot whose cached prefix best matches `tokens`.
    ///
    /// Returns the guard and the number of tokens that can be reused from the
    /// slot's KV cache. The reusable prefix is capped at `tokens.len() - 1`:
    /// the last prompt token always has to be evaluated again because its
    /// logits are what the sampler continues from.
    fn acquire_affine(&self, tokens: &[i32]) -> Result<(SlotGuard, usize), AtCapacity> {
        self.acquire(Some(tokens))
    }

    /// Shared implementation of both acquisition paths.
    ///
    /// `None` means "any free slot" and disables scoring. `Some(tokens)` means
    /// "pick the free slot with the longest matching cached prefix", unless
    /// affinity scheduling is disabled, in which case it is the Phase 8 FIFO.
    fn acquire(&self, tokens: Option<&[i32]>) -> Result<(SlotGuard, usize), AtCapacity> {
        let empty: [i32; 0] = [];
        let prompt = match tokens {
            Some(tokens) => tokens,
            None => &empty,
        };
        let affinity = self.enable_affinity && tokens.is_some();

        // The registry lock covers selection and the status change only. It is
        // released before the command is sent and never held across a
        // generation.
        let mut registry = lock_registry(&self.registry);
        let (index, shared) = match registry.select(prompt, affinity) {
            Some(selection) => selection,
            None => {
                return Err(AtCapacity {
                    capacity: registry.live,
                });
            }
        };

        let reusable = if tokens.is_some() {
            // `tokens` can never be empty here: tokenization rejects an empty
            // prompt, and the slice is only passed for a real request.
            shared.min(prompt.len().saturating_sub(1))
        } else {
            0
        };

        let commands = match registry.slots.get_mut(index) {
            Some(entry) if entry.status == SlotStatus::Free => {
                entry.status = SlotStatus::Busy;
                entry.commands.clone()
            }
            // Defensive: `select` only returns free slots, so this can only
            // happen if the registry was corrupted. Reject instead of handing
            // out a slot that may be busy.
            _ => {
                return Err(AtCapacity {
                    capacity: registry.live,
                });
            }
        };

        // The busy counter is incremented under the registry lock, before the
        // lock is dropped, so `active() == 0` implies that no slot is busy.
        let busy = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        drop(registry);

        metrics::gauge!("synapse_active_generations").set(busy as f64);

        if affinity && shared > 0 {
            metrics::counter!("synapse_slot_affinity_matches_total").increment(1);
        }

        Ok((
            SlotGuard {
                index,
                commands,
                registry: Arc::clone(&self.registry),
                active: Arc::clone(&self.active),
                released: false,
            },
            reusable,
        ))
    }

    /// Starts a generation on a free slot.
    ///
    /// The prompt is tokenized before a slot is chosen, so the affinity score
    /// compares the *incoming* prompt with every slot's cached prefix. The slot
    /// guard travels with the command: the slot thread releases the slot when
    /// the generation ends, whatever the reason.
    fn start_generation(
        &self,
        prompt: String,
        max_tokens: usize,
    ) -> Result<Receiver<TokenEvent>, SynapseError> {
        let tokens = self.model.tokenize_prompt(&prompt)?;

        let (guard, prefill_from) = self
            .acquire_affine(&tokens)
            .map_err(|err| SynapseError::Engine(err.to_string()))?;

        let command_sender = guard.commands().clone();
        let (token_sender, token_receiver) = flume::unbounded::<TokenEvent>();

        let command = SlotCommand::Generate {
            tokens,
            prefill_from,
            max_tokens,
            token_sender,
            guard,
        };

        if command_sender.send(command).is_err() {
            // Dropping the failed command drops the guard, which returns the
            // slot to the free queue.
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

    /// The live capacity of the pool, which is what telemetry advertises.
    ///
    /// A retired slot is subtracted here. The router must never send work to a
    /// node that advertises capacity it no longer has.
    fn max_capacity(&self) -> u32 {
        u32::try_from(self.capacity()).unwrap_or(u32::MAX)
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
    /// Where the thread reports the outcome of its context creation.
    ready: Sender<Result<(), SynapseError>>,
    /// Whether a slot may reuse the prompt prefix that is in its KV cache.
    enable_prefix_caching: bool,
    /// The registry the slot updates with its new cached prefix.
    registry: Arc<Mutex<SlotRegistry>>,
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
            ready,
            enable_prefix_caching,
            registry,
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

        if ready.send(Ok(())).is_err() {
            // The pool is gone; the registry is dropped with it.
            return;
        }

        info!(
            slot = index,
            n_ctx = context.n_ctx(),
            "inference slot is ready"
        );

        while let Ok(command) = commands.recv() {
            match command {
                SlotCommand::Generate {
                    tokens,
                    prefill_from,
                    max_tokens,
                    token_sender,
                    guard,
                } => {
                    // A panic anywhere in the generation path is caught here.
                    // The cleanup below runs before the guard can return the
                    // slot, so a panicked context is never handed to the next
                    // request without a full clear.
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        run_generation(
                            &mut context,
                            &tokens,
                            prefill_from,
                            max_tokens,
                            token_sender,
                            SlotCache {
                                enabled: enable_prefix_caching,
                                slot: index,
                                registry: &registry,
                            },
                        )
                    }));

                    let clean = match outcome {
                        Ok(Ok(())) => true,
                        Ok(Err(err)) => {
                            error!(slot = index, "generation failed: {err}");
                            clear_after_failure(&mut context, &registry, index)
                        }
                        Err(_) => {
                            error!(
                                slot = index,
                                "the generation panicked; clearing the context before reuse"
                            );
                            clear_after_failure(&mut context, &registry, index)
                        }
                    };

                    if clean {
                        // The test hook retires a healthy slot so the capacity
                        // accounting can be tested. It is never set by a node.
                        if FORCE_RETIRE_AFTER_GENERATION.swap(false, Ordering::SeqCst) {
                            warn!(
                                slot = index,
                                "the test hook is retiring a healthy inference slot"
                            );
                            guard.retire();
                        } else {
                            // Releases the slot and decrements the busy counter.
                            drop(guard);
                        }
                    } else {
                        // The context could not be cleared: never reuse it.
                        guard.retire();
                    }
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

/// Length of the longest common prefix of two token sequences.
///
/// The prefix cache compares token IDs, not bytes: two prompts that render the
/// same text tokenize to the same IDs, and only a shared *token* prefix can be
/// reused from the KV cache. The function is pure and model independent so the
/// cache decision can be unit tested without loading a model.
pub fn longest_common_prefix(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// The cache bookkeeping a slot thread needs while it serves one generation.
struct SlotCache<'a> {
    /// Whether a slot may reuse the prefix in its KV cache.
    enabled: bool,
    /// Stable index of the slot in the registry.
    slot: usize,
    /// The registry that stores the cached prefix.
    registry: &'a Mutex<SlotRegistry>,
}

/// Serves one generation on a slot.
///
/// `prefill_from` was computed by the request path from the registry's cached
/// prefix. The slot verifies it against its actual KV cache with
/// [`LlamaContext::retain_prefix`]; any discrepancy downgrades to a full
/// prefill. On success the prompt tokens are stored back into the registry
/// while the slot is still busy, so the next selection sees them only after
/// the guard releases the slot.
///
/// Returns an error when the request could not be served. The caller treats
/// every error (and every panic) as a reason to clear the whole context, so
/// this function does not have to restore a consistent cache state itself.
fn run_generation(
    context: &mut LlamaContext<'_>,
    tokens: &[i32],
    prefill_from: usize,
    max_tokens: usize,
    token_sender: Sender<TokenEvent>,
    cache: SlotCache<'_>,
) -> Result<(), SynapseError> {
    let mut prefill_from = prefill_from;

    if cache.enabled {
        metrics::counter!("synapse_prefix_cache_requests_total").increment(1);

        if prefill_from == 0 {
            // Nothing to reuse (or the slot is cold): start from a clean
            // cache instead of trusting whatever state it has.
            context.clear_cache();
        } else {
            match context.retain_prefix(prefill_from) {
                Ok(()) => {}
                Err(err) => {
                    // The registry and the KV cache disagree. Correctness wins
                    // over the hit: throw the cache away and prefill
                    // everything from zero.
                    warn!(
                        slot = cache.slot,
                        "the cached prefix could not be retained, prefilling from zero: {err}"
                    );
                    context.clear_cache();
                    prefill_from = 0;
                }
            }
        }

        if prefill_from > 0 {
            metrics::counter!("synapse_prefix_cache_hits_total").increment(1);
            metrics::counter!("synapse_prefill_tokens_saved_total").increment(prefill_from as u64);
            info!(
                slot = cache.slot,
                prompt_tokens = tokens.len(),
                reused_tokens = prefill_from,
                prefill_tokens = tokens.len().saturating_sub(prefill_from),
                "prefix cache hit"
            );
        }
    } else {
        // The disabled path is the Phase 7 path: every generation starts from
        // an empty KV cache.
        context.clear_cache();
    }

    context.prefill_and_generate(tokens, prefill_from, max_tokens, token_sender)?;

    // Test hook. It fires after the completion tokens were decoded and before
    // the slot is trimmed, which is exactly the state a panic in the middle of
    // a generation would leave behind.
    if FORCE_PANIC_AFTER_GENERATION.swap(false, Ordering::SeqCst) {
        panic!("the test hook forced a panic after the generation");
    }

    if cache.enabled {
        // Drop the completion tokens: the prompt is what the next request can
        // reuse. The trim is verified by `retain_prefix`, and a failure is
        // returned to the caller, which clears the context completely.
        context.retain_prefix(tokens.len())?;

        let mut registry = lock_registry(cache.registry);
        match registry.slots.get_mut(cache.slot) {
            Some(entry) if entry.status == SlotStatus::Busy => {
                entry.cached_prefix = tokens.to_vec();
            }
            Some(_) => {
                // A retired slot must not advertise a prefix. This cannot
                // happen while the guard is held, but treating it as a miss is
                // safer than caching into a slot that is out of service.
                warn!(
                    slot = cache.slot,
                    "the slot was not busy after its generation; not caching"
                );
            }
            None => {
                return Err(SynapseError::Engine(format!(
                    "inference slot {} disappeared from the registry",
                    cache.slot
                )));
            }
        }
    } else {
        context.clear_cache();
        clear_cached_prefix(cache.registry, cache.slot);
    }

    Ok(())
}

/// Clears the cached prefix of a slot in the registry.
///
/// The lock is held only for the duration of the removal.
fn clear_cached_prefix(registry: &Mutex<SlotRegistry>, slot: usize) {
    let mut registry = lock_registry(registry);
    if let Some(entry) = registry.slots.get_mut(slot) {
        entry.cached_prefix.clear();
    }
}

/// Clears the context after a failed or panicked generation.
///
/// The registry's prefix is forgotten first: even if the KV clear panics, no
/// future selection can believe that the slot holds a valid prefix. Returns
/// `true` when the context is proven clean. When even the clear panics the
/// context cannot be trusted any more, so the caller retires the slot instead
/// of returning it to the free queue.
fn clear_after_failure(
    context: &mut LlamaContext<'_>,
    registry: &Mutex<SlotRegistry>,
    slot: usize,
) -> bool {
    clear_cached_prefix(registry, slot);

    match catch_unwind(AssertUnwindSafe(|| context.clear_cache())) {
        Ok(()) => true,
        Err(_) => {
            error!(
                slot,
                "the context could not be cleared after a failure; retiring the slot"
            );
            false
        }
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

    /// A registry with `capacity` free slots and no threads behind them.
    fn test_registry(
        capacity: usize,
    ) -> (
        Arc<Mutex<SlotRegistry>>,
        Arc<AtomicUsize>,
        Vec<Receiver<SlotCommand>>,
    ) {
        let mut slots = Vec::with_capacity(capacity);
        let mut receivers = Vec::with_capacity(capacity);

        for _ in 0..capacity {
            let (sender, receiver) = flume::unbounded::<SlotCommand>();
            slots.push(SlotEntry {
                commands: sender,
                status: SlotStatus::Free,
                cached_prefix: Vec::new(),
            });
            receivers.push(receiver);
        }

        let registry = Arc::new(Mutex::new(SlotRegistry {
            slots,
            free: (0..capacity).collect(),
            live: capacity,
        }));

        (registry, Arc::new(AtomicUsize::new(0)), receivers)
    }

    /// Simulates an acquisition: marks a slot busy, removes it from the queue
    /// and returns the guard a real acquisition would return.
    fn let_a_slot_be_busy(
        registry: &Arc<Mutex<SlotRegistry>>,
        active: &Arc<AtomicUsize>,
        index: usize,
    ) -> SlotGuard {
        let commands = {
            let mut reg = lock_registry(registry);
            reg.free.retain(|queued| *queued != index);
            match reg.slots.get_mut(index) {
                Some(entry) => {
                    entry.status = SlotStatus::Busy;
                    entry.commands.clone()
                }
                None => panic!("the test slot {index} does not exist"),
            }
        };

        active.fetch_add(1, Ordering::SeqCst);

        SlotGuard {
            index,
            commands,
            registry: Arc::clone(registry),
            active: Arc::clone(active),
            released: false,
        }
    }

    #[test]
    fn dropping_a_guard_returns_the_slot() {
        let (registry, active, _receivers) = test_registry(1);
        let guard = let_a_slot_be_busy(&registry, &active, 0);

        assert_eq!(guard.index(), Some(0));
        assert_eq!(active.load(Ordering::SeqCst), 1);
        assert!(lock_registry(&registry).free.is_empty());

        drop(guard);

        assert_eq!(active.load(Ordering::SeqCst), 0);
        let registry = lock_registry(&registry);
        assert_eq!(registry.free.iter().copied().collect::<Vec<_>>(), [0]);
        assert_eq!(
            registry.slots.first().map(|entry| entry.status),
            Some(SlotStatus::Free)
        );
    }

    #[test]
    fn retiring_a_guard_releases_it_exactly_once() {
        let (registry, active, _receivers) = test_registry(2);
        let guard = let_a_slot_be_busy(&registry, &active, 0);

        guard.retire();

        // `retire` consumes the guard, so its `Drop` also runs. It must not
        // decrement the busy counter a second time or return the slot.
        assert_eq!(active.load(Ordering::SeqCst), 0);
        let registry = lock_registry(&registry);
        assert_eq!(registry.live, 1);
        assert_eq!(registry.free.iter().copied().collect::<Vec<_>>(), [1]);
        assert_eq!(
            registry.slots.first().map(|entry| entry.status),
            Some(SlotStatus::Retired)
        );
    }

    #[test]
    fn a_retired_slot_is_never_selected() {
        let (registry, active, _receivers) = test_registry(2);
        let guard = let_a_slot_be_busy(&registry, &active, 0);
        guard.retire();

        let mut registry = lock_registry(&registry);
        let selected = registry.select(&[1, 2, 3], true);

        assert_eq!(selected, Some((1, 0)));
        assert_eq!(
            registry.free.iter().copied().collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn affinity_selects_the_longest_cached_prefix() {
        let (registry, _active, _receivers) = test_registry(2);
        {
            let mut registry = lock_registry(&registry);
            if let Some(entry) = registry.slots.get_mut(0) {
                entry.cached_prefix = vec![1, 2, 9];
            }
            if let Some(entry) = registry.slots.get_mut(1) {
                entry.cached_prefix = vec![1, 2, 3, 4];
            }

            let selected = registry.select(&[1, 2, 3, 5], true);
            assert_eq!(selected, Some((1, 3)));
        }
    }

    #[test]
    fn disabled_affinity_keeps_the_fifo_order() {
        let (registry, _active, _receivers) = test_registry(2);
        {
            let mut registry = lock_registry(&registry);
            if let Some(entry) = registry.slots.get_mut(0) {
                entry.cached_prefix = vec![1, 2, 9];
            }
            if let Some(entry) = registry.slots.get_mut(1) {
                entry.cached_prefix = vec![1, 2, 3, 4];
            }

            // Slot 1 is by far the better cache match, but the disabled path
            // must behave exactly like the Phase 8 free channel: FIFO. It must
            // still report slot 0's real shared prefix (2), so the generation
            // reuses the cache of the slot it actually took.
            let selected = registry.select(&[1, 2, 3], false);
            assert_eq!(selected, Some((0, 2)));
        }
    }

    #[test]
    fn affinity_ties_keep_the_fifo_order() {
        let (registry, _active, _receivers) = test_registry(2);
        {
            let mut registry = lock_registry(&registry);
            for slot in 0..2 {
                if let Some(entry) = registry.slots.get_mut(slot) {
                    entry.cached_prefix = vec![1, 2];
                }
            }

            assert_eq!(registry.select(&[1, 2, 3], true), Some((0, 2)));
        }
    }

    #[test]
    fn the_longest_common_prefix_is_token_wise() {
        assert_eq!(longest_common_prefix(&[], &[]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[]), 0);
        assert_eq!(longest_common_prefix(&[], &[1, 2, 3]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2]), 2);
        assert_eq!(longest_common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(longest_common_prefix(&[9], &[1]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
    }
}
