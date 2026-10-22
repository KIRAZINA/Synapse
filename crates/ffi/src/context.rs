//! Streaming inference context for `llama.cpp`.
//!
//! ## Findings: the streaming API of `llama-cpp-2`
//!
//! `llama-cpp-2` (0.1.160) does **not** expose a callback based generation
//! API. There is no `generate_with_callback`, no `sample_loop` and no way to
//! register a per-token callback with the C API. The only callback the crate
//! offers is `LlamaModelParams::with_progress_callback`, which fires while
//! the model is *loaded* and cannot be used for token generation.
//!
//! What the crate does expose are the building blocks that are needed to
//! drive the sampler by hand:
//!
//! * `LlamaContext::decode` evaluates a batch of tokens,
//! * `LlamaSampler::chain_simple`, `LlamaSampler::sample` and
//!   `LlamaSampler::accept` sample the next token,
//! * `LlamaVocab::is_eog` and `LlamaVocab::token_to_piece` detokenise it.
//!
//! The streaming implementation below therefore drives the sampler in an
//! explicit loop and forwards every sampled token to the caller through a
//! channel. A C trampoline (and the `unsafe` code that comes with it) is not
//! necessary, and the workspace contains no `unsafe` block at all any more.

use std::time::Instant;

use flume::Sender;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext as RawLlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use synapse_core::{StopReason, SynapseError, TokenEvent};
use tracing::info;

use crate::model::LlamaModel;

/// Default seed used by the sampler, kept fixed so that runs are reproducible.
const DEFAULT_SEED: u32 = 1234;

/// Keeps the active-generation gauge balanced, even when generation fails or
/// the client disconnects half way through.
struct ActiveGeneration;

impl ActiveGeneration {
    fn start() -> Self {
        metrics::gauge!("synapse_active_generations").increment(1.0);
        Self
    }
}

impl Drop for ActiveGeneration {
    fn drop(&mut self) {
        metrics::gauge!("synapse_active_generations").decrement(1.0);
    }
}

/// An inference context that borrows the model it was created from.
///
/// # Ownership
///
/// `llama.cpp` requires a context to be destroyed before the model it was
/// created from. Rust cannot express that relation for a struct that owns
/// both, so the context *borrows* the model instead: the borrow checker
/// guarantees that the model outlives the context, and no lifetime transmute
/// (and therefore no `unsafe` code) is needed.
pub struct LlamaContext<'model> {
    /// The native context.
    inner: RawLlamaContext<'model>,
    /// The model the context belongs to, used for the vocabulary.
    model: &'model LlamaModel,
    /// Size of the context window in tokens.
    n_ctx: u32,
    /// Seed of the sampler.
    seed: u32,
}

impl<'model> LlamaContext<'model> {
    /// Creates a context for `model`.
    pub fn new(
        model: &'model LlamaModel,
        backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<Self, SynapseError> {
        let started = Instant::now();
        let n_ctx = params.n_ctx().map_or(0, |size| size.get());

        let inner = model.raw().new_context(backend, params).map_err(|err| {
            SynapseError::Engine(format!("failed to create the inference context: {err}"))
        })?;

        let elapsed = started.elapsed();
        info!(
            n_ctx,
            init_time_ms = elapsed.as_millis(),
            "inference context initialized"
        );

        Ok(Self {
            inner,
            model,
            n_ctx,
            seed: DEFAULT_SEED,
        })
    }

    /// Size of the context window in tokens.
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Overrides the sampler seed.
    pub fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
    }

    /// Generates up to `max_tokens` tokens for `prompt` and streams them.
    ///
    /// Every sampled token is sent to `token_sender` as a
    /// [`TokenEvent::Token`] before the next one is sampled, so the caller
    /// sees the text as it is produced. The generation stops when
    ///
    /// * the end-of-sequence token is sampled ([`StopReason::Eos`]),
    /// * `max_tokens` tokens have been produced ([`StopReason::MaxTokens`]),
    /// * or the receiver of `token_sender` is dropped, in which case the
    ///   generation is aborted immediately.
    ///
    /// The terminal [`TokenEvent::Done`] event is only sent when the
    /// generation ran to completion; a failure is reported as
    /// [`TokenEvent::Error`].
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        token_sender: Sender<TokenEvent>,
    ) -> Result<(), SynapseError> {
        if max_tokens == 0 {
            return Err(self.fail(
                &token_sender,
                "max_tokens must be greater than zero".to_owned(),
            ));
        }

        let formatted_prompt = self.model.format_chat_prompt(prompt);
        let tokens = self
            .model
            .vocab()
            .tokenize(formatted_prompt.as_bytes(), true, true);

        if tokens.is_empty() {
            return Err(self.fail(
                &token_sender,
                "the prompt did not produce a single token".to_owned(),
            ));
        }

        let n_ctx = self.n_ctx as usize;
        let max_prompt_tokens = n_ctx.saturating_sub(max_tokens);
        if tokens.len() > max_prompt_tokens {
            return Err(self.fail(
                &token_sender,
                format!(
                    "the prompt needs {} tokens but only {max_prompt_tokens} of the {n_ctx} token context window are available",
                    tokens.len()
                ),
            ));
        }

        // From here on the model is actually working, so the generation counts
        // as active. The guard decrements the gauge on every exit path.
        let _active_generation = ActiveGeneration::start();

        // The batch holds the whole prompt; the generated tokens are decoded
        // one at a time afterwards.
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_index = (tokens.len() - 1) as i32;
        for (index, token) in (0_i32..).zip(tokens.iter().copied()) {
            batch
                .add(token, index, &[0], index == last_index)
                .map_err(|err| {
                    self.fail(
                        &token_sender,
                        format!("failed to add the prompt token {index} to the batch: {err}"),
                    )
                })?;
        }

        self.inner.decode(&mut batch).map_err(|err| {
            self.fail(&token_sender, format!("failed to decode the prompt: {err}"))
        })?;

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(self.seed), LlamaSampler::greedy()]);

        let started = Instant::now();
        let mut n_decoded = 0usize;
        let mut position = batch.n_tokens();
        let mut time_to_first_token_ms: Option<u64> = None;
        let mut stop_reason = StopReason::MaxTokens;

        while n_decoded < max_tokens {
            // The client is gone: stop as fast as possible instead of
            // generating tokens nobody is waiting for.
            if token_sender.is_disconnected() {
                info!(
                    tokens = n_decoded,
                    "client disconnected, aborting generation"
                );
                return Ok(());
            }

            let token = sampler.sample(&self.inner, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.vocab().is_eog(token) {
                stop_reason = StopReason::Eos;
                break;
            }

            let piece = self.model.vocab().token_to_piece(token, true, None);
            let text = String::from_utf8_lossy(&piece).into_owned();

            if token_sender
                .send(TokenEvent::Token {
                    text,
                    token_id: token.0,
                })
                .is_err()
            {
                info!(
                    tokens = n_decoded,
                    "client disconnected, aborting generation"
                );
                return Ok(());
            }

            metrics::counter!("synapse_tokens_generated_total").increment(1);

            if time_to_first_token_ms.is_none() {
                let ttft_ms = started.elapsed().as_millis() as u64;
                time_to_first_token_ms = Some(ttft_ms);
                metrics::histogram!("synapse_ttft_ms").record(ttft_ms as f64);
            }

            batch.clear();
            batch.add(token, position, &[0], true).map_err(|err| {
                self.fail(
                    &token_sender,
                    format!("failed to add the generated token to the batch: {err}"),
                )
            })?;
            position += 1;

            self.inner.decode(&mut batch).map_err(|err| {
                self.fail(
                    &token_sender,
                    format!("failed to decode the generated token: {err}"),
                )
            })?;

            n_decoded += 1;
        }

        let elapsed_ms = started.elapsed().as_millis() as u64;
        let tokens_per_second = if elapsed_ms > 0 {
            n_decoded as f64 / (elapsed_ms as f64 / 1000.0)
        } else {
            0.0
        };

        info!(
            tokens = n_decoded,
            elapsed_ms,
            time_to_first_token_ms = time_to_first_token_ms.unwrap_or(0),
            tokens_per_second,
            stop_reason = %stop_reason,
            "streaming generation finished"
        );

        // The receiver may have gone away in the meantime, in which case there
        // is nobody left to tell.
        let _ = token_sender.send(TokenEvent::Done {
            total_tokens: n_decoded,
            elapsed_ms,
            stop_reason,
        });

        Ok(())
    }

    /// Reports a failure to the caller and returns the matching error.
    fn fail(&self, token_sender: &Sender<TokenEvent>, message: String) -> SynapseError {
        let _ = token_sender.send(TokenEvent::Error {
            message: message.clone(),
        });
        SynapseError::Engine(message)
    }
}

impl std::fmt::Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("n_ctx", &self.n_ctx)
            .field("seed", &self.seed)
            .finish()
    }
}
