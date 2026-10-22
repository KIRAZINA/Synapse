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
use llama_cpp_2::token::LlamaToken;
use synapse_core::{StopReason, SynapseError, TokenEvent};
use tracing::info;

use crate::model::LlamaModel;

/// Default seed used by the sampler, kept fixed so that runs are reproducible.
const DEFAULT_SEED: u32 = 1234;

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

    /// Empties the KV cache of the context.
    ///
    /// This is the `k = 0` case of [`LlamaContext::retain_prefix`]: no token of
    /// the previous request survives, so the next request starts from a blank
    /// slate and can never attend to tokens it did not send.
    pub fn clear_cache(&mut self) {
        self.inner.clear_kv_cache();
        tracing::debug!(n_ctx = self.n_ctx, "inference context cache cleared");
    }

    /// Formats `prompt` with the chat template of the model and tokenizes it.
    ///
    /// The returned token IDs are the unit the prefix cache works with: a slot
    /// remembers the prompt tokens that are resident in its KV cache, and the
    /// next request can reuse them when the two sequences share a prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when the formatted prompt produces no token at all,
    /// which no generation could serve.
    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<i32>, SynapseError> {
        let formatted_prompt = self.model.format_chat_prompt(prompt);
        let tokens = self
            .model
            .vocab()
            .tokenize(formatted_prompt.as_bytes(), true, true);

        if tokens.is_empty() {
            return Err(SynapseError::Engine(
                "the prompt did not produce a single token".to_owned(),
            ));
        }

        Ok(tokens.into_iter().map(|token| token.0).collect())
    }

    /// Keeps the KV cache entries of the first `k` prompt tokens and drops
    /// everything after them.
    ///
    /// The context then logically holds exactly `k` prefilled tokens. The next
    /// decode has to start at position `k`, which is the caller's job; this
    /// method only removes the entries.
    ///
    /// `k = 0` is [`LlamaContext::clear_cache`] and clears everything.
    ///
    /// # Implementation
    ///
    /// Positions `[k, end)` are removed with
    /// `LlamaContext::clear_kv_cache_seq(Some(0), Some(k), None)`, the wrapper
    /// of `llama_memory_seq_rm(seq 0, k, -1)` of `llama.cpp`. It is the
    /// supported way to discard a suffix of a sequence without rebuilding the
    /// prefix in front of it. The call reports `false` when only a partial
    /// removal was possible, and the position of the last surviving token is
    /// checked as well: a retained prefix of `k` tokens has to end at `k - 1`.
    /// Anything else is treated as an error, so the pool can fall back to a
    /// full clear instead of decoding on top of a KV cache it does not
    /// understand.
    pub fn retain_prefix(&mut self, k: usize) -> Result<(), SynapseError> {
        if k == 0 {
            self.clear_cache();
            return Ok(());
        }

        let boundary = u32::try_from(k).map_err(|_| {
            SynapseError::Engine(format!(
                "the cached prefix of {k} tokens does not fit into the KV cache"
            ))
        })?;

        let removed = self
            .inner
            .clear_kv_cache_seq(Some(0), Some(boundary), None)
            .map_err(|err| SynapseError::Engine(format!("failed to trim the KV cache: {err}")))?;

        if !removed {
            return Err(SynapseError::Engine(
                "llama.cpp could not remove the cached tokens after the prefix".to_owned(),
            ));
        }

        let last = self.inner.kv_cache_seq_pos_max(0);
        if usize::try_from(last).ok() != Some(k - 1) {
            return Err(SynapseError::Engine(format!(
                "the KV cache ends at position {last} after retaining {k} tokens"
            )));
        }

        tracing::debug!(cached_tokens = k, "KV cache trimmed to the cached prefix");
        Ok(())
    }

    /// Prefills `tokens[prefill_from..]` and then generates up to `max_tokens`
    /// tokens, streaming them.
    ///
    /// `tokens` is the complete, already tokenized prompt. The first
    /// `prefill_from` tokens are expected to be resident in the KV cache
    /// (see [`LlamaContext::retain_prefix`]); only the remaining prompt tokens
    /// are evaluated. `prefill_from` therefore has to be lower than
    /// `tokens.len()`: the logits of the last prompt token are produced by its
    /// own decode and are not part of the KV cache, so at least that token has
    /// to be evaluated again even on a full prefix hit.
    ///
    /// The prefill and the decode loop are separate and are accounted
    /// separately: the prefill time is recorded in the `synapse_prefill_ms`
    /// histogram and logged with the number of evaluated and reused tokens.
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
    pub fn prefill_and_generate(
        &mut self,
        tokens: &[i32],
        prefill_from: usize,
        max_tokens: usize,
        token_sender: Sender<TokenEvent>,
    ) -> Result<(), SynapseError> {
        if max_tokens == 0 {
            return Err(self.fail(
                &token_sender,
                "max_tokens must be greater than zero".to_owned(),
            ));
        }

        if tokens.is_empty() {
            return Err(self.fail(
                &token_sender,
                "the prompt did not produce a single token".to_owned(),
            ));
        }

        if prefill_from >= tokens.len() {
            return Err(self.fail(
                &token_sender,
                format!(
                    "a cached prefix of {prefill_from} tokens does not leave a prompt token to decode"
                ),
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

        // The batch holds only the tokens that are not in the KV cache yet;
        // the cached prefix keeps its original positions.
        let prefill_started = Instant::now();
        let prompt_tokens = tokens.len() - prefill_from;
        let last_index = tokens.len() - 1;

        // The last prompt token is deliberately *not* batched with the others:
        // it is the token the first sample is taken from, and a one-token
        // attention step follows a different code path in `llama.cpp` than a
        // large prefill batch. Evaluating it alone in both the cold and the
        // cached case makes the logits, and therefore the output, reproducible
        // across a cache hit.
        if prefill_from < last_index {
            let batched = last_index - prefill_from;
            let mut batch = LlamaBatch::new(batched, 1);

            for (offset, token) in tokens[prefill_from..last_index].iter().copied().enumerate() {
                let position = prefill_from + offset;
                let position_i32 = i32::try_from(position).map_err(|_| {
                    self.fail(
                        &token_sender,
                        format!("the prompt position {position} does not fit into llama.cpp"),
                    )
                })?;
                batch
                    .add(LlamaToken::new(token), position_i32, &[0], false)
                    .map_err(|err| {
                        self.fail(
                            &token_sender,
                            format!(
                                "failed to add the prompt token at position {position} to the batch: {err}"
                            ),
                        )
                    })?;
            }

            self.inner.decode(&mut batch).map_err(|err| {
                self.fail(&token_sender, format!("failed to decode the prompt: {err}"))
            })?;
        }

        let mut batch = LlamaBatch::new(1, 1);
        let last_position = i32::try_from(last_index).map_err(|_| {
            self.fail(
                &token_sender,
                format!("the prompt position {last_index} does not fit into llama.cpp"),
            )
        })?;
        batch
            .add(
                LlamaToken::new(tokens[last_index]),
                last_position,
                &[0],
                true,
            )
            .map_err(|err| {
                self.fail(
                    &token_sender,
                    format!("failed to add the last prompt token to the batch: {err}"),
                )
            })?;

        self.inner.decode(&mut batch).map_err(|err| {
            self.fail(&token_sender, format!("failed to decode the prompt: {err}"))
        })?;
        let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        metrics::histogram!("synapse_prefill_ms").record(prefill_ms);

        info!(
            prefill_tokens = prompt_tokens,
            cached_tokens = prefill_from,
            prefill_ms,
            "prompt prefill finished"
        );

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(self.seed), LlamaSampler::greedy()]);

        let started = Instant::now();
        let mut n_decoded = 0usize;
        let mut position = i32::try_from(tokens.len()).map_err(|_| {
            self.fail(
                &token_sender,
                format!(
                    "the prompt length {} does not fit into llama.cpp",
                    tokens.len()
                ),
            )
        })?;
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
            prefill_tokens = prompt_tokens,
            cached_tokens = prefill_from,
            prefill_ms,
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
