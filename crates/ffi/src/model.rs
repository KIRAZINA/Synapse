//! Safe wrapper around a loaded `llama.cpp` model.

use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaChatMessage;
use llama_cpp_2::model::LlamaModel as RawLlamaModel;
use llama_cpp_2::vocab::LlamaVocab;
use synapse_core::SynapseError;
use tracing::info;

use crate::context::LlamaContext;

/// A GGUF model loaded into memory.
///
/// The wrapper owns the native `llama_model` handle and releases it when it is
/// dropped. No raw pointer is exposed to the caller.
pub struct LlamaModel {
    inner: RawLlamaModel,
    path: String,
    size_bytes: u64,
}

impl LlamaModel {
    /// Loads a GGUF model from disk.
    ///
    /// The model is mapped read-only when the platform supports it, so the
    /// file is not copied into the heap.
    pub fn load_from_file(backend: &LlamaBackend, path: &str) -> Result<Self, SynapseError> {
        let started = Instant::now();

        // Phase 2 is CPU only: no layer is offloaded to a GPU.
        let params = LlamaModelParams::default().with_n_gpu_layers(0);

        let inner = RawLlamaModel::load_from_file(backend, path, &params)
            .map_err(|err| SynapseError::Engine(format!("failed to load model '{path}': {err}")))?;

        let elapsed = started.elapsed();
        let size_bytes = inner.size();

        info!(
            path = %path,
            size_mib = size_bytes / (1024 * 1024),
            parameters = inner.n_params(),
            context_length = inner.n_ctx_train(),
            load_time_ms = elapsed.as_millis(),
            "model loaded"
        );

        Ok(Self {
            inner,
            path: path.to_owned(),
            size_bytes,
        })
    }

    /// Path the model was loaded from.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Size of the model file in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Number of parameters in the model.
    pub fn n_params(&self) -> u64 {
        self.inner.n_params()
    }

    /// Context length the model was trained with.
    pub fn n_ctx_train(&self) -> u32 {
        self.inner.n_ctx_train()
    }

    /// Vocabulary of the model, used for tokenisation and detokenisation.
    pub fn vocab(&self) -> LlamaVocab<'_> {
        self.inner.vocab()
    }

    /// Formats `prompt` with the chat template of the model and tokenizes it.
    ///
    /// Tokenisation only reads the vocabulary; it does not touch the KV cache
    /// of an inference context. That is what lets the pool tokenize a prompt on
    /// the request path, before it chooses a slot: the token IDs are the unit
    /// the cache-affinity decision compares.
    ///
    /// # Errors
    ///
    /// Returns an error when the formatted prompt produces no token at all,
    /// which no generation could serve.
    pub fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<i32>, SynapseError> {
        let formatted_prompt = self.format_chat_prompt(prompt);
        let tokens = self
            .vocab()
            .tokenize(formatted_prompt.as_bytes(), true, true);

        if tokens.is_empty() {
            return Err(SynapseError::Engine(
                "the prompt did not produce a single token".to_owned(),
            ));
        }

        Ok(tokens.into_iter().map(|token| token.0).collect())
    }

    /// Formats `prompt` as a single user message using the chat template that
    /// is embedded in the model.
    ///
    /// Falls back to the raw prompt when the model does not ship a template.
    pub fn format_chat_prompt(&self, prompt: &str) -> String {
        let template = match self.inner.chat_template(None) {
            Ok(template) => template,
            Err(err) => {
                tracing::debug!("model has no chat template ({err}); using the raw prompt");
                return prompt.to_owned();
            }
        };

        let message = match LlamaChatMessage::new("user".to_owned(), prompt.to_owned()) {
            Ok(message) => message,
            Err(err) => {
                tracing::debug!("failed to build the chat message ({err}); using the raw prompt");
                return prompt.to_owned();
            }
        };

        match self
            .inner
            .apply_chat_template(&template, std::slice::from_ref(&message), true)
        {
            Ok(formatted) => formatted,
            Err(err) => {
                tracing::debug!("failed to apply the chat template ({err}); using the raw prompt");
                prompt.to_owned()
            }
        }
    }

    /// Estimates the memory one context allocates for its KV cache, in bytes.
    ///
    /// `llama.cpp` keeps the key and value cache in `f16` by default, so every
    /// token costs two bytes per head dimension and layer for the keys and the
    /// same amount again for the values:
    ///
    /// ```text
    /// n_ctx * n_layer * n_head_kv * (n_embd / n_head) * 2 (K + V) * 2 bytes
    /// ```
    ///
    /// The estimate is deliberately simple: it is used for logging and for
    /// warning an operator who configures a large pool, not for reserving
    /// memory. `n_ctx` is the context window of the context that is about to
    /// be created.
    pub fn estimated_kv_cache_bytes(&self, n_ctx: u32) -> u64 {
        // A zero in the metadata would be a broken model; the guards only keep
        // the arithmetic well defined.
        let n_head = self.inner.n_head().max(1);
        let n_head_kv = self.inner.n_head_kv().max(1);
        let n_embd = u32::try_from(self.inner.n_embd()).unwrap_or(0);
        let head_dim = n_embd / n_head;

        u64::from(n_ctx)
            .saturating_mul(u64::from(self.inner.n_layer()))
            .saturating_mul(u64::from(n_head_kv))
            .saturating_mul(u64::from(head_dim))
            .saturating_mul(4)
    }

    /// Creates an inference context for this model.
    ///
    /// The context borrows the model, so the model has to outlive it; the
    /// borrow checker enforces that.
    pub fn new_context<'model>(
        &'model self,
        backend: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext<'model>, SynapseError> {
        LlamaContext::new(self, backend, params)
    }

    /// Borrows the underlying native model.
    ///
    /// Only used inside this crate; the native handle never leaves it.
    pub(crate) fn raw(&self) -> &RawLlamaModel {
        &self.inner
    }
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("path", &self.path)
            .field("size_bytes", &self.size_bytes)
            .finish()
    }
}
