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
