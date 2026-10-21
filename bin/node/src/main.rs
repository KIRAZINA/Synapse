//! Synapse node daemon entry point.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use synapse_core::TokenEvent;
use synapse_engine::actor::InferenceActor;
use tracing::{error, info};

/// Model used by the local inference demo.
const MODEL_PATH: &str = "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

/// Prompt sent to the model once it is loaded.
const DEMO_PROMPT: &str = "Hello, I am Synapse. Tell me a short joke about distributed systems.";

/// Number of tokens generated for the demo prompt.
const DEMO_MAX_TOKENS: usize = 50;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(true).init();

    let actor = InferenceActor::spawn();
    info!("Synapse Node started successfully");

    let model_path = PathBuf::from(MODEL_PATH);
    if !model_path.exists() {
        error!(
            "model file '{}' was not found; run 'scripts/download_model.ps1' (or 'scripts/download_model.sh') to fetch it",
            model_path.display()
        );
    } else {
        match actor
            .load_model(model_path.to_string_lossy().into_owned())
            .await
        {
            Ok(()) => info!("model loaded successfully"),
            Err(err) => error!("failed to load the model: {err}"),
        }

        info!(
            prompt = DEMO_PROMPT,
            max_tokens = DEMO_MAX_TOKENS,
            "prompt sent"
        );

        match actor
            .generate_streaming(DEMO_PROMPT.to_owned(), DEMO_MAX_TOKENS)
            .await
        {
            Ok(receiver) => {
                let request_started = Instant::now();
                let consumer = tokio::spawn(stream_to_console(receiver, request_started));

                // The node keeps running until it is asked to stop.
                tokio::signal::ctrl_c().await?;
                info!("shutdown signal received, stopping inference actor");

                // Dropping the receiver tells the actor that nobody is
                // listening any more, so an in-flight generation is aborted.
                consumer.abort();
                let _ = consumer.await;
            }
            Err(err) => error!("failed to start the generation: {err}"),
        }
    }

    actor.shutdown()?;

    info!("Synapse Node stopped gracefully");
    Ok(())
}

/// Consumes the token stream and prints it as it arrives.
async fn stream_to_console(receiver: flume::Receiver<TokenEvent>, request_started: Instant) {
    let mut stdout = std::io::stdout();
    let mut first_token_at: Option<Instant> = None;

    while let Ok(event) = receiver.recv_async().await {
        match event {
            TokenEvent::Token { text, token_id } => {
                if first_token_at.is_none() {
                    first_token_at = Some(Instant::now());
                }
                print!("{text}");
                let _ = stdout.flush();
                tracing::trace!(token_id, "token streamed");
            }
            TokenEvent::Done {
                total_tokens,
                elapsed_ms,
                stop_reason,
            } => {
                println!();
                let time_to_first_token_ms = first_token_at
                    .map(|at| at.duration_since(request_started).as_millis())
                    .unwrap_or(0);
                let tokens_per_second = if elapsed_ms > 0 {
                    total_tokens as f64 / (elapsed_ms as f64 / 1000.0)
                } else {
                    0.0
                };
                info!(
                    total_tokens,
                    elapsed_ms,
                    time_to_first_token_ms,
                    tokens_per_second,
                    stop_reason = %stop_reason,
                    "generation done"
                );
            }
            TokenEvent::Error { message } => {
                error!("generation failed: {message}");
            }
        }
    }
}
