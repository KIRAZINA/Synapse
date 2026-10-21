//! Synapse command line client entry point.

use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use synapse_core::TokenEvent;
use synapse_net::client::RemoteInferenceClient;
use synapse_net::discovery::{fetch_quic_info, parse_bootstrap};
use tracing::{error, info};

/// Command line arguments of the client.
#[derive(Debug, Parser)]
#[command(name = "synapse-cli", version, about = "Synapse command line client")]
struct Cli {
    /// Multiaddr of a bootstrap node that is dialed on start up.
    #[arg(long, value_name = "MULTIADDR")]
    bootstrap: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

/// The subcommands of the client.
#[derive(Debug, Subcommand)]
enum Command {
    /// Send a prompt to a Synapse node.
    Chat {
        /// Text of the prompt to send.
        #[arg(long)]
        prompt: String,

        /// Multiaddr of the node that runs the generation.
        #[arg(long, value_name = "MULTIADDR")]
        remote: Option<String>,

        /// Number of tokens that are requested.
        #[arg(long, default_value_t = 50, value_name = "COUNT")]
        max_tokens: u32,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(true).init();

    // The Ctrl+C handler is installed right away: Tokio only registers it
    // when the future is polled for the first time, and until then the
    // default handler of the console would kill the process.
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = shutdown_sender.send(());
        }
    });

    let cli = Cli::parse();

    match cli.command {
        Some(Command::Chat {
            prompt,
            remote,
            max_tokens,
        }) => match remote.as_deref() {
            Some(target) => {
                chat_remote(target, &prompt, max_tokens, shutdown_receiver).await?;
            }
            None => {
                println!("prompt: {prompt}");
            }
        },
        None => match cli.bootstrap.as_deref() {
            Some(address) => {
                println!("bootstrap: {address}");
            }
            None => {
                println!("{}", Cli::command().render_help());
            }
        },
    }

    Ok(())
}

/// Runs a generation on a remote node and prints the answer as it arrives.
async fn chat_remote(
    target: &str,
    prompt: &str,
    max_tokens: u32,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
    let target = parse_bootstrap(target)?;

    info!(%target, "requesting the quic endpoint of the remote node");
    let (addr, cert_der) = fetch_quic_info(&target).await?;

    let client = RemoteInferenceClient::new(addr, cert_der)?;
    let (events, abort) = client.generate(prompt.to_owned(), max_tokens).await?;

    let request_started = Instant::now();
    let mut stdout = std::io::stdout();
    let mut first_token_at: Option<Instant> = None;

    loop {
        tokio::select! {
            event = events.recv_async() => {
                match event {
                    Ok(TokenEvent::Token { text, .. }) => {
                        if first_token_at.is_none() {
                            first_token_at = Some(Instant::now());
                        }
                        print!("{text}");
                        let _ = stdout.flush();
                    }
                    Ok(TokenEvent::Done { total_tokens, elapsed_ms, stop_reason }) => {
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
                            "the remote generation is done"
                        );
                        return Ok(());
                    }
                    Ok(TokenEvent::Error { message }) => {
                        error!("the remote generation failed: {message}");
                        return Ok(());
                    }
                    Err(err) => {
                        error!("the remote generation stream ended: {err}");
                        return Ok(());
                    }
                }
            }
            _ = &mut shutdown => {
                // The abort resets the stream, so the remote node stops
                // generating right away.
                abort.abort();
                info!("interrupted, the remote generation is aborted");
                return Ok(());
            }
        }
    }
}
