//! Synapse command line client entry point.

use std::io::Write;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use synapse_core::TokenEvent;
use synapse_net::client::{AbortHandle, RemoteInferenceClient};
use synapse_net::discovery::{
    collect_telemetry, fetch_quic_info, parse_bootstrap, DiscoverySnapshot,
};
use tracing::{error, info};

/// How long the client listens to the gossiped telemetry.
const DISCOVERY_WINDOW: Duration = Duration::from_secs(3);

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

        /// Discover a node with the gossiped telemetry and send the prompt to
        /// the least loaded one.
        #[arg(long)]
        discover: bool,

        /// Only route to a node that serves this model.
        #[arg(long, value_name = "NAME")]
        model: Option<String>,

        /// Number of tokens that are requested.
        #[arg(long, default_value_t = 50, value_name = "COUNT")]
        max_tokens: u32,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt().with_target(true).init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            error!("{err}");
            ExitCode::FAILURE
        }
    }
}

#[tokio::main]
async fn run() -> Result<()> {
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
            discover,
            model,
            max_tokens,
        }) => {
            if discover {
                let bootstrap = cli.bootstrap.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("--discover needs a --bootstrap multiaddr to join the mesh")
                })?;
                chat_discover(
                    bootstrap,
                    model.as_deref(),
                    &prompt,
                    max_tokens,
                    shutdown_receiver,
                )
                .await?;
            } else {
                match remote.as_deref() {
                    Some(target) => {
                        chat_remote(target, &prompt, max_tokens, shutdown_receiver).await?;
                    }
                    None => {
                        println!("prompt: {prompt}");
                    }
                }
            }
        }
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

/// Finds the least loaded node of the mesh that can serve `model` and runs the
/// generation there.
async fn chat_discover(
    bootstrap: &str,
    model: Option<&str>,
    prompt: &str,
    max_tokens: u32,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
    let bootstrap = parse_bootstrap(bootstrap)?;

    info!(%bootstrap, "joining the telemetry mesh");
    let discovery = collect_telemetry(&bootstrap, DISCOVERY_WINDOW).await?;

    info!(
        peers = discovery.len(),
        models = ?discovery.models(),
        "the routing table holds the telemetry of the known peers"
    );

    let best = discovery
        .best_peer(model)
        .ok_or_else(|| no_peer_error(model, &discovery))?;

    info!(
        peer_id = %best.peer_id,
        active_requests = best.active_requests,
        max_capacity = best.max_capacity,
        supported_models = ?best.supported_models,
        "selected the least loaded peer"
    );

    let address = discovery
        .address_of(&best)
        .ok_or_else(|| anyhow::anyhow!("the selected peer did not publish a reachable address"))?;

    info!(%address, "requesting the quic endpoint of the selected peer");
    let (addr, cert_der, nonce, _expiry) = fetch_quic_info(&address).await?;

    let keypair = synapse_net::generate_keypair();
    let client = RemoteInferenceClient::new(addr, cert_der, keypair, nonce)?;
    let (events, abort) = client.generate(prompt.to_owned(), max_tokens).await?;

    print_events(events, abort, &mut shutdown).await
}

/// Explains why no peer could be selected.
///
/// A requested model that no peer serves is the most likely reason, so the
/// error names the models the mesh did advertise. That turns a dead end into
/// something the operator can act on.
fn no_peer_error(model: Option<&str>, discovery: &DiscoverySnapshot) -> anyhow::Error {
    let models = discovery.models();

    match model {
        Some(name) if !models.is_empty() && !models.iter().any(|known| known == name) => {
            anyhow::anyhow!(
                "no peer serves model '{name}'; discovered models: [{}]",
                models.join(", ")
            )
        }
        Some(name) if models.iter().any(|known| known == name) => {
            anyhow::anyhow!("every peer that serves model '{name}' is at capacity")
        }
        Some(name) => {
            anyhow::anyhow!("no peer serves model '{name}'; no peer advertised any model")
        }
        None => anyhow::anyhow!("no peer with free capacity was found in the routing table"),
    }
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
    let (addr, cert_der, nonce, _expiry) = fetch_quic_info(&target).await?;

    let keypair = synapse_net::generate_keypair();
    let client = RemoteInferenceClient::new(addr, cert_der, keypair, nonce)?;
    let (events, abort) = client.generate(prompt.to_owned(), max_tokens).await?;

    print_events(events, abort, &mut shutdown).await
}

/// Prints the events of a generation until it is done.
async fn print_events(
    events: flume::Receiver<TokenEvent>,
    abort: AbortHandle,
    shutdown: &mut tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
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
            _ = &mut *shutdown => {
                // The abort resets the stream, so the remote node stops
                // generating right away.
                abort.abort();
                info!("interrupted, the remote generation is aborted");
                return Ok(());
            }
        }
    }
}
