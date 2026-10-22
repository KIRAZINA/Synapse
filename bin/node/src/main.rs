//! Synapse node daemon entry point.

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use synapse_core::TokenEvent;
use synapse_engine::actor::InferenceActor;
use synapse_net::client::RemoteInferenceClient;
use synapse_net::discovery::{fetch_quic_info, parse_bootstrap};
use synapse_net::NetworkNode;
use tracing::{error, info};

/// Model used by the local inference demo.
const MODEL_PATH: &str = "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

/// Prompt sent to the model once it is loaded.
const DEMO_PROMPT: &str = "Hello, I am Synapse. Tell me a short joke about distributed systems.";

/// Number of tokens generated for the demo prompt.
const DEMO_MAX_TOKENS: usize = 50;

/// The port the Prometheus exporter listens on.
const DEFAULT_METRICS_PORT: u16 = 9090;

/// Command line arguments of the node.
#[derive(Debug, Parser)]
#[command(name = "synapse-node", version, about = "Synapse node daemon")]
struct Cli {
    /// Multiaddr of a bootstrap node that is dialed on start up.
    #[arg(long, value_name = "MULTIADDR")]
    bootstrap: Option<String>,

    /// Multiaddr of a remote node that a prompt is forwarded to.
    #[arg(long, value_name = "MULTIADDR")]
    target: Option<String>,

    /// Prompt that is forwarded to the remote node.
    #[arg(long, value_name = "PROMPT")]
    prompt: Option<String>,

    /// Number of tokens that are requested from the remote node.
    #[arg(long, default_value_t = 50, value_name = "COUNT")]
    max_tokens: u32,

    /// Port of the Prometheus exporter.
    #[arg(long, default_value_t = DEFAULT_METRICS_PORT, value_name = "PORT")]
    metrics_port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(true).init();

    let cli = Cli::parse();

    // The Prometheus exporter is isolated: a busy port or a broken exporter
    // must never take the node down.
    start_metrics_exporter(cli.metrics_port);

    // The Ctrl+C handler is installed right away: Tokio only registers it
    // when the future is polled for the first time, and until then the
    // default handler of the console would kill the process.
    let (shutdown_sender, mut shutdown_receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = shutdown_sender.send(());
        }
    });

    let actor = Arc::new(InferenceActor::spawn());
    info!("Synapse Node started successfully");

    // The control plane (libp2p) and the data plane (QUIC) are started in the
    // background; the node keeps running until it is asked to stop.
    let bootstrap = match cli.bootstrap.as_deref() {
        Some(address) => match parse_bootstrap(address) {
            Ok(address) => Some(address),
            Err(err) => {
                error!("{err}");
                None
            }
        },
        None => None,
    };

    let mut network = match NetworkNode::new(bootstrap, Arc::clone(&actor) as Arc<_>).await {
        Ok(network) => network,
        Err(err) => {
            error!("failed to start the network node: {err}");
            shutdown_actor(actor).await;
            return Ok(());
        }
    };

    let network_task = tokio::spawn(async move {
        if let Err(err) = network.run().await {
            error!("the network task stopped: {err}");
        }
    });

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
                let _ = (&mut shutdown_receiver).await;
                info!("shutdown signal received, stopping inference actor");

                // Dropping the receiver tells the actor that nobody is
                // listening any more, so an in-flight generation is aborted.
                consumer.abort();
                let _ = consumer.await;
            }
            Err(err) => error!("failed to start the generation: {err}"),
        }
    }

    // Forward a prompt to a remote node when one was requested.
    if let (Some(target), Some(prompt)) = (cli.target.as_deref(), cli.prompt.as_deref()) {
        match forward_prompt(target, prompt, cli.max_tokens, shutdown_receiver).await {
            Ok(()) => info!("the remote generation finished"),
            Err(err) => error!("the remote generation failed: {err}"),
        }
    }

    // Stopping the network task closes the libp2p swarm and the QUIC
    // endpoint.
    network_task.abort();
    let _ = network_task.await;

    shutdown_actor(actor).await;

    info!("Synapse Node stopped gracefully");
    Ok(())
}

/// Starts the Prometheus exporter.
///
/// The exporter runs in the background and serves the metrics on
/// `http://<host>:<port>/metrics`. A failure is logged, the node keeps
/// running without the exporter.
fn start_metrics_exporter(port: u16) {
    let address = SocketAddr::from(([0, 0, 0, 0], port));

    // A true Prometheus histogram needs bucket bounds. Without them the
    // exporter renders histograms as summaries.
    const TTFT_BUCKETS_MS: [f64; 13] = [
        1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0,
    ];

    let builder = PrometheusBuilder::new().with_http_listener(address);
    let builder = match builder.set_buckets_for_metric(
        Matcher::Full("synapse_ttft_ms".to_owned()),
        &TTFT_BUCKETS_MS,
    ) {
        Ok(builder) => builder,
        Err(err) => {
            error!("the prometheus exporter could not be configured: {err}");
            return;
        }
    };

    match builder.install() {
        Ok(()) => info!(%address, "the prometheus exporter is serving /metrics"),
        Err(err) => error!("the prometheus exporter could not be started: {err}"),
    }
}

/// Forwards a prompt to a remote node and prints the answer.
async fn forward_prompt(
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
        let event = tokio::select! {
            event = events.recv_async() => event,
            _ = &mut shutdown => {
                abort.abort();
                info!("interrupted, the remote generation is aborted");
                return Ok(());
            }
        };

        match event {
            Ok(TokenEvent::Token { text, .. }) => {
                if first_token_at.is_none() {
                    first_token_at = Some(Instant::now());
                }
                print!("{text}");
                let _ = stdout.flush();
            }
            Ok(TokenEvent::Done {
                total_tokens,
                elapsed_ms,
                stop_reason,
            }) => {
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
}

/// Stops the inference actor.
async fn shutdown_actor(actor: Arc<InferenceActor>) {
    match Arc::try_unwrap(actor) {
        Ok(actor) => {
            if let Err(err) = actor.shutdown() {
                error!("failed to stop the inference actor: {err}");
            }
        }
        Err(_) => {
            error!("the inference actor is still in use, it is not stopped");
        }
    }
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
