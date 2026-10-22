//! Synapse node daemon entry point.

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use synapse_core::{Config, TokenEvent, TokenStreamProvider, DEFAULT_CONFIG_FILE};
use synapse_engine::InferencePool;
use synapse_net::client::RemoteInferenceClient;
use synapse_net::discovery::{fetch_quic_info, parse_bootstrap};
use synapse_net::transport::TransportSettings;
use synapse_net::{NetworkNode, NetworkSettings, PeerMultiaddr, ServedModels};
use tracing::{error, info, warn};

/// Prompt sent to the model once it is loaded.
const DEMO_PROMPT: &str = "Hello, I am Synapse. Tell me a short joke about distributed systems.";

/// Number of tokens generated for the demo prompt.
const DEMO_MAX_TOKENS: usize = 50;

/// Command line arguments of the node.
#[derive(Debug, Parser)]
#[command(name = "synapse-node", version, about = "Synapse node daemon")]
struct Cli {
    /// Path of the TOML configuration file.
    #[arg(long, default_value = DEFAULT_CONFIG_FILE, value_name = "PATH")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Command>,

    /// Multiaddr of an additional bootstrap node that is dialed on start up.
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
}

/// The subcommands of the node.
#[derive(Debug, Subcommand)]
enum Command {
    /// Write a commented default configuration file and exit.
    InitConfig {
        /// Destination of the template.
        #[arg(long, default_value = DEFAULT_CONFIG_FILE, value_name = "PATH")]
        out: PathBuf,
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
    let cli = Cli::parse();

    if let Some(Command::InitConfig { out }) = &cli.command {
        write_config_template(out)?;
        return Ok(());
    }

    let config = load_config(&cli.config)?;

    // The Prometheus exporter is isolated: a busy port or a broken exporter
    // must never take the node down.
    if config.metrics.enabled {
        start_metrics_exporter(config.metrics.port);
    } else {
        info!("the prometheus exporter is disabled in the configuration");
    }

    // The Ctrl+C handler is installed right away: Tokio only registers it
    // when the future is polled for the first time, and until then the
    // default handler of the console would kill the process.
    let (shutdown_sender, mut shutdown_receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = shutdown_sender.send(());
        }
    });

    // The model is loaded and every inference slot is created before the
    // network starts. A node that cannot serve its model must not join the
    // mesh at all: if it did, it would either advertise a model it does not
    // have or answer every request with an error. Creating the contexts up
    // front is also what makes `max_capacity` real: no request pays for a
    // context (or its KV cache) any more.
    let pool = Arc::new(load_pool(&config)?);

    info!(
        max_capacity = pool.capacity(),
        context_length = config.model.context_length,
        "inference pool is running"
    );

    // Only a model that the engine reported as loaded is advertised.
    let served_models = Arc::new(ServedModels::new());
    served_models.set_loaded(config.model.name.clone());

    // The control plane (libp2p) and the data plane (QUIC) are started in the
    // background; the node keeps running until it is asked to stop.
    let settings = network_settings(&cli, &config, Arc::clone(&served_models))?;

    let mut network = NetworkNode::new(settings, Arc::clone(&pool) as Arc<dyn TokenStreamProvider>)
        .await
        .map_err(|err| anyhow!("failed to start the network node: {err}"))?;

    let network_task = tokio::spawn(async move {
        if let Err(err) = network.run().await {
            error!("the network task stopped: {err}");
        }
    });

    info!(
        prompt = DEMO_PROMPT,
        max_tokens = DEMO_MAX_TOKENS,
        "prompt sent"
    );

    let consumer = match pool.generate_streaming(DEMO_PROMPT.to_owned(), DEMO_MAX_TOKENS) {
        Ok(receiver) => {
            let request_started = Instant::now();
            Some(tokio::spawn(stream_to_console(receiver, request_started)))
        }
        Err(err) => {
            error!("failed to start the generation: {err}");
            None
        }
    };

    // The node keeps running until it is asked to stop.
    let _ = (&mut shutdown_receiver).await;
    info!("shutdown signal received, stopping the inference pool");

    // Dropping the receiver tells the actor that nobody is listening any
    // more, so an in-flight generation is aborted.
    if let Some(consumer) = consumer {
        consumer.abort();
        let _ = consumer.await;
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

    shutdown_pool(pool);

    info!("Synapse Node stopped gracefully");
    Ok(())
}

/// Reads the configuration, or falls back to the defaults when the file is
/// absent.
///
/// An absent file is not an error, but it is worth a warning: a node that runs
/// with defaults should be a deliberate choice.
fn load_config(path: &Path) -> Result<Config> {
    if !path.exists() {
        warn!(
            path = %path.display(),
            "configuration file not found, using the defaults; run `synapse-node init-config` to write a commented template"
        );
        return Ok(Config::default());
    }

    Ok(Config::load(path)?)
}

/// Writes the commented configuration template.
fn write_config_template(path: &Path) -> Result<()> {
    if path.exists() {
        return Err(anyhow!(
            "'{}' already exists; pass --out <path> to write the template somewhere else",
            path.display()
        ));
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create the directory '{}'", parent.display())
            })?;
        }
    }

    std::fs::write(path, Config::default_template())
        .with_context(|| format!("failed to write the template to '{}'", path.display()))?;

    println!("wrote the configuration template to '{}'", path.display());
    Ok(())
}

/// Loads the configured model and creates one context per slot.
///
/// The work is blocking, and it happens before the network starts, so waiting
/// for it cannot stall a request.
fn load_pool(config: &Config) -> Result<InferencePool> {
    let model_path = PathBuf::from(&config.model.path);

    if !model_path.is_file() {
        return Err(anyhow!(
            "the model file '{}' was not found; run 'scripts/download_model.ps1' (or 'scripts/download_model.sh') to fetch it, or point model.path at a GGUF file",
            model_path.display()
        ));
    }

    let pool = InferencePool::load(
        &model_path.to_string_lossy(),
        config.node.max_capacity,
        config.model.context_length,
        config.cache.enable_prefix_caching,
    )
    .map_err(|err| anyhow!("failed to load the model '{}': {err}", model_path.display()))?;

    info!(
        model = %config.model.name,
        path = %model_path.display(),
        prefix_caching = config.cache.enable_prefix_caching,
        "model loaded successfully"
    );

    Ok(pool)
}

/// Translates the configuration into the settings of the networking layer.
fn network_settings(
    cli: &Cli,
    config: &Config,
    served_models: Arc<ServedModels>,
) -> Result<NetworkSettings> {
    let listen_addr = config
        .node
        .listen_addr
        .parse::<PeerMultiaddr>()
        .map_err(|err| {
            anyhow!(
                "invalid node.listen_addr '{}': {err}",
                config.node.listen_addr
            )
        })?;

    let mut bootstrap = Vec::with_capacity(config.bootstrap.peers.len() + 1);
    for peer in &config.bootstrap.peers {
        let address = parse_bootstrap(peer)
            .with_context(|| format!("invalid bootstrap.peers entry '{peer}'"))?;
        if !bootstrap.contains(&address) {
            bootstrap.push(address);
        }
    }

    if let Some(extra) = cli.bootstrap.as_deref() {
        let address = parse_bootstrap(extra)?;
        if !bootstrap.contains(&address) {
            bootstrap.push(address);
        }
    }

    let transport = TransportSettings {
        idle_timeout: Duration::from_millis(config.quic.idle_timeout_ms),
        keep_alive_interval: Duration::from_millis(config.quic.keep_alive_interval_ms),
    };

    Ok(NetworkSettings {
        listen_addr,
        bootstrap,
        transport,
        served_models,
    })
}

/// Starts the Prometheus exporter.
///
/// The exporter runs in the background and serves the metrics on
/// `http://<host>:<port>/metrics`. A failure is logged, the node keeps
/// running without the exporter.
fn start_metrics_exporter(port: u16) {
    let address = SocketAddr::from(([0, 0, 0, 0], port));

    // A true Prometheus histogram needs bucket bounds. Without them the
    // exporter renders histograms as summaries. TTFT and prefill latency use
    // the same scale.
    const LATENCY_BUCKETS_MS: [f64; 13] = [
        1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0,
    ];

    let builder = PrometheusBuilder::new().with_http_listener(address);
    let builder = match builder.set_buckets_for_metric(
        Matcher::Full("synapse_ttft_ms".to_owned()),
        &LATENCY_BUCKETS_MS,
    ) {
        Ok(builder) => builder,
        Err(err) => {
            error!("the prometheus exporter could not be configured: {err}");
            return;
        }
    };
    let builder = match builder.set_buckets_for_metric(
        Matcher::Full("synapse_prefill_ms".to_owned()),
        &LATENCY_BUCKETS_MS,
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

/// Stops the inference pool and waits for its slot threads.
fn shutdown_pool(pool: Arc<InferencePool>) {
    match Arc::try_unwrap(pool) {
        Ok(pool) => {
            if let Err(err) = pool.shutdown() {
                error!("failed to stop the inference pool: {err}");
            }
        }
        Err(_) => {
            error!("the inference pool is still in use, it is not stopped");
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
