//! The TOML configuration of a Synapse node.
//!
//! Every field has a default, so a configuration file is allowed to be
//! partial: a file that only sets `model.path` keeps the defaults for the
//! ports, the timeouts and the capacity. Unknown fields are rejected, because
//! a typo in a key would otherwise be ignored silently and the operator would
//! wonder why nothing changed.
//!
//! The configuration is data only: it is parsed here, and the crates that own
//! a subsystem (`synapse-net`, `synapse-engine`, the binaries) translate their
//! section into their own types. That keeps `synapse-core` free of the
//! networking and inference stacks.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The configuration file that is used when no other path is given.
pub const DEFAULT_CONFIG_FILE: &str = "synapse.toml";

/// A conservative KV-cache cost of one token, in bytes.
///
/// The estimate is only used to warn an operator whose configuration is
/// plausibly too large for the host. A 7B model with grouped-query attention
/// (32 layers, 8 KV heads of 128 dimensions, stored as `f16`) costs about
/// 128 KiB per token; smaller models cost less.
const CONSERVATIVE_KV_BYTES_PER_TOKEN: u64 = 128 * 1024;

/// The estimated KV-cache size above which the operator is warned.
const KV_CACHE_WARN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// The commented template that `synapse-node init-config` writes.
///
/// A test keeps the template in sync with [`Config::default`]: the template
/// has to parse to exactly the defaults it documents.
const DEFAULT_TEMPLATE: &str = r#"# Synapse node configuration.
#
# Every section and every field is optional: a missing value falls back to the
# value shown here, so a partial file is a valid file. Unknown keys are
# rejected, so a typo fails loudly instead of being ignored.
#
# Written by `synapse-node init-config`.

[node]
# The multiaddr the control plane (libp2p) listens on. Port 0 picks a free
# port; the clients learn the real address from the mesh.
listen_addr = "/ip4/0.0.0.0/tcp/0"

# The number of generations the node serves in parallel. Every slot owns its
# own inference context and KV cache and runs on its own thread, so raising
# this really enables concurrent serving. It is bounded by memory: each slot
# adds roughly one KV cache on top of the model. Requests that arrive while
# every slot is busy wait in the bounded queue configured in [pool].
max_capacity = 1

[model]
# The GGUF file that is loaded at start up. A node that cannot load its model
# exits instead of advertising a model it cannot serve.
path = "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"

# The name that is advertised to the mesh and that `synapse-cli --model`
# matches against.
name = "tinyllama-1.1b"

# The context window of the inference contexts, in tokens.
context_length = 2048

[quic]
# A peer that stops answering is considered dead after this many
# milliseconds. This is what bounds how long a killed client can keep a
# generation alive.
idle_timeout_ms = 10000

# How often a QUIC PING is sent on an idle connection. It has to stay below the
# idle timeout or it never runs; keep-alive also keeps a long prompt prefill
# from being mistaken for a dead peer.
keep_alive_interval_ms = 5000

[metrics]
# Serve Prometheus metrics on /metrics.
enabled = true

# The port of the metrics listener.
port = 9090

[cache]
# Reuse the KV cache of a slot for the prefix that a new prompt shares with the
# previous request of that slot. Prefill is the most expensive part of serving,
# so a shared system prompt is only evaluated once per slot instead of once per
# request. A hit never changes the output: the reused tokens are exactly the
# ones the model would evaluate again. Set this to false to clear the KV cache
# after every generation, which is the pre-caching behaviour.
enable_prefix_caching = true

# Prefer a free slot whose cached prompt prefix best matches the incoming
# prompt. The prompt is tokenized before a slot is chosen, and only the
# slot with the longest common prefix is taken. When every slot is busy the
# request waits in the [pool] queue; affinity only chooses which free slot a
# dequeued request receives, never the queue order. Disable it to get the
# Phase 8 behaviour of taking the oldest free slot.
enable_affinity_scheduling = true

[pool]
# Maximum number of requests that may wait for a free inference slot. Dispatch
# is strict FIFO. Set this to 0 to disable queueing: every request that finds
# all slots busy is rejected immediately, which is the Phase 9 behaviour.
queue_capacity = 8

# Maximum time, in milliseconds, that a queued request waits before it is
# rejected with a "queue timeout" error event.
queue_wait_ms = 5000

[security]
# "allowlist" (default) or "open" (dev only, logs a loud warning).
mode = "allowlist"
auth_timeout_ms = 5000
nonce_ttl_ms = 30000
allow_loopback = false

# The largest max_tokens a single request may ask for. This is a ceiling on
# the request, not a budget: the per-peer budget lives in [security.access].
max_tokens_per_request = 4096

[security.access]
trusted_peers = []
per_peer_concurrency = 1
per_peer_tokens_per_minute = 4096

[verification]
# Redundant verification: run the same request on several nodes and compare
# the outputs. This is OFF by default and it costs K times the compute, so it
# is meant for sampled or explicitly requested checks, never for all traffic.
#
# The comparison is only meaningful when every node runs the same model and
# the same quantization at temperature 0 (greedy decoding). A node that
# samples, or that serves a different quantization, produces different tokens
# for the same prompt and would be flagged as a liar by a naive comparison.
# Requests that cannot be pinned to a deterministic configuration are refused
# instead of being compared.
enabled = false

# How many nodes a verified request is fanned out to. 3 tolerates one liar
# (a strict majority still agrees); 2 only detects a disagreement without
# telling you which side is wrong.
redundancy = 3

# Verify one request in every N. 0 disables sampling entirely, which is what
# an operator who only wants on-demand verification should set.
sample_every = 0

# Reputation delta applied to every node that agreed with the majority.
reward = 1.0

# Reputation delta applied to a node that disagreed with the majority.
penalty = -5.0

# Reputation is bounded to this range so that a node cannot buy its way out
# of a bad history with a long run of good answers.
min_reputation = -50.0
max_reputation = 50.0

# A reputation entry that has not been updated for this long is pruned, so
# the ledger cannot grow without bound on a long-lived node.
entry_ttl_ms = 3600000

# The maximum number of reputation entries that are kept.
max_entries = 4096

[bootstrap]
# Multiaddrs of the peers that are dialed at start up.
peers = []
"#;

/// Errors produced while loading or validating a configuration.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error(
        "failed to read the configuration file '{path}': {source}; run `synapse-node init-config` to write a template"
    )]
    Read {
        /// The path that was read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The file is not valid TOML, or it does not match the schema.
    #[error("the configuration file '{path}' is invalid: {source}")]
    ParseFile {
        /// The path that was read.
        path: PathBuf,
        /// The underlying TOML error.
        ///
        /// Boxed because `toml::de::Error` is large and the happy path is the
        /// one that has to stay cheap.
        #[source]
        source: Box<toml::de::Error>,
    },

    /// The text is not valid TOML, or it does not match the schema.
    #[error("the configuration is invalid: {source}")]
    Parse {
        /// The underlying TOML error.
        #[source]
        source: Box<toml::de::Error>,
    },

    /// The configuration could not be serialized.
    #[error("failed to serialize the configuration: {source}")]
    Serialize {
        /// The underlying TOML error.
        #[source]
        source: Box<toml::ser::Error>,
    },

    /// A value is out of range or contradicts another value.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

/// The complete configuration of a node.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The control plane and the capacity of the node.
    pub node: NodeConfig,
    /// The model the node loads and advertises.
    pub model: ModelConfig,
    /// The timeouts of every QUIC connection.
    pub quic: QuicConfig,
    /// The Prometheus exporter.
    pub metrics: MetricsConfig,
    /// The prompt prefix cache of the inference slots.
    pub cache: CacheConfig,
    /// The admission queue of the inference slots.
    pub pool: PoolConfig,
    /// The authentication and authorization policy.
    pub security: SecurityConfig,
    /// The compute-integrity policy: redundant verification.
    pub verification: VerificationConfig,
    /// The peers that are dialed at start up.
    pub bootstrap: BootstrapConfig,
}

/// The authentication and authorization policy of the node.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecurityConfig {
    /// "open" (explicit dev only) or "allowlist".
    pub mode: String,
    /// Milliseconds to wait for an AUTH frame before closing.
    pub auth_timeout_ms: u64,
    /// Lifetime of an issued nonce, in milliseconds.
    pub nonce_ttl_ms: u64,
    /// Maximum max_tokens accepted per request.
    pub max_tokens_per_request: u64,
    /// Whether loopback peers are admitted without listing.
    pub allow_loopback: bool,
    /// Access control lists and quotas.
    pub access: SecurityAccessConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            mode: "allowlist".to_owned(),
            auth_timeout_ms: 5_000,
            nonce_ttl_ms: 30_000,
            max_tokens_per_request: 4096,
            allow_loopback: false,
            access: SecurityAccessConfig::default(),
        }
    }
}

/// Access control lists and per-peer quotas.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecurityAccessConfig {
    /// libp2p PeerId strings that are allowed to use this node.
    pub trusted_peers: Vec<String>,
    /// Maximum concurrent admitted requests per peer.
    pub per_peer_concurrency: usize,
    /// Rolling 60s budget of requested max_tokens per peer.
    pub per_peer_tokens_per_minute: u64,
}

impl Default for SecurityAccessConfig {
    fn default() -> Self {
        Self {
            trusted_peers: Vec::new(),
            per_peer_concurrency: 1,
            per_peer_tokens_per_minute: 4096,
        }
    }
}

impl Config {
    /// Reads, parses and validates a configuration file.
    ///
    /// A missing file is an error: the caller decides whether the defaults are
    /// acceptable, so that an operator who *did* write a path is never
    /// silently served a default configuration.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;

        let config: Self = toml::from_str(&text).map_err(|source| ConfigError::ParseFile {
            path: path.to_path_buf(),
            source: Box::new(source),
        })?;

        config.validate()?;
        Ok(config)
    }

    /// Parses a configuration from a TOML string.
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text).map_err(|source| ConfigError::Parse {
            source: Box::new(source),
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Serializes the configuration as TOML.
    ///
    /// This is the machine readable counterpart of [`Config::default_template`]:
    /// it writes the *current* values, without comments.
    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        toml::to_string_pretty(self).map_err(|source| ConfigError::Serialize {
            source: Box::new(source),
        })
    }

    /// The commented template of a default configuration.
    pub fn default_template() -> &'static str {
        DEFAULT_TEMPLATE
    }

    /// Checks the values that no default can guard against.
    ///
    /// Zero is never a sensible value for a capacity, a timeout or a length,
    /// and a keep-alive that is not shorter than the idle timeout never runs.
    /// Both cases are reported instead of being silently accepted.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.node.max_capacity == 0 {
            return Err(ConfigError::Invalid(
                "node.max_capacity has to be at least 1".to_owned(),
            ));
        }

        if self.node.listen_addr.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "node.listen_addr must not be empty".to_owned(),
            ));
        }

        if self.model.path.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "model.path must not be empty".to_owned(),
            ));
        }

        if self.model.name.trim().is_empty() {
            return Err(ConfigError::Invalid(
                "model.name must not be empty: it is the name the mesh routes on".to_owned(),
            ));
        }

        if self.model.context_length == 0 {
            return Err(ConfigError::Invalid(
                "model.context_length has to be at least 1".to_owned(),
            ));
        }

        if self.quic.idle_timeout_ms == 0 {
            return Err(ConfigError::Invalid(
                "quic.idle_timeout_ms has to be at least 1".to_owned(),
            ));
        }

        if self.quic.keep_alive_interval_ms == 0 {
            return Err(ConfigError::Invalid(
                "quic.keep_alive_interval_ms has to be at least 1".to_owned(),
            ));
        }

        if self.quic.keep_alive_interval_ms >= self.quic.idle_timeout_ms {
            return Err(ConfigError::Invalid(format!(
                "quic.keep_alive_interval_ms ({}) has to be lower than quic.idle_timeout_ms ({}), \
                 otherwise the keep-alive never runs before the connection times out",
                self.quic.keep_alive_interval_ms, self.quic.idle_timeout_ms
            )));
        }

        // Raising `max_capacity` used to be an advertisement only; it now
        // pre-creates one context (and therefore one KV cache) per slot. The
        // exact size depends on the model, so this is a deliberately
        // conservative warning based on the configured window; the engine logs
        // the precise estimate from the model metadata at startup.
        let token_slots =
            u64::from(self.node.max_capacity).saturating_mul(u64::from(self.model.context_length));
        let estimated_kv_bytes = token_slots.saturating_mul(CONSERVATIVE_KV_BYTES_PER_TOKEN);

        if estimated_kv_bytes > KV_CACHE_WARN_BYTES {
            tracing::warn!(
                max_capacity = self.node.max_capacity,
                context_length = self.model.context_length,
                estimated_kv_cache_mib = estimated_kv_bytes / (1024 * 1024),
                "node.max_capacity enables concurrent serving now, and every slot allocates its \
                 own KV cache; make sure the host has enough memory for the model plus this cache \
                 (the estimate is conservative)"
            );
        }

        Ok(())
    }
}

/// The compute-integrity policy of the node.
///
/// Authentication (Phase 11) proves *who* answered. It says nothing about
/// whether the answer is *correct*: an authenticated node can still return
/// garbage, or the output of a smaller model. Redundant verification closes
/// that gap by running the same request on several nodes and comparing the
/// outputs.
///
/// # Cost
///
/// A verified request costs `redundancy` times the compute of a normal one.
/// That is why verification is **off by default** and why the two ways to
/// turn it on are both bounded: sampling (`sample_every`) checks a fraction of
/// the traffic, and an explicit per-request flag checks a single request.
/// Verification is never forced on all traffic.
///
/// # The determinism constraint
///
/// Comparing outputs is only valid when every node runs the **same model at
/// the same quantization** and decodes **greedily (temperature 0)**. Any
/// sampling, any different quantization, any different context or prompt
/// template produces a different token sequence for the same prompt, and a
/// naive comparison would then punish an honest node. A request that cannot be
/// pinned to a deterministic configuration is therefore refused rather than
/// compared; see [`crate::types::Determinism`].
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VerificationConfig {
    /// Whether sampled verification runs at all.
    pub enabled: bool,
    /// How many nodes a verified request is fanned out to.
    pub redundancy: usize,
    /// Verify one request in every `N`. `0` disables sampling.
    pub sample_every: u64,
    /// Reputation delta for a node that agreed with the majority.
    pub reward: f64,
    /// Reputation delta for a node that disagreed with the majority.
    pub penalty: f64,
    /// Lower bound of a reputation score.
    pub min_reputation: f64,
    /// Upper bound of a reputation score.
    pub max_reputation: f64,
    /// How long an untouched reputation entry is kept, in milliseconds.
    pub entry_ttl_ms: u64,
    /// The maximum number of reputation entries that are kept.
    pub max_entries: usize,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            // Off by default: verification multiplies the cost of a request by
            // `redundancy`, so it is an explicit decision, never a surprise.
            enabled: false,
            redundancy: 3,
            sample_every: 0,
            reward: 1.0,
            penalty: -5.0,
            min_reputation: -50.0,
            max_reputation: 50.0,
            entry_ttl_ms: 3_600_000,
            max_entries: 4096,
        }
    }
}

impl VerificationConfig {
    /// Whether a request with this sequence number is sampled for verification.
    ///
    /// Sampling is deterministic in the request counter, so a run of requests
    /// is spread evenly instead of being verified in random clumps.
    pub fn should_sample(&self, request_seq: u64) -> bool {
        self.enabled && self.sample_every > 0 && request_seq % self.sample_every == 0
    }

    /// Checks the values that no default can guard against.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.redundancy < 2 {
            return Err(ConfigError::Invalid(
                "verification.redundancy has to be at least 2: a single output has nothing to be \
                 compared against"
                    .to_owned(),
            ));
        }

        if self.min_reputation >= self.max_reputation {
            return Err(ConfigError::Invalid(format!(
                "verification.min_reputation ({}) has to be lower than verification.max_reputation \
                 ({})",
                self.min_reputation, self.max_reputation
            )));
        }

        if self.max_entries == 0 {
            return Err(ConfigError::Invalid(
                "verification.max_entries has to be at least 1".to_owned(),
            ));
        }

        if self.entry_ttl_ms == 0 {
            return Err(ConfigError::Invalid(
                "verification.entry_ttl_ms has to be at least 1".to_owned(),
            ));
        }

        if !self.reward.is_finite() || !self.penalty.is_finite() {
            return Err(ConfigError::Invalid(
                "verification.reward and verification.penalty have to be finite".to_owned(),
            ));
        }

        Ok(())
    }
}

/// The control plane and the capacity of the node.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NodeConfig {
    /// The multiaddr the control plane listens on.
    pub listen_addr: String,
    /// The number of generations the node serves in parallel.
    ///
    /// Every slot owns one inference context and one KV cache, so this is a
    /// real promise, not an advertisement.
    pub max_capacity: u32,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            listen_addr: "/ip4/0.0.0.0/tcp/0".to_owned(),
            max_capacity: 1,
        }
    }
}

/// The model the node loads and advertises.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// The path of the GGUF file.
    pub path: String,
    /// The name that is advertised to the mesh.
    pub name: String,
    /// The context window of an inference context, in tokens.
    pub context_length: u32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf".to_owned(),
            name: "tinyllama-1.1b".to_owned(),
            context_length: 2048,
        }
    }
}

/// The timeouts of every QUIC connection.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct QuicConfig {
    /// How long a connection may stay idle before it is considered dead.
    pub idle_timeout_ms: u64,
    /// How often a QUIC `PING` is sent on an idle connection.
    pub keep_alive_interval_ms: u64,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            idle_timeout_ms: 10_000,
            keep_alive_interval_ms: 5_000,
        }
    }
}

/// The Prometheus exporter of a node.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetricsConfig {
    /// Whether the exporter is started at all.
    pub enabled: bool,
    /// The port the exporter listens on.
    pub port: u16,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 9090,
        }
    }
}

/// The prompt prefix cache of the inference slots.
///
/// The cache is per slot and holds at most one prefix: the prompt tokens of
/// the previous request that are still in the KV cache. It is not shared
/// between slots and it has no eviction policy; a miss simply prefills the
/// whole prompt.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    /// Whether a slot may reuse the prompt prefix that is in its KV cache.
    ///
    /// When this is `false`, every generation starts from an empty KV cache,
    /// exactly like before prefix caching existed.
    pub enable_prefix_caching: bool,
    /// Whether a free slot is chosen by its cached prompt prefix.
    ///
    /// The prompt is tokenized before the slot is selected and every free
    /// slot's cached prefix is scored by the length of the common prefix. The
    /// best free slot wins; when every slot is busy the request waits in the
    /// admission queue. When this is `false`, the oldest free slot is taken,
    /// which is the Phase 8 behaviour.
    pub enable_affinity_scheduling: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_prefix_caching: true,
            enable_affinity_scheduling: true,
        }
    }
}

/// The bounded admission queue of the inference slots.
///
/// A request that finds every slot busy waits here, in strict FIFO order, for
/// at most [`queue_wait_ms`](Self::queue_wait_ms). A capacity of zero disables
/// the queue and restores immediate reject-on-full.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoolConfig {
    /// Maximum number of requests that may wait for a slot.
    pub queue_capacity: usize,
    /// Maximum time a queued request waits before it is rejected, in ms.
    pub queue_wait_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 8,
            queue_wait_ms: 5_000,
        }
    }
}

/// The peers that are dialed at start up.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BootstrapConfig {
    /// The multiaddrs of the bootstrap peers.
    pub peers: Vec<String>,
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_complete() {
        let config = Config::default();

        assert_eq!(config.node.listen_addr, "/ip4/0.0.0.0/tcp/0");
        assert_eq!(config.node.max_capacity, 1);
        assert_eq!(
            config.model.path,
            "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
        );
        assert_eq!(config.model.name, "tinyllama-1.1b");
        assert_eq!(config.model.context_length, 2048);
        assert_eq!(config.quic.idle_timeout_ms, 10_000);
        assert_eq!(config.quic.keep_alive_interval_ms, 5_000);
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.port, 9090);
        assert!(config.cache.enable_prefix_caching);
        assert!(config.cache.enable_affinity_scheduling);
        assert_eq!(config.pool.queue_capacity, 8);
        assert_eq!(config.pool.queue_wait_ms, 5_000);
        assert!(config.bootstrap.peers.is_empty());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn prefix_caching_can_be_disabled() {
        let config = Config::from_toml_str("[cache]\nenable_prefix_caching = false\n")
            .unwrap_or_else(|err| panic!("the cache section does not parse: {err}"));

        assert!(!config.cache.enable_prefix_caching);
        assert_eq!(config.node, Config::default().node);
    }

    #[test]
    fn affinity_scheduling_can_be_disabled() {
        let config = Config::from_toml_str(
            "[cache]\nenable_prefix_caching = true\nenable_affinity_scheduling = false\n",
        )
        .unwrap_or_else(|err| panic!("the cache section does not parse: {err}"));

        assert!(config.cache.enable_prefix_caching);
        assert!(!config.cache.enable_affinity_scheduling);
        assert_eq!(config.node, Config::default().node);
    }

    #[test]
    fn the_queue_can_be_disabled() {
        let config = Config::from_toml_str("[pool]\nqueue_capacity = 0\nqueue_wait_ms = 250\n")
            .unwrap_or_else(|err| panic!("the pool section does not parse: {err}"));

        assert_eq!(config.pool.queue_capacity, 0);
        assert_eq!(config.pool.queue_wait_ms, 250);
        assert_eq!(config.node, Config::default().node);
    }

    #[test]
    fn the_template_parses_to_the_defaults() {
        let template = Config::from_toml_str(Config::default_template())
            .unwrap_or_else(|err| panic!("the template does not parse: {err}"));

        assert_eq!(template, Config::default());
    }

    #[test]
    fn a_partial_file_keeps_the_defaults() {
        let config = Config::from_toml_str(
            r#"
            [model]
            name = "llama-3-8b"

            [metrics]
            port = 9100
            "#,
        )
        .unwrap_or_else(|err| panic!("the partial configuration does not parse: {err}"));

        assert_eq!(config.model.name, "llama-3-8b");
        assert_eq!(config.model.path, Config::default().model.path);
        assert_eq!(config.metrics.port, 9100);
        assert!(config.metrics.enabled);
        assert_eq!(config.quic, Config::default().quic);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let result = Config::from_toml_str(
            r#"
            [quic]
            idle_timout_ms = 5000
            "#,
        );

        let message = match result {
            Ok(_) => panic!("a misspelled key was accepted"),
            Err(err) => err.to_string(),
        };

        assert!(
            message.contains("idle_timout_ms"),
            "the error does not name the offending key: {message}"
        );
    }

    #[test]
    fn invalid_values_are_rejected() {
        let no_capacity = Config::from_toml_str("[node]\nmax_capacity = 0\n");
        assert!(no_capacity.is_err());

        let no_timeout = Config::from_toml_str("[quic]\nidle_timeout_ms = 0\n");
        assert!(no_timeout.is_err());

        // A keep-alive that is not shorter than the idle timeout never runs.
        let slow_keep_alive = Config::from_toml_str(
            "[quic]\nidle_timeout_ms = 1000\nkeep_alive_interval_ms = 1000\n",
        );
        assert!(slow_keep_alive.is_err());

        let no_name = Config::from_toml_str("[model]\nname = \"\"\n");
        assert!(no_name.is_err());
    }

    #[test]
    fn a_large_capacity_is_a_warning_not_an_error() {
        // 64 slots of 8192 tokens are definitely more KV cache than a normal
        // host has memory for. That is worth a warning, but it is not the
        // configuration parser's job to decide how much memory the operator
        // has: the node starts and the engine reports the real estimate.
        let config =
            Config::from_toml_str("[node]\nmax_capacity = 64\n\n[model]\ncontext_length = 8192\n");

        let config = match config {
            Ok(config) => config,
            Err(err) => panic!("a large capacity was rejected: {err}"),
        };

        assert_eq!(config.node.max_capacity, 64);
        assert_eq!(config.model.context_length, 8192);
    }

    #[test]
    fn serializes_and_round_trips() {
        let config = Config::default();
        let text = match config.to_toml_string() {
            Ok(text) => text,
            Err(err) => panic!("serialization failed: {err}"),
        };

        let parsed = match Config::from_toml_str(&text) {
            Ok(parsed) => parsed,
            Err(err) => panic!("the serialized configuration does not parse: {err}"),
        };

        assert_eq!(parsed, config);
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let result = Config::load(Path::new("this/file/does/not/exist.toml"));

        let message = match result {
            Ok(_) => panic!("a missing configuration file was accepted"),
            Err(err) => err.to_string(),
        };

        assert!(
            message.contains("failed to read"),
            "the error is not actionable: {message}"
        );
    }
}
