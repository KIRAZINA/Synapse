# Synapse

Synapse is a decentralized peer-to-peer LLM inference network written in Rust.
A node loads a `llama.cpp` model, advertises it on a libp2p mesh, serves
streaming generations over QUIC, and routes requests to a node that actually
has the requested model.

## Workspace

| Crate / binary | Responsibility |
| --- | --- |
| `crates/core` | Shared types, token events, errors, TOML configuration |
| `crates/ffi` | The only crate with `unsafe`: safe wrappers around `llama.cpp` |
| `crates/engine` | Bounded pool of inference slots, one context and thread per slot |
| `crates/net` | libp2p control plane, discovery, routing and the QUIC data plane |
| `bin/node` | The node daemon, configuration loading and the Prometheus exporter |
| `bin/cli` | The client (`chat`, `discover`, abort handling) |

## Build and run

A GGUF model is required. The default configuration expects
`models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf`:

```sh
cargo build --release
cargo run --release -p synapse-node -- init-config
cargo run --release -p synapse-node -- --config synapse.toml
```

In another terminal:

```sh
cargo run --release -p synapse-cli -- chat --discover \
  --model tinyllama-1.1b --prompt "Tell me a short joke." --max-tokens 50
```

The node serves Prometheus metrics on `http://localhost:9090/metrics` by
default.

## Performance expectations

**`node.max_capacity` is not a throughput multiplier.** It is the number of
generations a node can serve *at the same time*, and it is a real capability
because every slot owns its own inference context and KV cache. On a CPU, the
aggregate token rate barely moves when the capacity is raised.

Measured on an 8-logical-core CPU with TinyLlama-1.1B Q4_K_M:

| Mode | End-to-end | Per request | Per thread |
| --- | --- | --- | --- |
| Serial (1 slot, 8 threads) | 14.6 tok/s | 14.6 tok/s | 1.83 tok/s |
| Parallel (2 slots, 4 threads each) | 16.2 tok/s | 8.1 tok/s | 2.03 tok/s |

The aggregate rate improves by about 11%, the per-request rate drops by about
45%, and per-thread efficiency is roughly unchanged. This is the physics of
CPU-bound inference, not a scheduling defect: the bottleneck is the total
number of FLOPs and the memory bandwidth of the host, not the overhead of a
context.

What concurrency buys:

- several clients are served simultaneously instead of waiting for each other,
- a slow or disconnected client cannot block everybody else,
- the advertised `max_capacity` is honest: the contexts and their KV caches
  exist before the first request arrives,
- when every slot is busy, the node rejects the next request immediately
  instead of building a queue of work it cannot keep up with.

What concurrency does **not** buy is more tokens per second on the same CPU.
The real levers for throughput are batching (evaluating several sequences in
one `llama.cpp` batch) and GPU offload. Those are deliberately not part of this
phase.

### Prefix reuse

The other way to make serving cheaper is not to repeat work. Every slot keeps
the prompt tokens of its last request in the KV cache and reuses the longest
common prefix (LCP) of the next prompt:

- after a generation, the completion tokens are trimmed away and the prompt
  tokens stay resident,
- the next request compares its tokenized prompt with that prefix, keeps the
  shared part of the KV cache and prefills only the new suffix,
- a shared system prompt is therefore evaluated once per slot instead of once
  per request,
- `synapse_prefill_ms` drops for the repeated requests and
  `synapse_prefill_tokens_saved_total` counts the skipped prompt tokens.

A hit never changes the output: the reused tokens are exactly the ones the
model would otherwise evaluate again. The cache is per slot (there is no
cross-slot sharing, no radix tree, no eviction and no sticky routing), and any
error, failed trim or panic falls back to a full `clear_cache()` and a prefill
from zero.

To keep a hit byte-identical to a full prefill, the engine disables
`llama.cpp`'s tiled K-quant matmul (`GGML_CPU_TILED_MM=0`). That kernel picks
its K-chunking from the number of tokens in the batch, which changes the last
bits of the KV values between a large prefill and the shorter suffix of a hit.
The reference kernel is batch-shape invariant; on the CPUs measured here it
costs nothing measurable for prefill, while the cache still makes a repeated
prompt roughly 30x cheaper.

Set `cache.enable_prefix_caching = false` to clear the KV cache after every
generation, which is the behaviour from before the cache existed.

## Configuration

Every field is optional and unknown keys are rejected. A missing file falls
back to the defaults with a warning. The most important sections:

```toml
[node]
listen_addr = "/ip4/0.0.0.0/tcp/0"
# Concurrent generations. Each slot costs one context and one KV cache.
max_capacity = 1

[model]
path = "./models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
name = "tinyllama-1.1b"
context_length = 2048

[quic]
idle_timeout_ms = 10000
keep_alive_interval_ms = 5000

[metrics]
enabled = true
port = 9090

[cache]
enable_prefix_caching = true

[bootstrap]
peers = []
```

`cargo run -p synapse-node -- init-config` writes the full commented template.

## Metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `synapse_active_generations` | gauge | Slots that are currently serving |
| `synapse_capacity_total` | gauge | Slots the pool was created with |
| `synapse_tokens_generated_total` | counter | Sampled tokens |
| `synapse_ttft_ms` | histogram | Time from the end of prefill to the first token |
| `synapse_prefill_ms` | histogram | Prompt prefill time, measured separately from decode |
| `synapse_prefix_cache_requests_total` | counter | Requests served with prefix caching enabled |
| `synapse_prefix_cache_hits_total` | counter | Requests that reused at least one prompt token |
| `synapse_prefill_tokens_saved_total` | counter | Prompt tokens skipped through the cache |

Derive the hit rate from `synapse_prefix_cache_hits_total /
synapse_prefix_cache_requests_total`. The saved-token counter is the direct
measure of prefill work that was avoided; it says nothing about decode time,
which is not affected by the cache.

## Tests

```sh
cargo test --workspace
```

The model-backed pool tests skip themselves when the GGUF file is absent. They
cover real concurrency, capacity rejection, disconnects, reuse after a panic,
and byte-identical output for cold, partial and full prefix hits.
