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
- a short burst is absorbed by a bounded FIFO admission queue instead of being
  rejected immediately, while sustained overload is still rejected once that
  queue is full or its deadline passes.

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

With `cache.enable_affinity_scheduling = true` (the default), the pool
tokenizes the prompt before it takes a slot and picks the *free* slot whose
cached prefix has the longest common token prefix with the incoming prompt.
Affinity is opportunistic: a busy slot is never waited for. With the default
admission queue a request waits for the next free slot; with `[pool]
queue_capacity = 0` a full pool is rejected immediately. Affinity only chooses
which free slot a dequeued request receives, never the queue order. Set it to
`false` for the Phase 8 behaviour of taking the oldest free slot.

Affinity is greedy: it always takes the free slot with the longest match, even
when that match is weak and the slot currently holds a *hot* prefix that is
likely to be requested again. Evicting the hot prefix for a weak match can
cause cache shaking under heterogeneous load. This is the classic online
caching trade-off and accepted for now. If real traffic shows shaking, the next
refinement is to prefer an empty slot when the best match is below a threshold.

To keep a hit byte-identical to a full prefill, the engine disables
`llama.cpp`'s tiled K-quant matmul (`GGML_CPU_TILED_MM=0`). That kernel picks
its K-chunking from the number of tokens in the batch, which changes the last
bits of the KV values between a large prefill and the shorter suffix of a hit.
The reference kernel is batch-shape invariant; on the CPUs measured here it
costs nothing measurable for prefill, while the cache still makes a repeated
prompt roughly 30x cheaper.

This byte identity is a deliberate trade-off rather than a correctness
requirement. Production inference is correct with either kernel: the tiled
matmul only changes floating-point accumulation order. Byte identity is a very
useful *test* invariant, because it lets the cache tests compare a hit against a
full prefill exactly instead of within a tolerance, and the reference kernel is
currently free. The environment variable is written exactly once, through
`_putenv` on Windows and `std::env::set_var` elsewhere, while
`init_backend` holds its process-wide initialisation lock. That happens before
the backend exists and before any slot thread or inference context is spawned,
and `llama.cpp` reads the variable on its first matrix multiplication, so no
generation can race the write. If large models or large batches later show a
real gain from the tiled kernel, the honest escape hatch would be a
`[cache] deterministic = false` option; byte identity would then become a
test-only mode. It is intentionally not configurable today, so that the
byte-identical invariant stays unconditional.

Set `cache.enable_prefix_caching = false` to clear the KV cache after every
generation, which is the behaviour from before the cache existed. Affinity is
disabled together with the cache, because there is no prefix to match.

### Admission queue

When every slot is busy, a request enters a bounded FIFO queue instead of being
rejected immediately. The request path only enqueues the tokenized prompt and
returns a stream receiver; the actual waiting happens on a dedicated admission
thread. That thread releases the registry mutex while it waits on a condition
variable, and a slot release or a new enqueue wakes it without busy-waiting.
Dispatch order is strict arrival order. When a slot frees, the oldest request
is taken and affinity chooses *which* free slot it receives. A request whose
client disconnected while it was queued is dropped before it can take a slot.

`[pool] queue_capacity` is the number of waiting requests (the request already
running on a slot is not counted), and `queue_wait_ms` is the deadline measured
from enqueue. A request that finds a full queue is rejected immediately with
`queue full`; one whose deadline passes is rejected with a terminal
`queue timeout` error event. `queue_capacity = 0` disables the queue entirely
and reproduces the Phase 9 reject-on-full behaviour exactly. A node whose last
slot retired has live capacity zero, so it does not queue work it can never
serve.

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
enable_affinity_scheduling = true

[pool]
# Waiting requests when every slot is busy. 0 disables the queue.
queue_capacity = 8
# Deadline for a queued request, in milliseconds.
queue_wait_ms = 5000

[bootstrap]
peers = []
```

`cargo run -p synapse-node -- init-config` writes the full commented template.

## Metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `synapse_active_generations` | gauge | Slots that are currently serving |
| `synapse_capacity_total` | gauge | Slots the pool was created with |
| `synapse_capacity_live` | gauge | Live slots that are not retired (what telemetry advertises) |
| `synapse_slots_retired_total` | counter | Slots permanently taken out of service |
| `synapse_tokens_generated_total` | counter | Sampled tokens |
| `synapse_ttft_ms` | histogram | Time from the end of prefill to the first token |
| `synapse_prefill_ms` | histogram | Prompt prefill time, measured separately from decode |
| `synapse_prefix_cache_requests_total` | counter | Requests served with prefix caching enabled |
| `synapse_prefix_cache_hits_total` | counter | Requests that reused at least one prompt token |
| `synapse_prefill_tokens_saved_total` | counter | Prompt tokens skipped through the cache |
| `synapse_slot_affinity_matches_total` | counter | Selections of a free slot with a non-empty cached-prefix match |
| `synapse_queue_depth` | gauge | Requests currently waiting for a slot |
| `synapse_queue_wait_ms` | histogram | Time a queued request waited before admission or timeout |
| `synapse_queue_admitted_total` | counter | Queued requests that were assigned a slot |
| `synapse_queue_rejected_total` | counter | Queue rejections, labeled `reason="full"`, `"timeout"` or `"no_capacity"` |

Derive the hit rate from `synapse_prefix_cache_hits_total /
synapse_prefix_cache_requests_total`. The saved-token counter is the direct
measure of prefill work that was avoided; it says nothing about decode time,
which is not affected by the cache. A full hit saves `prompt_tokens - 1` tokens:
the last prompt token is always evaluated again because its logits are what
sampling continues from. A saving that is consistently lower than that is a
partial prefix match, not a missed or double-counted full match.

`synapse_capacity_total` is the configured capacity. `synapse_capacity_live`
drops when a slot is retired, and the same live number is what
`max_capacity` — and therefore `NodeTelemetry.max_capacity` — reports to the
router. A node never advertises capacity it has lost.

A node that retires its last slot therefore advertises `max_capacity = 0`. It
stays in the mesh as an observer (its telemetry is still gossiped, so operators
can see it), but the router explicitly excludes zero-capacity peers from both
best-peer selection and the advertised model set. Such a node cannot queue
work either: the pool rejects immediately when `live == 0` instead of waiting
for a slot that will never exist.

## Tests

```sh
cargo test --workspace
```

The model-backed pool tests skip themselves when the GGUF file is absent. They
cover real concurrency, capacity rejection, disconnects, reuse after a panic,
byte-identical output for cold, partial and full prefix hits, the live-capacity
drop when a slot is retired, and the A, A, B, A affinity comparison at
capacity 2. The queue tests cover burst absorption, queue-full and timeout
reasons, FIFO dispatch order, a client that disconnects while queued, the
`queue_capacity = 0` Phase 9 regression, and a 20-request stress burst. The
affinity comparison prints the hit count, saved tokens and affinity matches for
the enabled and disabled paths.
