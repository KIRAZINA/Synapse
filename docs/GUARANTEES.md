# Synapse — Guarantees & Limitations

Phase 13 validation artifact. Profiling was performed on
`Qwen2.5-7B-Instruct-Q4_K_M.gguf` (7.6B parameters, 4460.4 MiB model) on an
8-logical-core CPU, Windows 11, MinGW gcc 16.2.0, Rust 1.99.1 (stable toolchain).

## Profiling numbers (verbatim)

Source: `target/profile/profile-20261024-123330-numbers.txt` (full raw log:
`target/profile/profile-20261024-123330.log`).

### Memory per slot (n_ctx = 2048, 4096, 8192)

```
PROFILE scenario=memory model_mib=4460.4 parameters=7615616512 n_ctx=2048 slots=1 kv_est_mib_per_slot=112.0 kv_est_mib_total=112.0 rss_before_mib=5.4 rss_after_load_mib=6121.1 peak_rss_mib=6121.1 load_ms=6206
PROFILE scenario=memory_touch ttft_ms=3582.4 decode_tok_s=2.09 rss_mib=7323.3 peak_rss_mib=7323.4
PROFILE scenario=memory model_mib=4460.4 parameters=7615616512 n_ctx=4096 slots=1 kv_est_mib_per_slot=224.0 kv_est_mib_total=224.0 rss_before_mib=5.4 rss_after_load_mib=6233.1 peak_rss_mib=6233.1 load_ms=7204
PROFILE scenario=memory_touch ttft_ms=6177.5 decode_tok_s=0.65 rss_mib=1399.4 peak_rss_mib=7435.7
PROFILE scenario=memory model_mib=4460.4 parameters=7615616512 n_ctx=8192 slots=1 kv_est_mib_per_slot=448.0 kv_est_mib_total=448.0 rss_before_mib=5.3 rss_after_load_mib=6457.0 peak_rss_mib=6457.0 load_ms=7660
PROFILE scenario=memory_touch ttft_ms=10754.1 decode_tok_s=0.95 rss_mib=3400.7 peak_rss_mib=7659.6
```

KV-cache estimate matches reality: each doubling of `n_ctx` adds exactly the
estimated per-slot KV bytes to resident memory (2048 -> 4096: +112.0 MiB;
4096 -> 8192: +224.0 MiB), and `rss_after_load` rises by the same amount.

### Single request and concurrent load (n_ctx = 2048, 64 tokens)

```
PROFILE scenario=single_cold ttft_ms=3801.3 decode_ms=23865.0 tokens=45 decode_tok_s=1.89 e2e_tok_s=1.63
PROFILE scenario=single_cold ttft_ms=3102.3 decode_ms=31321.0 tokens=49 decode_tok_s=1.56 e2e_tok_s=1.42
PROFILE scenario=single_cold ttft_ms=2915.2 decode_ms=18782.0 tokens=45 decode_tok_s=2.40 e2e_tok_s=2.08
PROFILE scenario=single_warm ttft_ms=507.8 decode_ms=27469.0 tokens=55 decode_tok_s=2.00 e2e_tok_s=1.97
PROFILE scenario=single_warm ttft_ms=423.6 decode_ms=23276.0 tokens=55 decode_tok_s=2.36 e2e_tok_s=2.32
PROFILE scenario=single_summary cold_runs=3 median_cold_ttft_ms=3102.3 median_decode_tok_s=1.89 median_e2e_tok_s=1.63 warm_runs=2 median_warm_ttft_ms=507.8 peak_rss_mib=7323.5
PROFILE scenario=concurrent ttft_ms=7165.8 decode_ms=26789.0 tokens=45 decode_tok_s=1.68 e2e_tok_s=1.33
PROFILE scenario=concurrent ttft_ms=8373.2 decode_ms=27158.0 tokens=49 decode_tok_s=1.80 e2e_tok_s=1.38
PROFILE scenario=concurrent_summary slots=2 requests=2 wall_ms=35516.2 total_tokens=94 aggregate_tok_s=2.65 median_ttft_ms=8373.2 min_ttft_ms=7165.8 max_ttft_ms=8373.2 median_decode_tok_s=1.80 peak_rss_mib=7450.0
PROFILE scenario=concurrent ttft_ms=11048.0 decode_ms=44709.0 tokens=45 decode_tok_s=1.01 e2e_tok_s=0.81
PROFILE scenario=concurrent ttft_ms=13335.8 decode_ms=45562.0 tokens=49 decode_tok_s=1.08 e2e_tok_s=0.83
PROFILE scenario=concurrent ttft_ms=12688.1 decode_ms=43807.0 tokens=45 decode_tok_s=1.03 e2e_tok_s=0.80
PROFILE scenario=concurrent ttft_ms=14350.0 decode_ms=43347.0 tokens=45 decode_tok_s=1.04 e2e_tok_s=0.78
PROFILE scenario=concurrent_summary slots=4 requests=4 wall_ms=58881.1 total_tokens=184 aggregate_tok_s=3.12 median_ttft_ms=13335.8 min_ttft_ms=11048.0 max_ttft_ms=14350.0 median_decode_tok_s=1.04 peak_rss_mib=7678.4
```

Single request: median cold TTFT 3102.3 ms, median decode 1.89 tok/s (e2e 1.63 tok/s).
Concurrent: capacity 2 -> aggregate 2.65 tok/s; capacity 4 -> aggregate 3.12 tok/s.

### Prefix cache (317-token shared system prompt)

```
PROFILE scenario=prefix_cold ttft_ms=31562.1 decode_ms=2394.0 tokens=7 decode_tok_s=2.92 e2e_tok_s=0.21
PROFILE scenario=prefix_warm ttft_ms=1308.9 decode_ms=2879.0 tokens=7 decode_tok_s=2.43 e2e_tok_s=1.68
PROFILE scenario=prefix_warm ttft_ms=1146.6 decode_ms=2215.0 tokens=7 decode_tok_s=3.16 e2e_tok_s=2.09
PROFILE scenario=prefix_warm ttft_ms=1238.1 decode_ms=2613.0 tokens=7 decode_tok_s=2.68 e2e_tok_s=1.82
PROFILE scenario=prefix_summary prompt_tokens=317 cold_ttft_ms=31562.1 warm_median_ttft_ms=1238.1 warm_median_decode_tok_s=2.68 cold_prefill_ms=31532.6 warm_prefill_median_ms=1220.2 prefill_saved_pct=96.1 cache_requests=3 cache_hits=3 hit_rate=1.000 prefill_tokens_saved_per_request=310.0 saved_pct_of_prompt=97.8 peak_rss_mib=7403.3
```

Hit rate 1.000 across 3 requests; 310.0 of 317 prompt tokens (97.8%) skipped per
request; prefill time 31532.6 ms -> 1220.2 ms (96.1% reduction).

### Verification cost (K = 3, in-process fan-out, QUIC excluded)

```
PROFILE scenario=verification_setup model_mib=4460.4 parameters=7615616512 n_ctx=2048 slots=3 kv_est_mib_per_slot=112.0 kv_est_mib_total=336.0 threads_per_slot=2
PROFILE scenario=verification_policy redundancy=3 sample_every=0 participant_timeout_s=600 quorum=all_must_answer
PROFILE scenario=verification_baseline run=0 wall_ms=12706.4 cpu_ms=24421.9 tokens=32 cpu_ms_per_token=763.18
PROFILE scenario=verification_enabled run=0 wall_ms=26935.2 cpu_ms=140000.0 generated_tokens_upper=96 cpu_ms_per_token_upper=1458.33 requested=3 answered=3 accepted=true
PROFILE scenario=verification_baseline run=1 wall_ms=10280.9 cpu_ms=19640.6 tokens=32 cpu_ms_per_token=613.77
PROFILE scenario=verification_enabled run=1 wall_ms=21413.1 cpu_ms=122093.8 generated_tokens_upper=96 cpu_ms_per_token_upper=1271.81 requested=3 answered=3 accepted=true
PROFILE scenario=verification_baseline run=2 wall_ms=10394.9 cpu_ms=20078.1 tokens=32 cpu_ms_per_token=627.44
PROFILE scenario=verification_enabled run=2 wall_ms=21514.7 cpu_ms=122234.4 generated_tokens_upper=96 cpu_ms_per_token_upper=1273.27 requested=3 answered=3 accepted=true
PROFILE scenario=verification_summary runs=3 max_tokens=32 baseline_wall_ms=10394.9 baseline_cpu_ms=20078.1 baseline_cpu_ms_per_token=627.44 verify_wall_ms=21514.7 verify_cpu_ms=122234.4 verify_cpu_ms_per_token_upper=1273.27 wall_multiplier=2.07 cpu_multiplier=6.09 answered_per_run=3.0/3 accepted_runs=3/3 note=in-process_fanout_quic_excluded
```

Real cost multiplier: wall-clock 2.07x, CPU 6.09x (K = 3 participants on one host
with thread contention; on three separate hosts the wall multiplier is
`max(node latency) + RTT`).

## GUARANTEED

- Identity: QUIC data-plane authentication (peer identity proven).
- Authorization: allowlist access gate (default mode); open mode is dev-only.
- Abuse bounds: per-peer quotas — concurrency, rolling 60s token budget,
  `max_tokens` ceiling.
- Gross-misbehavior detection: redundant verification with strict majority
  (K of K must answer; no strict majority = flag, no reputation movement).
- Reject-not-accuse: a participant reporting incompatible determinism fields is
  excluded, not penalized; a request that cannot be pinned is refused.

## NOT GUARANTEED

- Bit-exact output correctness across nodes.
- Detection of a colluding majority (K-1 malicious nodes can outvote one honest
  node).
- Floating-point determinism across different hardware vector widths (AVX2 vs
  AVX-512), different `llama.cpp` builds, or OS scheduling differences.
- Honest-compute proofs: there is no zkML. Verification is a spot check, not a
  proof.

## Operational characteristics

- Verification is opt-in (`[verification].enabled = false` by default) and never
  forced on all traffic. Reached only via sampling (`sample_every = N`) or an
  explicit verified request.
- Verification is strict: all K participants must answer. A timeout is not an
  abstention; it fails quorum and flags the request with no reputation movement.
- A verified request costs K times the compute of a normal request (measured
  6.09x CPU on one host for K = 3).
- Residual FP risk (Phase 12): pinning removes predictable divergence sources
  (same model, quantization, greedy decoding, seed, thread count, backend) but
  not all of them. Verification catches gross misbehavior (garbage output, a
  cheaper model, truncated generation), not bit-exact computation. When the
  pinned configuration differs, the request is rejected rather than an honest
  node being accused.

## Test tiers (green on profiling commit)

Fast tier (`scripts/test.ps1`, no model): all crates pass, 0 failures.
Model tier (`--ignored`, TinyLlama/Qwen): `pool` 18 passed; `auth` 11 passed;
`quota` 8 passed; `phase12` 7 passed; `phase12_e2e` 12 passed. Verbatim
summary lines:

```
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;  (synapse-core, fast)
test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;  (pool, model)
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;  (auth)
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;  (quota)
test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;  (phase12)
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;  (phase12_e2e)
```
```

Wait the fast tier had many test result lines. Fine.

Now BETA_GATE.md.<|end|><|start|> fullPath">
C:\1001110001000111101(1)\Rust\Synapse\docs\BETA_GATE.md