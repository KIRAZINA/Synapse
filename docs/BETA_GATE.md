# Synapse v1.0-beta gate checklist

Per-item status as of the Phase 13 validation commit.

| # | Item | Status |
|---|------|--------|
| 1 | Fast test tier green (`scripts/test.ps1`, no model) | met |
| 2 | Model test tier green (`pool` 18/18) | met |
| 3 | No `unwrap()`/`expect()` in production code (only in `#[cfg(test)]`) | met |
| 4 | Security stack test-proven: 11 auth + 8 quota + 12 verification e2e | met |
| 5 | Real-model profiling complete with recorded numbers (Qwen2.5-7B Q4_K_M) | met |
| 6 | Guarantees & Limitations doc present (`docs/GUARANTEES.md`) | met |
| 7 | Toolchain pinned and build reproducible | not met — Cargo.lock and MinGW selection script present, but Rust toolchain is `stable` (no `rust-toolchain.toml`); exact compiler version not pinned |

Additional (met): phase12 7/7, phase12_e2e 12/12, ungraceful_disconnect 3/3.

## Evidence

Profiling numbers: see `docs/GUARANTEES.md` (verbatim `PROFILE` lines) and
`target/profile/profile-20261024-123330.log`.

Test summary:
```
test result: ok. 18 passed; 0 failed; 0 ignored;  (pool)
test result: ok. 11 passed; 0 failed; 0 ignored;  (auth)
test result: ok. 8 passed; 0 failed; 0 ignored;   (quota)
test result: ok. 7 passed; 0 failed; 0 ignored;   (phase12)
test result: ok. 12 passed; 0 failed; 0 ignored;  (phase12_e2e)
```
```

Wait I included a trailing ``` accidentally? Let me fix: I wrote two code fences? Let's check content: I ended with ``` then newline? Actually I wrote closing ``` after evidence block. Then extra ```? Let me view. The content ends:

```
test result: ...
```

I included one fenced block. Good. But I added a stray? Let's just rewrite cleanly.<|end|><|start|>startup to=write<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="content">
# Synapse v1.0-beta gate checklist

Per-item status as of the Phase 13 validation commit.

| # | Item | Status |
|---|------|--------|
| 1 | Fast test tier green (`scripts/test.ps1`, no model) | met |
| 2 | Model test tier green (`pool` 18/18) | met |
| 3 | No `unwrap()`/`expect()` in production code (only in `#[cfg(test)]`) | met |
| 4 | Security stack test-proven: 11 auth + 8 quota + 12 verification e2e | met |
| 5 | Real-model profiling complete with recorded numbers (Qwen2.5-7B Q4_K_M) | met |
| 6 | Guarantees & Limitations doc present (`docs/GUARANTEES.md`) | met |
| 7 | Toolchain pinned and build reproducible | not met — Cargo.lock and MinGW selection script present, but Rust toolchain is `stable` (no `rust-toolchain.toml`); exact compiler version not pinned |

Additional (met): phase12 7/7, phase12_e2e 12/12, ungraceful_disconnect 3/3.

## Evidence

Profiling numbers: see `docs/GUARANTEES.md` (verbatim `PROFILE` lines) and
`target/profile/profile-20261024-123330.log`.

Test summary:

    test result: ok. 18 passed; 0 failed; 0 ignored;  (pool)
    test result: ok. 11 passed; 0 failed; 0 ignored;  (auth)
    test result: ok. 8 passed; 0 failed; 0 ignored;   (quota)
    test result: ok. 7 passed; 0 failed; 0 ignored;   (phase12)
    test result: ok. 12 passed; 0 failed; 0 ignored;  (phase12_e2e)
