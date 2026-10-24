<#
.SYNOPSIS
    Runs the Phase 13 real-model profiling scenarios.

.DESCRIPTION
    Each scenario runs in its own process so the resident set of one run cannot
    inflate the next and so the KV-cache is measured from a clean allocator.
    The raw output and the machine-readable `PROFILE key=value` lines are
    written to `target/profile/`.

    The scenarios are the `#[ignore]`d tests in:
      * crates/engine/tests/profile.rs
      * crates/net/tests/profile_verification.rs

    They are not part of any test tier: they need a multi-gigabyte model and
    minutes of CPU time. `scripts/test.ps1` never runs them.

.PARAMETER Model
    Path of the GGUF. Defaults to the Qwen2.5 7B profile model, then TinyLlama.

.PARAMETER Contexts
    Context windows measured by the memory scenario. Default 2048, 4096, 8192.

.PARAMETER Tokens
    Maximum generated tokens per measured request. Default 64.

.PARAMETER Runs
    Measured repetitions of the single, prefix and verification scenarios.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/profile.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/profile.ps1 `
        -Model models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf -Contexts 2048 -SkipVerification
#>
[CmdletBinding()]
param(
    [string]$Model = '',
    [string]$ModelName = '',
    [int[]]$Contexts = @(2048, 4096, 8192),
    [int]$Tokens = 64,
    [int]$Runs = 3,
    [int]$PrefixSentences = 12,
    [switch]$SkipMemory,
    [switch]$SkipSingle,
    [switch]$SkipConcurrent,
    [switch]$SkipPrefix,
    [switch]$SkipVerification
)

$ErrorActionPreference = 'Continue'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$modelsDir = Join-Path $repoRoot 'models'

# ---------------------------------------------------------------------------
# Toolchain: one MinGW on PATH (same rule as scripts/test.ps1)
# ---------------------------------------------------------------------------

$mingw = $env:SYNAPSE_MINGW_BIN
if (-not $mingw -and (Test-Path -LiteralPath (Join-Path $env:USERPROFILE 'msys64\mingw64\bin'))) {
    $mingw = Join-Path $env:USERPROFILE 'msys64\mingw64\bin'
}
if (-not $mingw -and (Test-Path -LiteralPath 'C:\msys64\mingw64\bin')) {
    $mingw = 'C:\msys64\mingw64\bin'
}
if (-not $mingw -or -not (Test-Path -LiteralPath (Join-Path $mingw 'gcc.exe'))) {
    throw 'No MinGW toolchain found. Set SYNAPSE_MINGW_BIN to the directory that contains gcc.exe.'
}

$stale = 'C:\MinGW\mingw64\bin'
$entries = $env:PATH -split ';' |
    Where-Object { $_ -and $_ -ne $stale -and $_ -ne $mingw }
$env:PATH = (@($mingw) + $entries) -join ';'

Write-Host "gcc: $((& (Join-Path $mingw 'gcc.exe') -dumpfullversion)) from $mingw"

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

if (-not $Model) {
    $seven = Join-Path $modelsDir 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'
    $small = Join-Path $modelsDir 'TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf'
    if (Test-Path -LiteralPath $seven) {
        $Model = $seven
    } elseif (Test-Path -LiteralPath $small) {
        $Model = $small
    } else {
        throw "No profile model found. Run scripts/download_model.ps1 or pass -Model."
    }
}
if (-not (Test-Path -LiteralPath $Model)) {
    throw "The model does not exist: $Model"
}
$Model = (Resolve-Path -LiteralPath $Model).Path

if (-not $ModelName) {
    $ModelName = [System.IO.Path]::GetFileNameWithoutExtension($Model).ToLowerInvariant()
}

$modelBytes = (Get-Item -LiteralPath $Model).Length
Write-Host ("model: {0} ({1:N1} MiB)" -f $Model, ($modelBytes / 1MB))

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

$outDir = Join-Path $repoRoot 'target\profile'
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $outDir "profile-$stamp.log"
$numbersPath = Join-Path $outDir "profile-$stamp-numbers.txt"
$summary = New-Object System.Collections.Generic.List[string]

function Write-Scenario {
    param(
        [string]$Label,
        [string[]]$Lines
    )

    $summary.Add("# $Label")
    Add-Content -LiteralPath $logPath -Value "# $Label"
    foreach ($line in $Lines) {
        Add-Content -LiteralPath $logPath -Value $line
        if ($line -match '^PROFILE ') {
            $summary.Add($line)
        }
    }
    Write-Host "# $Label"
    $Lines | Where-Object { $_ -match '^PROFILE ' } | ForEach-Object { Write-Host $_ }
}

$baseEnv = @{
    'SYNAPSE_PROFILE_MODEL' = $Model
    'SYNAPSE_PROFILE_MODEL_NAME' = $ModelName
    'SYNAPSE_PROFILE_TOKENS' = "$Tokens"
    'SYNAPSE_PROFILE_RUNS' = "$Runs"
    'SYNAPSE_PROFILE_N_CTX' = '2048'
    'SYNAPSE_PROFILE_SLOTS' = '1'
    'SYNAPSE_PROFILE_REQUESTS' = '1'
    'SYNAPSE_PROFILE_PREFIX_SENTENCES' = "$PrefixSentences"
    'SYNAPSE_PROFILE_VERIFY_TIMEOUT_S' = '600'
}

$managedKeys = @($baseEnv.Keys)

function Set-ProfileEnv {
    param([hashtable]$Overrides)

    $merged = $baseEnv.Clone()
    foreach ($key in $Overrides.Keys) {
        $merged[$key] = $Overrides[$key]
    }

    foreach ($key in $managedKeys) {
        Remove-Item -Path "env:$key" -ErrorAction SilentlyContinue
    }
    foreach ($key in $merged.Keys) {
        Set-Item -Path "env:$key" -Value $merged[$key]
    }
}

# ---------------------------------------------------------------------------
# Build the two profile binaries and resolve their paths from cargo's JSON
# ---------------------------------------------------------------------------

function Resolve-TestExe {
    param(
        [string]$Package,
        [string]$Test
    )

    $json = & cargo test -p $Package --test $Test --no-run --message-format=json 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "failed to build $Package/$Test"
    }

    $exe = $null
    foreach ($line in $json) {
        if ($line -notmatch '"reason":"compiler-artifact"') {
            continue
        }
        try {
            $artifact = $line | ConvertFrom-Json
        } catch {
            continue
        }
        if ($artifact.target.name -eq $Test -and $artifact.executable) {
            $exe = $artifact.executable
        }
    }
    if (-not $exe -or -not (Test-Path -LiteralPath $exe)) {
        throw "cargo did not report an executable for $Package/$Test"
    }
    return $exe
}

Write-Host 'building profiling harnesses...'
$engineExe = Resolve-TestExe -Package 'synapse-engine' -Test 'profile'
$netExe = $null
if (-not $SkipVerification) {
    $netExe = Resolve-TestExe -Package 'synapse-net' -Test 'profile_verification'
}

function Invoke-IgnoredTest {
    param(
        [string]$Exe,
        [string]$Test
    )

    $output = & $Exe $Test --ignored --exact --nocapture 2>&1
    $code = $LASTEXITCODE
    $lines = @($output | ForEach-Object { "$_" })
    if ($code -ne 0) {
        $lines += "ERROR: $Test exited with code $code"
    }
    return ,$lines
}

# ---------------------------------------------------------------------------
# 1. Memory and KV cache at every context window
# ---------------------------------------------------------------------------

if (-not $SkipMemory) {
    foreach ($ctx in $Contexts) {
        Set-ProfileEnv @{
            'SYNAPSE_PROFILE_N_CTX' = "$ctx"
            'SYNAPSE_PROFILE_SLOTS' = '1'
        }
        Write-Scenario "memory n_ctx=$ctx" (Invoke-IgnoredTest -Exe $engineExe -Test 'profile_memory')
    }
}

# ---------------------------------------------------------------------------
# 2. Single request: cold prefill and cache-hit TTFT
# ---------------------------------------------------------------------------

if (-not $SkipSingle) {
    Set-ProfileEnv @{
        'SYNAPSE_PROFILE_N_CTX' = '2048'
        'SYNAPSE_PROFILE_SLOTS' = '1'
    }
    Write-Scenario 'single request' (Invoke-IgnoredTest -Exe $engineExe -Test 'profile_single_request')
}

# ---------------------------------------------------------------------------
# 3. Concurrent load at capacity 2 and 4
# ---------------------------------------------------------------------------

if (-not $SkipConcurrent) {
    foreach ($slots in 2, 4) {
        Set-ProfileEnv @{
            'SYNAPSE_PROFILE_N_CTX' = '2048'
            'SYNAPSE_PROFILE_SLOTS' = "$slots"
            'SYNAPSE_PROFILE_REQUESTS' = "$slots"
        }
        Write-Scenario "concurrent slots=$slots" (Invoke-IgnoredTest -Exe $engineExe -Test 'profile_concurrent_load')
    }
}

# ---------------------------------------------------------------------------
# 4. Prefix cache on a long shared system prompt
# ---------------------------------------------------------------------------

if (-not $SkipPrefix) {
    Set-ProfileEnv @{
        'SYNAPSE_PROFILE_N_CTX' = '2048'
        'SYNAPSE_PROFILE_SLOTS' = '1'
    }
    Write-Scenario 'prefix cache' (Invoke-IgnoredTest -Exe $engineExe -Test 'profile_prefix_cache')
}

# ---------------------------------------------------------------------------
# 5. Verification cost: K=3 through the production orchestrator
# ---------------------------------------------------------------------------

if (-not $SkipVerification) {
    Set-ProfileEnv @{
        'SYNAPSE_PROFILE_N_CTX' = '2048'
        'SYNAPSE_PROFILE_TOKENS' = '32'
    }
    Write-Scenario 'verification K=3' (Invoke-IgnoredTest -Exe $netExe -Test 'profile_verification_cost')
}

# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------

Set-Content -LiteralPath $numbersPath -Value $summary
Add-Content -LiteralPath $logPath -Value "# numbers: $numbersPath"

Write-Host ''
Write-Host "raw log:    $logPath"
Write-Host "PROFILE:    $numbersPath"
Write-Host ''
Get-Content -LiteralPath $numbersPath
