<#
.SYNOPSIS
    Downloads the TinyLlama GGUF model used by the local inference demo.

.DESCRIPTION
    Fetches a small, quantised GGUF model into the repository `models/`
    directory. The script is idempotent: if the target file already exists it
    is left untouched.

.EXAMPLE
    pwsh -File scripts/download_model.ps1
#>
[CmdletBinding()]
param(
    [string]$ModelUrl = 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    [string]$ModelFile = 'TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$modelsDir = Join-Path $repoRoot 'models'
$destPath = Join-Path $modelsDir $ModelFile

if (-not (Test-Path -LiteralPath $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
    Write-Host "Created models directory: $modelsDir"
}

if ((Test-Path -LiteralPath $destPath) -and -not $Force) {
    $size = (Get-Item -LiteralPath $destPath).Length
    Write-Host "Model already present: $destPath ($([math]::Round($size / 1MB, 1)) MB)"
    exit 0
}

Write-Host "Downloading model from $ModelUrl"
Write-Host "Destination: $destPath"

$tmpPath = "$destPath.part"

if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    # `curl` writes its progress meter to stderr, which Windows PowerShell
    # 5.1 surfaces as a terminating error. Silence the meter and relax the
    # error preference for the duration of the call; the exit code is checked
    # explicitly below.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & curl.exe -L --fail --retry 3 --retry-delay 5 --silent --show-error -o $tmpPath $ModelUrl
        $curlExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($curlExitCode -ne 0) {
        throw "curl failed with exit code $curlExitCode"
    }
}
else {
    Invoke-WebRequest -Uri $ModelUrl -OutFile $tmpPath -UseBasicParsing
}

$size = (Get-Item -LiteralPath $tmpPath).Length
if ($size -lt 1MB) {
    throw "Downloaded file is suspiciously small ($size bytes); aborting."
}

Move-Item -LiteralPath $tmpPath -Destination $destPath -Force
Write-Host "Downloaded $ModelFile ($([math]::Round($size / 1MB, 1)) MB)"
exit 0
