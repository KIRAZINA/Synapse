# Runs the workspace tests with exactly one MinGW toolchain on PATH.
#
# STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139) at test-binary startup means the
# loader picked up the DLLs of a second MinGW installation. The usual cause is
# a stale C:\MinGW\mingw64\bin (gcc 8.1.0) ahead of MSYS2 (gcc 16.x). This
# script removes the stale entry and puts the selected toolchain first.
#
# Usage (everything is forwarded to `cargo test` verbatim):
#   powershell -File scripts/test.ps1
#       -> cargo test --workspace
#   powershell -File scripts/test.ps1 -p synapse-net --test phase12_e2e
#   powershell -File scripts/test.ps1 -p synapse-engine --test pool -- --ignored --test-threads=1
#
# The toolchain is selected from SYNAPSE_MINGW_BIN, then the usual MSYS2
# locations. No CmdletBinding is used on purpose: `-p` must reach cargo instead
# of being interpreted as a PowerShell parameter prefix.

$ErrorActionPreference = 'Stop'

$mingw = $env:SYNAPSE_MINGW_BIN
# Prefer the per-user MSYS2 install (the one the developer's PATH normally
# points at), then the machine-wide default. The selected toolchain must be the
# one that built the workspace: after switching, run `cargo clean -p
# synapse-ffi` once so the native llama.cpp objects are rebuilt against it.
if (-not $mingw -and (Test-Path -LiteralPath (Join-Path $env:USERPROFILE 'msys64\mingw64\bin'))) {
    $mingw = Join-Path $env:USERPROFILE 'msys64\mingw64\bin'
}
if (-not $mingw -and (Test-Path -LiteralPath 'C:\msys64\mingw64\bin')) {
    $mingw = 'C:\msys64\mingw64\bin'
}
if (-not $mingw -or -not (Test-Path -LiteralPath (Join-Path $mingw 'gcc.exe'))) {
    throw "No MinGW toolchain found. Set SYNAPSE_MINGW_BIN to the directory that contains gcc.exe."
}

$stale = 'C:\MinGW\mingw64\bin'
$entries = $env:PATH -split ';' |
    Where-Object { $_ -and $_ -ne $stale -and $_ -ne $mingw }
$env:PATH = (@($mingw) + $entries) -join ';'

Write-Host "gcc: $((& (Join-Path $mingw 'gcc.exe') -dumpfullversion)) from $mingw"
Write-Host "stale C:\MinGW entry removed: $(-not ($env:PATH -split ';' | Where-Object { $_ -eq $stale }))"

$cargoArgs = @($args)
if ($cargoArgs.Count -eq 0) {
    $cargoArgs = @('--workspace')
}

& cargo test @cargoArgs
exit $LASTEXITCODE
