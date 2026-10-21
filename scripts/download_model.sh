#!/usr/bin/env bash
#
# Downloads the TinyLlama GGUF model used by the local inference demo.
#
# The script is idempotent: if the target file already exists it is left
# untouched.
#
# Usage: bash scripts/download_model.sh

set -euo pipefail

MODEL_URL="${MODEL_URL:-https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
MODEL_FILE="${MODEL_FILE:-TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf}"

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "${script_root}")"
models_dir="${repo_root}/models"
dest_path="${models_dir}/${MODEL_FILE}"

mkdir -p "${models_dir}"

if [[ -f "${dest_path}" ]]; then
    size=$(du -h "${dest_path}" | cut -f1)
    echo "Model already present: ${dest_path} (${size})"
    exit 0
fi

echo "Downloading model from ${MODEL_URL}"
echo "Destination: ${dest_path}"

tmp_path="${dest_path}.part"
curl -L --fail --retry 3 --retry-delay 5 -o "${tmp_path}" "${MODEL_URL}"

size_bytes=$(wc -c < "${tmp_path}")
if (( size_bytes < 1048576 )); then
    echo "Downloaded file is suspiciously small (${size_bytes} bytes); aborting." >&2
    exit 1
fi

mv "${tmp_path}" "${dest_path}"
echo "Downloaded ${MODEL_FILE} (${size_bytes} bytes)"
