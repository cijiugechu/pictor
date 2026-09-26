#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
directory="${1:-$root/models/flux2-klein-4b}"
target="$directory/full_encoder_small_decoder.safetensors"
expected=ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62
revision=a3efc24f613ef42d9428af62fdbd6f5fd8856c4a
checksum() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
if [[ -f "$target" ]]; then
    [[ "$(checksum "$target")" == "$expected" ]] || { echo "Checksum mismatch: $target" >&2; exit 1; }
else
    mkdir -p "$directory"
    curl --fail --location --retry 3 --continue-at - --output "$target.part" \
        "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/$revision/full_encoder_small_decoder.safetensors"
    [[ "$(checksum "$target.part")" == "$expected" ]] || { echo "Checksum mismatch: $target.part" >&2; exit 1; }
    mv "$target.part" "$target"
fi
echo "Verified: $target ($expected)"
