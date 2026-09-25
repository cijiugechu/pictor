#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$root/models/Anima-P3-Turbo-AIO-Q4_K.gguf}"
revision=ddb1225449828d4e5d69208c008b53f838e626c5
expected=3290dc9abad9cf98cf1b39b491a464bd4e7bba200ed508dcf0a7e9cd8fdb9168
url="https://huggingface.co/n-Arno/Anima-P3-Turbo-AIO-Q4_K/resolve/$revision/Anima-P3-Turbo-AIO-Q4_K.gguf"

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

if [[ -f "$target" ]]; then
    [[ "$(checksum "$target")" == "$expected" ]] || {
        echo "Existing model has an unexpected checksum: $target" >&2
        exit 1
    }
    echo "Model already verified: $target"
    exit 0
fi

mkdir -p "$(dirname "$target")"
echo "Downloading Anima P3 Turbo AIO Q4 (1.79 GB) to $target"
curl --fail --location --retry 3 --continue-at - --output "$target.part" "$url"
[[ "$(checksum "$target.part")" == "$expected" ]] || {
    echo "Checksum mismatch; remove $target.part before downloading again." >&2
    exit 1
}
mv "$target.part" "$target"
echo "Verified SHA-256: $expected"
