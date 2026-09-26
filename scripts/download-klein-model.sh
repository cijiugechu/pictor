#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
directory="${1:-$root/models/flux2-klein-4b}"
checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}
download() {
    local repo="$1" revision="$2" file="$3" expected="$4"
    local target="$directory/${file##*/}"
    if [[ -f "$target" ]]; then
        [[ "$(checksum "$target")" == "$expected" ]] || {
            echo "Existing model has an unexpected checksum: $target" >&2; return 1;
        }
        echo "Model already verified: $target"
        return
    fi
    mkdir -p "$directory"
    echo "Downloading $file to $target"
    curl --fail --location --retry 3 --continue-at - --output "$target.part" \
        "https://huggingface.co/$repo/resolve/$revision/$file"
    [[ "$(checksum "$target.part")" == "$expected" ]] || {
        echo "Checksum mismatch; remove $target.part before downloading again." >&2; return 1;
    }
    mv "$target.part" "$target"
    echo "Verified SHA-256: $expected"
}
download leejet/FLUX.2-klein-4B-GGUF 3b1f5a9dc3abb32238b053aeb3d823c30afdacbd \
    flux-2-klein-4b-Q4_0.gguf d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4
download unsloth/Qwen3-4B-GGUF 22c9fc8a8c7700b76a1789366280a6a5a1ad1120 \
    Qwen3-4B-Q4_K_M.gguf f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a
download Comfy-Org/vae-text-encorder-for-flux-klein-4b 5f526678002e43af5551dadb73ce2e8c91b43afe \
    split_files/vae/flux2-vae.safetensors 868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3
