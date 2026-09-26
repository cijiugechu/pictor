#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/vendor/stable-diffusion.cpp"
sd_revision=90e87bc846f17059771efb8aaa31e9ef0cab6f78
ggml_revision=404fcb9d7c96989569e68c9e7881ee3465a05c50
patch="$root/patches/anima-ggml-metal-im2col3d-pad.patch"
progress_patch="$root/patches/sd-model-progress-callback.patch"
spdlog="$root/vendor/spdlog"
spdlog_revision=79524ddd08a4ec981b7fea76afd08ee05f83755d

if [[ ! -e "$spdlog/.git" ]]; then
    git -C "$root" submodule update --init vendor/spdlog
fi
[[ "$(git -C "$spdlog" rev-parse HEAD)" == "$spdlog_revision" ]] || {
    echo "spdlog revision mismatch. Expected $spdlog_revision." >&2
    exit 1
}

if [[ ! -e "$src/.git" ]]; then
    git -C "$root" submodule update --init vendor/stable-diffusion.cpp
fi
[[ "$(git -C "$src" rev-parse HEAD)" == "$sd_revision" ]] || {
    echo "sd.cpp revision mismatch. Expected $sd_revision; restore the pinned submodule before building." >&2
    exit 1
}
if [[ ! -e "$src/ggml/.git" ]]; then
    git -C "$src" submodule update --init ggml
fi
[[ "$(git -C "$src/ggml" rev-parse HEAD)" == "$ggml_revision" ]] || {
    echo "ggml revision mismatch. Expected $ggml_revision; restore the pinned submodule before building." >&2
    exit 1
}

if ! git -C "$src/ggml" apply --unidiff-zero --reverse --check "$patch" 2>/dev/null; then
    git -C "$src/ggml" apply --unidiff-zero --check "$patch" || {
        echo "Anima Metal patch does not apply. Resolve local changes in the ggml submodule first." >&2
        exit 1
    }
    git -C "$src/ggml" apply --unidiff-zero "$patch"
    echo "Applied Anima Metal im2col3d/padding patch."
fi

for sd_patch in "$progress_patch" "$root/patches/sd-sampling-image-index.patch" "$root/patches/sd-flux-hidden-state-compression.patch"; do
    if git -C "$src" apply --unidiff-zero --reverse --check "$sd_patch" 2>/dev/null; then
        continue
    fi
    git -C "$src" apply --unidiff-zero --check "$sd_patch" || {
        echo "sd.cpp patch does not apply: $sd_patch. Resolve local changes first." >&2
        exit 1
    }
    git -C "$src" apply --unidiff-zero "$sd_patch"
    echo "Applied $(basename "$sd_patch")."
done
