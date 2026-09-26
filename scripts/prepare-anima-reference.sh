#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
ref_dir="$root_dir/build/anima-mlx-reference"
revision=2338e281ee533568db558963f6e3da064eb60ff0
if [[ ! -d "$ref_dir" ]]; then
    git clone https://github.com/xocialize/anima-mlx.git "$ref_dir"
    git -C "$ref_dir" checkout --detach "$revision"
fi
actual="$(git -C "$ref_dir" rev-parse HEAD)"
if [[ "$actual" != "$revision" ]]; then
    echo "Reference revision mismatch: expected $revision, got $actual" >&2
    exit 1
fi
if [[ -n "$(git -C "$ref_dir" status --porcelain)" ]]; then
    echo "Reference has local edits; use an unmodified pinned checkout for parity." >&2
    exit 1
fi
