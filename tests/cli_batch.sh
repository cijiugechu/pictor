#!/usr/bin/env bash
set -euo pipefail
cli="$1"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
: > "$work/model"
"$cli" anima --model "$work/model" -p seed-colors -W 64 -H 64 --count 9 --seed 42 \
    -o "$work/image.png" > "$work/paths" 2> "$work/log"
[[ "$(wc -l < "$work/paths" | tr -d ' ')" == 9 ]]
[[ "$(grep -c 'Model loaded' "$work/log")" == 1 ]]
grep -q 'Image 9/9: step' "$work/log"
grep -q '"seed": 42,' "$work/image-001.json"
grep -q '"seed": 50,' "$work/image-009.json"
grep -q '"batch_count": 8,' "$work/image-001.json"
grep -q '"batch_count": 1,' "$work/image-009.json"
grep -q '"generation_seconds_kind": "batch_average"' "$work/image-001.json"
"$cli" flux-klein --diffusion-model "$work/model" --text-encoder "$work/model" --vae "$work/model" \
    --hs-compression -p winter -W 64 -H 64 --count 2 --seed 42 --ref-image "$work/image-001.png" \
    -o "$work/edit.png" > "$work/edits" 2> "$work/edit-log"
[[ "$(wc -l < "$work/edits" | tr -d ' ')" == 2 ]]
grep -q '"batch_count": 2,' "$work/edit-002.json"
grep -q '"mode": "reference-edit"' "$work/edit-002.json"
grep -q '"hidden_state_compression": true,' "$work/edit-002.json"
grep -q '"hidden_state_compression": false,' "$work/image-001.json"
if "$cli" anima --model "$work/model" -p null-last -W 64 -H 64 --count 2 \
    -o "$work/fail.png" > "$work/failure" 2> "$work/failure-log"; then
    echo 'invalid backend output unexpectedly succeeded' >&2
    exit 1
fi
[[ ! -s "$work/failure" && ! -e "$work/fail-001.png" && ! -e "$work/fail-002.png" ]]
echo 'PASS: CLI batch chunking, filenames, seeds, metadata, reference editing and atomic generation failure'
