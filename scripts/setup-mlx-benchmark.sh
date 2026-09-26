#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ ! -x build/mlx-venv/bin/python ]]; then
    uv venv --python 3.12.11 build/mlx-venv
fi
uv pip sync --python build/mlx-venv/bin/python benchmarks/mlx-requirements.txt
build/mlx-venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
manifest = json.loads(Path('benchmarks/mlx-model.json').read_text())
root = Path('models/mlx-flux2-klein-4b-4bit')
snapshot_download(manifest['repo'], revision=manifest['revision'], local_dir=root,
                  allow_patterns=list(manifest['files']), max_workers=3)
for name, expected in manifest['files'].items():
    with (root / name).open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
print('Pinned MLX model files verified')
PY
