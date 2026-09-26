#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python3 - "${1:-models/mlx-flux2-klein-4b-4bit}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
manifest = json.loads(Path('benchmarks/mlx-model.json').read_text())
root = Path(sys.argv[1])
for name, expected in manifest['files'].items():
    target = root/name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.part')
    file = target if target.exists() else partial
    if not target.exists():
        subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-', '--output',str(partial),
                        f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"],check=True)
    digest = hashlib.sha256()
    with file.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): digest.update(chunk)
    if digest.hexdigest() != expected: raise SystemExit(f'Checksum mismatch: {file}')
    if file == partial: partial.rename(target)
    print(f'Verified: {target}',flush=True)
PY
bash scripts/download-small-decoder.sh
