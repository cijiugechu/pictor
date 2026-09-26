#!/usr/bin/env bash
# Prepare the default native Anima model explicitly; inference never invokes Python.
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
backend_library="${1:-$root_dir/zig-out/lib/libstable-diffusion.dylib}"
bash scripts/download-model.sh
convert_python=build/mlx-venv/bin/python
if [[ ! -x "$convert_python" ]] || ! "$convert_python" -c 'import numpy; assert numpy.__version__ == "2.5.3"' 2>/dev/null; then
    convert_python=build/anima-convert-venv/bin/python
    [[ -x "$convert_python" ]] || uv venv --python 3.12.11 build/anima-convert-venv
    uv pip install --python "$convert_python" 'numpy==2.5.3'
fi
"$convert_python" scripts/convert-anima-p3.py --library "$backend_library" --verify-existing
