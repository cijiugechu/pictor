#!/usr/bin/env bash
# Python installs pinned binary development files only. Inference links libmlx.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Native MLX requires Apple Silicon' >&2; exit 1; }
target=build/mlx-native
if [[ -f "$target/version" && "$(cat "$target/version")" == 0.32.2 && -f "$target/lib/libmlx.dylib" && -f "$target/lib/mlx.metallib" && -f "$target/include/mlx/mlx.h" ]]; then exit 0; fi
python=build/mlx-venv/bin/python
if [[ ! -x "$python" ]] || ! "$python" -c 'import importlib.metadata as m; assert m.version("mlx")=="0.32.2" and m.version("mlx-metal")=="0.32.2"' 2>/dev/null; then
    python=build/mlx-runtime-venv/bin/python
    [[ -x "$python" ]] || uv venv --python 3.12.11 build/mlx-runtime-venv
    uv pip install --python "$python" 'mlx==0.32.2' 'mlx-metal==0.32.2'
fi
"$python" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import shutil
root = Path('build/mlx-native')
package = distribution('mlx')
source = Path(package.locate_file('mlx'))
root.mkdir(parents=True, exist_ok=True)
for part in ('include', 'lib'):
    shutil.copytree(source/part, root/part, dirs_exist_ok=True)
license_file = next(p for p in package.files if str(p).endswith('/licenses/LICENSE'))
shutil.copyfile(package.locate_file(license_file), root/'LICENSE')
(root/'version').write_text('0.32.2\n')
PY
