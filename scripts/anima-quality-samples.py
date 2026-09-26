#!/usr/bin/env python3
"""Generate the additional Anima comparison images serially (single-call observations)."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'benchmarks/anima-cases.json').read_text())


def main():
    runs = [
        ('ggml-512-quality-none', 'landscape', ['--steps', '16']),
        ('ggml-512-portrait-balanced', 'portrait', ['--steps', '8', '--cache', 'spectrum']),
        ('p3-512-portrait-bf16', 'portrait', ['--backend', 'mlx', '--bf16-compute', '--steps', '8']),
        ('public-512-portrait-int4', 'portrait', [
            '--backend', 'mlx', '--int4', '--model', 'models/xocialize-anima-mlx',
            '--steps', '16', '--cfg', '4.5', '--negative', CASES['public_negative_prompt']]),
        ('ggml-256-weighted-cfg2', 'weighted_unicode', ['--steps', '3', '--cfg', '2']),
        ('p3-256-weighted-cfg2', 'weighted_unicode', [
            '--backend', 'mlx', '--bf16-compute', '--steps', '3', '--cfg', '2']),
        ('public-512-16-int4-bf16', 'landscape', [
            '--backend', 'mlx', '--int4', '--bf16-compute', '--model', 'models/xocialize-anima-mlx',
            '--steps', '16', '--cfg', '4.5', '--negative', CASES['public_negative_prompt']]),
    ]
    for name, case, extra in runs:
        output = ROOT / 'outputs/anima-mlx' / name
        if output.exists():
            raise SystemExit(f'{output} already exists; preserve results and choose a fresh location')
    for name, case, extra in runs:
        request = CASES[case]
        command = [sys.executable, str(ROOT / 'scripts/benchmark-anima.py'),
                   '--warmup', '0', '--runs', '1', '--output', f'outputs/anima-mlx/{name}']
        for key in ('prompt', 'seed', 'width', 'height'):
            command += ['--' + key, str(request[key])]
        if 'negative_prompt' in request:
            command += ['--negative', request['negative_prompt']]
        print(f'Running {name}', flush=True)
        subprocess.run(command + extra, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
