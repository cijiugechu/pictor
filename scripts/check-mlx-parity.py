#!/usr/bin/env python3
"""Check native MLX component fixtures against the pinned Python reference."""
import json
from pathlib import Path
import mlx.core as mx
import numpy as np

root = Path('outputs/mlx-port')
reference = mx.load(str(root/'python.safetensors'))
results = {}
for file in ('cpp-text', 'cpp-dit', 'cpp-vae', 'cpp-small'):
    if file == 'cpp-small':
        reference = mx.load(str(root/'python-small.safetensors'))
    for key, value in mx.load(str(root/(file+'.safetensors'))).items():
        actual = np.array(value.astype(mx.float32))
        expected = np.array(reference[key].astype(mx.float32))
        assert actual.shape == expected.shape, key
        diff = np.abs(actual-expected)
        result_key = f'{file}/{key}'
        results[result_key] = dict(max=float(diff.max()), mae=float(diff.mean()), different=int(np.count_nonzero(diff)))
        assert np.array_equal(actual, expected), (result_key, results[result_key])
tokens = [list(map(int,line.split())) for line in (root/'cpp-tokens.txt').read_text().splitlines()]
expected_tokens = json.loads((root/'python-tokens.json').read_text())
assert tokens == expected_tokens, 'token IDs differ'
results['tokens'] = [True]*len(tokens)
(root/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
print('PASS: native MLX component and Unicode tokenizer parity')
