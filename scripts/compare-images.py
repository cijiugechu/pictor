#!/usr/bin/env python3
"""Report RGB differences, not perceptual quality. Requires Pillow and NumPy."""
import argparse
import hashlib
import json
import math
import numpy as np
from PIL import Image

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("reference")
p.add_argument("candidate")
args = p.parse_args()
a = np.asarray(Image.open(args.reference).convert("RGB"))
b = np.asarray(Image.open(args.candidate).convert("RGB"))
if a.shape != b.shape:
    raise SystemExit(f"shape mismatch: {a.shape} != {b.shape}")
d = a.astype(np.float64) - b.astype(np.float64)
mse = float(np.mean(d * d))
print(json.dumps({"reference": args.reference, "candidate": args.candidate, "shape": list(a.shape),
                  "reference_rgb_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                  "candidate_rgb_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                  "changed_channels": int(np.count_nonzero(d)), "max_delta": int(np.max(np.abs(d))),
                  "mae": float(np.mean(np.abs(d))), "psnr_db": 10 * math.log10(255**2 / mse) if mse else None,
                  "identical": mse == 0, "note": "Difference metrics, not a visual-quality score; null PSNR means identical."}, indent=2))
