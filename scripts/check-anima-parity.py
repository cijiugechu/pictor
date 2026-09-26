#!/usr/bin/env python3
"""Compare native fixtures to independent reference; report ggml differences separately."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
 mx.set_default_device(mx.cpu)
 native=mx.load(str(a.directory/'native.safetensors'));result={}
 if not (a.directory/'reference.safetensors').is_file():raise ValueError('independent reference fixture is required')
 expected={'text','context','noise','decoded'}
 if set(native)!=expected:raise ValueError('native fixture must contain all four stages')
 for filename in ('reference.safetensors','ggml.safetensors'):
  path=a.directory/filename
  if not path.exists():continue
  reference=mx.load(str(path));result[filename]={}
  if filename=='reference.safetensors' and set(reference)!=expected:raise ValueError('reference fixture must contain all four stages')
  for k,v in reference.items():
   x=np.asarray(native[k].astype(mx.float32));y=np.asarray(v.astype(mx.float32))
   if x.shape!=y.shape:raise ValueError(f'shape mismatch {k}')
   if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError(f'nonfinite {k}')
   delta=abs(x-y)
   metric={'max_abs':float(delta.max()),'mae':float(delta.mean()),'relative_l2':float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)),
           'cosine':float(np.dot(x.ravel(),y.ravel())/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))}
   result[filename][k]=metric
   # Tight numerical tolerance for FP32 computation of the same model in two MLX frontends.
   # GGML comparisons include Q4_K->BF16 rounding and execution precision differences.
   if filename=='reference.safetensors' and not np.allclose(x,y,atol=3e-5,rtol=3e-5):raise ValueError((k,metric))
 (a.directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
