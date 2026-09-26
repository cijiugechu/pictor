#!/usr/bin/env python3
"""Resident Anima ggml/native MLX comparison; every run records provenance and peak RSS."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bench',ROOT/'scripts/benchmark-klein.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path)
 p.add_argument('--backend',choices=('ggml','mlx'),default='ggml');p.add_argument('--int4',action='store_true')
 p.add_argument('--bf16-compute',action='store_true');p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=768)
 p.add_argument('--steps',type=int,default=3);p.add_argument('--cfg',type=float,default=1);p.add_argument('--seed',type=int,default=666)
 p.add_argument('--cache',choices=('none','spectrum'),default='none');p.add_argument('--runs',type=int,default=3);p.add_argument('--warmup',type=int,choices=(0,1),default=1)
 p.add_argument('--prompt',default='masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people')
 p.add_argument('--negative',default='');a=p.parse_args()
 if a.backend=='mlx' and a.cache!='none':p.error('MLX experiment does not implement Spectrum')
 if a.backend=='ggml' and (a.int4 or a.bf16_compute):p.error('MLX precision switches require --backend mlx')
 if not 1<=a.runs<=64:p.error('--runs must be 1..64')
 a.output.mkdir(parents=True,exist_ok=False)
 model=a.model or ROOT/('models/Anima-P3-Turbo-AIO-Q4_K.gguf' if a.backend=='ggml' else 'models/anima-p3-mlx-bf16')
 binary=ROOT/'zig-out/bin'/('pictor_anima_bench' if a.backend=='ggml' else 'pictor_anima_mlx_bench')
 cmd=[str(binary),'anima','--prompt',a.prompt,'--negative-prompt',a.negative,'--width',str(a.width),'--height',str(a.height),
      '--steps',str(a.steps),'--cfg-scale',str(a.cfg),'--seed',str(a.seed),'--cache',a.cache,'--count',str(a.runs),'--output',str(a.output.resolve()/'image.png')]
 env={k:v for k,v in os.environ.items() if not k.startswith('PICTOR_ANIMA_')}
 env['PICTOR_BENCH_WARMUP']=str(a.warmup)
 if a.backend=='ggml':cmd+=['--model',str(model.resolve())]
 else:
  env['PICTOR_ANIMA_WEIGHTS']=str(model.resolve())
  if a.int4:env['PICTOR_ANIMA_INT4']='1'
  if a.bf16_compute:env['PICTOR_ANIMA_BF16']='1'
 files=[model] if model.is_file() else [model/f'{n}-bf16.safetensors' for n in ('transformer','text_encoder','llm_adapter','vae')]
 if a.int4:files[0]=model/'transformer-int4.safetensors'
 manifest={'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'command':cmd,
           'model_files':[{'path':str(x),'sha256':bench.sha256(x)} for x in files],
           'binary_sha256':bench.sha256(binary),
           'runtime_hashes':{n:bench.sha256(ROOT/'zig-out/lib'/n) for n in ('libpictor.dylib','libstable-diffusion.dylib','libmlx.dylib') if (ROOT/'zig-out/lib'/n).exists()},
           'runner_sha256':bench.sha256(Path(__file__)),
           'git':bench.command_output(['git','rev-parse','HEAD']),
           'machine':bench.command_output(['sysctl','-n','machdep.cpu.brand_string']),
           'memory_bytes':int(bench.command_output(['sysctl','-n','hw.memsize'])),
           'thermal_before':bench.command_output(['pmset','-g','therm']),
           'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'sampler':'er_sde','scheduler':'smoothstep','rng':'Philox',
           'activation_policy':'bf16' if a.bf16_compute else 'reference-fp32-promotion',
           'environment':{k:v for k,v in env.items() if k.startswith('PICTOR_')}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 start=time.perf_counter()
 with (a.output/'run.log').open('w') as log:
  result=subprocess.run(['/usr/bin/time','-l']+cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 text=(a.output/'run.log').read_text();runs=bench.parse_log(text);measured=[r for r in runs if r['index']>=0]
 rss=re.search(r'(\d+)\s+maximum resident set size',text)
 summary={'returncode':result.returncode,'process_seconds':time.perf_counter()-start,'runs':runs,
          'peak_rss_bytes':int(rss[1]) if rss else None,
          'peak_footprint_bytes':int(re.search(r'(\d+)\s+peak memory footprint',text)[1]) if re.search(r'(\d+)\s+peak memory footprint',text) else None,
          'mlx_peak_allocation_bytes':max(map(int,re.findall(r'BENCH_MLX_PEAK (\d+)',text)),default=None),
          'median_seconds':{k:statistics.median(r[k] for r in measured if k in r) for k in ('text','sampling','decode','generation') if any(k in r for r in measured)},
          'thermal_after':bench.command_output(['pmset','-g','therm'])}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
 if result.returncode or len(measured)!=a.runs or any(not {'text','sampling','decode'}<=r.keys() for r in measured):raise SystemExit('benchmark failed; inspect run.log')

if __name__=='__main__':main()
