#!/usr/bin/env python3
"""Build a local, self-contained index of original images and measured run metadata."""
import argparse
import hashlib
import html
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/anima-mlx'

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--snapshot',type=Path,help='also save a portable JSON record of measured results')
 args=parser.parse_args()
 rows=[];cards=[];records=[];dates=set();machines=set()
 for summary in sorted(OUT.glob('*/summary.json')):
  d=summary.parent;s=json.loads(summary.read_text());m=json.loads((d/'manifest.json').read_text());a=m['settings']
  if s['returncode']:continue
  dates.add(m['started'][:10]);machines.add(m['machine'])
  label=d.name;med=s['median_seconds'];model=a.get('model') or ('P3 GGUF' if a['backend']=='ggml' else 'P3 dequantized')
  metadata=f"{model} | {a['backend']} | {a['width']}×{a['height']} | {a['steps']} steps | CFG {a['cfg']} | {a['cache']} | seed {a['seed']}"
  images=sorted(d.glob('image*.png'))
  if not images:continue
  measured=a['warmup']==1 and a['runs']>=3
  records.append({'name':label,'settings':{k:v for k,v in a.items() if k not in ('output','model')},
                  'model':str(model),'summary':s,'binary_sha256':m['binary_sha256'],
                  'model_files':[{**v,'path':Path(v['path']).name} for v in m['model_files']],
                  'images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in images}})
  rows.append('<tr><td><a href="#'+html.escape(label)+'">'+html.escape(label)+'</a></td>'+''.join('<td>'+f'{med[k]:.3f}'+'</td>' for k in ('text','sampling','decode','generation'))+'<td>'+f"{s['peak_rss_bytes']/2**30:.2f}"+'</td><td>'+('resident median' if measured else 'functional observation')+'</td></tr>')
  rel=images[0].relative_to(OUT).as_posix()
  cards.append(f'<article id="{html.escape(label)}"><h2>{html.escape(label)}</h2><p>{html.escape(metadata)}</p><p>{html.escape(a["prompt"])}</p><a href="{rel}"><img loading="lazy" src="{rel}"></a><p><a href="{d.name}/summary.json">timings</a> · <a href="{d.name}/manifest.json">settings / hashes</a> · <a href="{d.name}/run.log">raw log</a></p></article>')
 page='''<!doctype html><meta charset="utf-8"><title>Anima MLX comparison</title>
 <style>body{font:16px system-ui;margin:24px;background:#11151b;color:#e5eaf0}a{color:#8bd3ff}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #414952}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:#202630;padding:16px;border-radius:8px}img{width:100%;max-height:768px;object-fit:contain}p{line-height:1.5}h2{font-size:18px}</style>
 <h1>Anima MLX comparison</h1><p>Original output images. Click an image for full resolution. Generation excludes loading and PNG writing. Different model recipes are end-to-end comparisons, not backend-only speedups. Lower pixel difference does not prove higher quality.</p>
 <label>Filter runs <input id="filter" placeholder="e.g. portrait, public, balanced"></label>
 <table><thead><tr><th>Run</th><th>Text + adapter* (s)</th><th>Sampling (s)</th><th>Decode (s)</th><th>Total (s)</th><th>Peak RSS (GiB)</th><th>Timing type</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table><p>* MLX computes the adapter once with text; ggml includes its repeated work in sampling.</p><main>'''+''.join(cards)+'''</main><script>document.querySelector('#filter').oninput=e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('article,tbody tr').forEach(x=>x.style.display=x.textContent.toLowerCase().includes(q)?'':'none')};</script>'''
 (OUT/'comparison.html').write_text(page)
 if args.snapshot:
  parity={p.parent.name:json.loads(p.read_text()) for p in OUT.glob('*-fixture/comparison.json')}
  sources=[ROOT/'build.zig',*sorted((ROOT/'src/anima_mlx').glob('*')),
           ROOT/'benchmarks/anima.cpp',ROOT/'benchmarks/anima_mlx.cpp',
           ROOT/'scripts/convert-anima-p3.py',ROOT/'scripts/anima_gguf.py']
  args.snapshot.write_text(json.dumps({'machines':sorted(machines),'dates':sorted(dates),
                                      'scope':'Serial local measurements; different-checkpoint recipes are not backend-only comparisons.',
                                      'snapshot_source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                                      'parity':parity,'runs':records},indent=2)+'\n')
 print(OUT/'comparison.html')

if __name__=='__main__':main()
