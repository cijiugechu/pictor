#!/usr/bin/env python3
"""Stream the pinned Turbo P3 GGUF into canonical Anima MLX safetensors.

Q4_K dequantization uses the exact linked ggml implementation. No re-quantization.
Key mapping follows xocialize/anima-mlx (MIT, see licenses/anima-mlx.txt).
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
import numpy as np
from anima_gguf import GGUF

ROOT = Path(__file__).resolve().parents[1]
BLOCK = {
 'self_attn.q_proj':'attn1.to_q','self_attn.k_proj':'attn1.to_k','self_attn.v_proj':'attn1.to_v',
 'self_attn.output_proj':'attn1.to_out.0','self_attn.q_norm':'attn1.norm_q','self_attn.k_norm':'attn1.norm_k',
 'cross_attn.q_proj':'attn2.to_q','cross_attn.k_proj':'attn2.to_k','cross_attn.v_proj':'attn2.to_v',
 'cross_attn.output_proj':'attn2.to_out.0','cross_attn.q_norm':'attn2.norm_q','cross_attn.k_norm':'attn2.norm_k',
 'mlp.layer1':'ff.net.0.proj','mlp.layer2':'ff.net.2',
 'adaln_modulation_self_attn.1':'norm1.linear_1','adaln_modulation_self_attn.2':'norm1.linear_2',
 'adaln_modulation_cross_attn.1':'norm2.linear_1','adaln_modulation_cross_attn.2':'norm2.linear_2',
 'adaln_modulation_mlp.1':'norm3.linear_1','adaln_modulation_mlp.2':'norm3.linear_2'}
TOP = {'x_embedder.proj.1.weight':'patch_embed.proj.weight',
 't_embedder.1.linear_1.weight':'time_embed.t_embedder.linear_1.weight',
 't_embedder.1.linear_2.weight':'time_embed.t_embedder.linear_2.weight',
 't_embedding_norm.weight':'time_embed.norm.weight',
 'final_layer.adaln_modulation.1.weight':'norm_out.linear_1.weight',
 'final_layer.adaln_modulation.2.weight':'norm_out.linear_2.weight','final_layer.linear.weight':'proj_out.weight'}


def mapping(name):
    if name.startswith('text_encoders.llm.model.'):
        return 'text_encoder',name.removeprefix('text_encoders.llm.model.')
    if name.startswith('model.diffusion_model.'):
        k=name.removeprefix('model.diffusion_model.').removeprefix('net.')
        if k.startswith('llm_adapter.'):
            return 'llm_adapter',k.removeprefix('llm_adapter.')
        if k in TOP: return 'transformer',TOP[k]
        m=re.fullmatch(r'blocks\.(\d+)\.(.+)\.weight',k)
        if not m: raise ValueError(name)
        return 'transformer',f'transformer_blocks.{m[1]}.{BLOCK[m[2]]}.weight'
    k=name.removeprefix('first_stage_model.')
    if k.startswith('encoder.') or k.startswith('conv1.'): return None
    k=re.sub(r'^conv2\.', 'post_quant_conv.conv.', k)
    k=re.sub(r'^decoder\.conv1\.', 'decoder.conv_in.conv.', k)
    k=k.replace('decoder.head.0.gamma','decoder.norm_out.gamma').replace('decoder.head.2.','decoder.conv_out.conv.')
    for old,new in [('middle.0','mid_block.resnets.0'),('middle.1','mid_block.attentions.0'),('middle.2','mid_block.resnets.1')]:
        k=k.replace('decoder.'+old+'.','decoder.'+new+'.')
    m=re.match(r'decoder\.upsamples\.(\d+)\.(.+)',k)
    if m:
        i=int(m[1]); tail=m[2]
        base=f'decoder.up_blocks.{i//4}.'
        if i%4==3:
            k=base+'upsamplers.0.'+tail.replace('resample.1.','resample.0.').replace('time_conv.','time_conv.conv.')
        else: k=base+f'resnets.{i%4}.'+tail
    for old,new in [('residual.0.gamma','norm1.gamma'),('residual.2.','conv1.conv.'),
                    ('residual.3.gamma','norm2.gamma'),('residual.6.','conv2.conv.'),('shortcut.','conv_shortcut.conv.')]:
        k=k.replace(old,new)
    return 'vae',k


def digest(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'models/Anima-P3-Turbo-AIO-Q4_K.gguf')
    p.add_argument('--output',type=Path,default=ROOT/'models/anima-p3-mlx-bf16')
    p.add_argument('--dtype',choices=('bf16','f32'),default='bf16')
    p.add_argument('--library',type=Path,default=ROOT/'zig-out/lib/libstable-diffusion.dylib')
    p.add_argument('--verify-existing',action='store_true',help='verify and reuse a complete existing conversion')
    a=p.parse_args()
    sha=digest(a.source)
    if sha!='3290dc9abad9cf98cf1b39b491a464bd4e7bba200ed508dcf0a7e9cd8fdb9168':
        raise ValueError('conversion is validated only for the pinned P3 checkpoint')
    if a.output.exists() and a.verify_existing:
        saved=json.loads((a.output/'conversion.json').read_text())
        expected={f'{name}-{a.dtype}.safetensors' for name in ('transformer','llm_adapter','text_encoder','vae')}
        if saved['sha256']!=sha or saved['dtype']!=a.dtype or saved['requantized'] or {f['path'] for f in saved['files']}!=expected:
            raise ValueError('existing conversion manifest does not match requested model/dtype')
        if (a.output/'transformer-int4.safetensors').exists():
            raise ValueError('unexpected INT4 transformer in dequantized P3 directory')
        for f in saved['files']:
            path=a.output/f['path']
            if path.stat().st_size!=f['bytes'] or digest(path)!=f['sha256']:
                raise ValueError(f'existing conversion checksum mismatch: {path}')
        print(f'Verified existing {a.dtype} P3 weights: {a.output}',flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=False)
    r=GGUF(a.source,a.library)
    mapping_groups={k:[] for k in ('transformer','llm_adapter','text_encoder','vae')}
    for src in r.tensors:
        dest=mapping(src)
        if dest: mapping_groups[dest[0]].append((src,dest[1]))
    manifest={'source':str(a.source.resolve()),'sha256':sha,'dtype':a.dtype,'requantized':False,'files':[]}
    for group,pairs in mapping_groups.items():
        header={}; offset=0
        raw=a.output/(group+'.raw')
        with raw.open('wb') as out:
            for src,dst in pairs:
                x=r.tensor(src)
                if group=='vae':
                    if dst.endswith('.gamma'): x=x.reshape(-1)
                    elif dst.endswith('.weight') and '.conv.' in dst:
                        # GGUF folds the O/I dimensions of original OITHW into one axis.
                        out_channels=r.tensors[src.removesuffix('weight')+'bias'][0][0]
                        x=x.reshape(out_channels,-1,*x.shape[-3:]).transpose(0,2,3,4,1)
                    elif dst.endswith('.weight') and x.ndim==4:
                        x=x.transpose(0,2,3,1)
                x=np.ascontiguousarray(x,dtype=np.float32)
                if not np.isfinite(x).all(): raise ValueError(f'nonfinite tensor {src}')
                if a.dtype=='bf16':
                    # IEEE round-to-nearest-even, rather than truncating mantissas.
                    bits=x.view(np.uint32)
                    data=((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')
                else: data=x.astype('<f4',copy=False)
                if dst in header: raise ValueError(f'duplicate destination {dst}')
                header[dst]={'dtype':'BF16' if a.dtype=='bf16' else 'F32','shape':list(x.shape),
                             'data_offsets':[offset,offset+data.nbytes]}
                out.write(data.tobytes()); offset+=data.nbytes
        blob=json.dumps(header,separators=(',',':')).encode()
        blob+=b' '*((-len(blob))%8)
        path=a.output/(group+'-'+a.dtype+'.safetensors')
        with path.open('wb') as out,raw.open('rb') as inp:
            out.write(struct.pack('<Q',len(blob))); out.write(blob)
            while chunk:=inp.read(8*1024*1024): out.write(chunk)
        raw.unlink()
        manifest['files'].append({'path':path.name,'sha256':digest(path),'tensors':len(header),'bytes':path.stat().st_size})
        print(f'{group}: {len(header)} tensors, {offset/2**30:.3f} GiB',flush=True)
    (a.output/'conversion.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
