#!/usr/bin/env python3
"""Generate independent xocialize component fixtures from local pinned weights."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'build/anima-mlx-reference'))
from anima_mlx.models.cosmos_dit import CosmosTransformer3DModel,CosmosDiTConfig
from anima_mlx.models.qwen3_te import Qwen3TextEncoder
from anima_mlx.models.llm_adapter import LLMAdapter
from anima_mlx.models.wan_vae import WanVAE

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--int4',action='store_true')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 rng=np.random.default_rng(142)
 qi=mx.array([[151643,9707,11,279,1917]],mx.int32);ti=mx.array([[8774,6,8,296,1]],mx.int32)
 latent=mx.array(rng.normal(size=(1,16,8,8)).astype(np.float32))
 context=mx.array(rng.normal(size=(1,512,1024)).astype(np.float32))
 inputs={'qwen_ids':qi,'t5_ids':ti,'latent':latent,'context':context};mx.eval(inputs)
 mx.save_safetensors(str(a.output/'inputs.safetensors'),inputs)
 outputs={}
 for name,model in [('text_encoder',Qwen3TextEncoder()),('llm_adapter',LLMAdapter()),('transformer',CosmosTransformer3DModel(CosmosDiTConfig())),('vae',WanVAE())]:
  suffix='bf16'
  if name=='transformer' and a.int4:
   nn.quantize(model,group_size=64,bits=4,class_predicate=lambda p,m:isinstance(m,nn.Linear) and 'transformer_blocks' in p and ('.attn' in p or '.ff.' in p))
   suffix='int4'
  model.load_weights(str(a.model/(name+'-'+suffix+'.safetensors')),strict=True);mx.eval(model.parameters())
  if name=='text_encoder': outputs['text']=model(qi)
  elif name=='llm_adapter': outputs['context']=model(outputs['text'],ti)
  elif name=='transformer': outputs['noise']=mx.squeeze(model(mx.expand_dims(latent,2),mx.array([0.5]),context),2)
  else:
   img=model.decode(WanVAE.denormalize(mx.expand_dims(latent,2)))
   outputs['decoded']=mx.clip((img[:,:,0].transpose(0,2,3,1)+1)*0.5,0,1)
  mx.eval(outputs);print(name,[(k,str(v.dtype),v.shape) for k,v in outputs.items()],flush=True)
  del model;mx.clear_cache()
 mx.save_safetensors(str(a.output/'reference.safetensors'),outputs)

if __name__=='__main__':main()
