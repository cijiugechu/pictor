#!/usr/bin/env python3
"""Generate small, model-dependent MLX stage fixtures for the native C++ port."""
import json
from pathlib import Path
import mlx.core as mx
from mflux.models.flux2.variants import Flux2Klein
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

root = Path('outputs/mlx-port')
root.mkdir(parents=True, exist_ok=True)
model = Flux2Klein(model_path='models/mlx-flux2-klein-4b-4bit')
prompt = 'A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light'
text, tids = Flux2PromptEncoder.encode_prompt(prompt, model.tokenizers['qwen3'], model.text_encoder)
latents, ids, _, _ = Flux2LatentCreator.prepare_packed_latents(666, 64, 64, 1)
timestep = mx.array(1000., dtype=mx.float32)
packed = latents.transpose(0, 2, 1).reshape(1, 128, 4, 4)
pixels = mx.random.uniform(low=-1., high=1., shape=(1, 64, 64, 3), key=mx.random.key(42)).astype(mx.bfloat16)
mx.save_safetensors(str(root/'inputs.safetensors'), dict(latents=latents, ids=ids[0], text=text, timestep=timestep, packed=packed, pixels=pixels))
noise = model._predict(model.transformer)(latents=latents, latent_ids=ids, prompt_embeds=text, text_ids=tids,
    negative_prompt_embeds=None, negative_text_ids=None, guidance=1., timestep=timestep)
decoded = model.vae.decode_packed_latents(packed)
encoded = model.vae.encode(pixels.transpose(0, 3, 1, 2))
encoded = Flux2LatentCreator.patchify_latents(encoded)
encoded = (encoded-model.vae.bn.running_mean.reshape(1,128,1,1))/mx.sqrt(model.vae.bn.running_var.reshape(1,128,1,1)+1e-4)
mx.save_safetensors(str(root/'python.safetensors'), dict(text=text, noise=noise, decoded=decoded, encoded=encoded))
prompts = [prompt, '一只红狐狸，在雪地中。 café\n✨', "  (fox:1.2)  don't  12345", '<|im_start|>fox']
(root/'python-tokens.json').write_text(json.dumps([model.tokenizers['qwen3'].tokenize(p,max_length=512).input_ids.tolist()[0] for p in prompts]))
print('Reference fixtures saved', flush=True)

# Independent MFLUX Small Decoder module, with the official full encoder intact.
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.model.flux2_vae.decoder.decoder import Flux2Decoder
small = Flux2VAE()
small.decoder = Flux2Decoder(block_out_channels=(96,192,384,384))
weights = {}
for name, value in mx.load('models/flux2-klein-4b/full_encoder_small_decoder.safetensors').items():
    if name == 'bn.num_batches_tracked':
        continue
    name = name.replace('encoder.quant_conv.', 'quant_conv.').replace('decoder.post_quant_conv.', 'post_quant_conv.')
    name = name.replace('.norm_out.', '.conv_norm_out.').replace('.nin_shortcut.', '.conv_shortcut.')
    name = name.replace('.mid.block_1.', '.mid_block.resnets.0.').replace('.mid.block_2.', '.mid_block.resnets.1.')
    attention = '.mid.attn_1.' in name
    if attention:
        name = name.replace('.mid.attn_1.', '.mid_block.attentions.0.').replace('.norm.', '.group_norm.')
        for a,b in [('q','to_q'),('k','to_k'),('v','to_v'),('proj_out','to_out')]:
            name = name.replace(f'.{a}.',f'.{b}.')
    for i in range(4):
        name = name.replace(f'encoder.down.{i}.block.',f'encoder.down_blocks.{i}.resnets.')
        name = name.replace(f'encoder.down.{i}.downsample.',f'encoder.down_blocks.{i}.downsamplers.0.')
        name = name.replace(f'decoder.up.{i}.block.',f'decoder.up_blocks.{3-i}.resnets.')
        name = name.replace(f'decoder.up.{i}.upsample.',f'decoder.up_blocks.{3-i}.upsamplers.0.')
    if value.ndim == 4:
        value = value.reshape(value.shape[:2]) if attention else value.transpose(0,2,3,1)
    weights[name] = value.astype(mx.bfloat16)
small.load_weights(list(weights.items()), strict=True)
encoded = Flux2LatentCreator.patchify_latents(small.encode(pixels.transpose(0,3,1,2)))
encoded = (encoded-small.bn.running_mean.reshape(1,128,1,1))/mx.sqrt(small.bn.running_var.reshape(1,128,1,1)+1e-4)
mx.save_safetensors(str(root/'python-small.safetensors'), dict(decoded=small.decode_packed_latents(packed),encoded=encoded))
print('Small Decoder reference saved', flush=True)
