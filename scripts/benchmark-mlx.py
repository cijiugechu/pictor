#!/usr/bin/env python3
"""Isolated MFLUX baseline, with explicit stage synchronization and optional ggml inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np
from mflux.models.flux2.variants import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.callbacks.generation_context import GenerationContext

ROOT = Path(__file__).resolve().parents[1]
FOX = "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light"


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def command_output(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/mlx-flux2-klein-4b-4bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned-inputs", type=Path, help="ggml benchmark directory containing noise.f32 and sigmas.json")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, choices=(0, 1), default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--prompt", default=FOX)
    args = parser.parse_args()
    if not 1 <= args.runs <= 64 or not 1 <= args.steps <= 100 or any(not 16 <= n <= 4096 or n % 16 for n in (args.width, args.height)):
        parser.error("runs 1..64, steps 1..100 and dimensions 16..4096 divisible by 16 required")
    args.output.mkdir(parents=True, exist_ok=False)
    timings = {}
    schedule = []
    original_schedule = FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas
    initial_noise = None
    if args.aligned_inputs:
        initial_noise = np.fromfile(args.aligned_inputs / "noise.f32", dtype="<f4").reshape(1, 128, args.height // 16, args.width // 16)
        schedule = json.loads((args.aligned_inputs / "sigmas.json").read_text())
        if len(schedule) != args.steps + 1:
            raise ValueError("aligned schedule length mismatch")
        prior = json.loads((args.aligned_inputs / "manifest.json").read_text())["command"]
        for flag, value in [("--seed", str(args.seed)), ("--width", str(args.width)), ("--height", str(args.height))]:
            if prior[prior.index(flag) + 1] != value:
                raise ValueError(f"aligned input mismatch: {flag}")

        def prepare_latents(seed, height, width, batch_size, num_latents_channels=32, vae_scale_factor=8):
            if (seed, height, width, batch_size) != (args.seed, args.height, args.width, 1):
                raise ValueError("unexpected latent configuration")
            noise = mx.array(initial_noise).astype(ModelConfig.precision)
            return noise, Flux2LatentCreator.prepare_grid_ids(noise, t_coord=0), height // 16, width // 16

        Flux2LatentCreator.prepare_latents = staticmethod(prepare_latents)

    def get_schedule(image_seq_len, num_inference_steps, num_train_timesteps=1000):
        nonlocal schedule
        if args.aligned_inputs:
            sigmas = mx.array(schedule, dtype=mx.float32)
            return sigmas[:-1] * num_train_timesteps, sigmas
        result = original_schedule(image_seq_len, num_inference_steps, num_train_timesteps)
        mx.eval(result)
        schedule = result[1].tolist()
        return result

    FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas = staticmethod(get_schedule)
    original_encode = Flux2PromptEncoder.encode_prompt

    def encode(*a, **kw):
        mx.synchronize()
        start = time.perf_counter()
        result = original_encode(*a, **kw)
        mx.eval(result)
        timings["text"] = time.perf_counter() - start
        timings["text_shape"] = list(result[0].shape)
        return result

    Flux2PromptEncoder.encode_prompt = staticmethod(encode)
    original_decode = Flux2VAE.decode_packed_latents

    def decode(self, *a, **kw):
        mx.synchronize()
        start = time.perf_counter()
        result = original_decode(self, *a, **kw)
        mx.eval(result)
        timings["decode"] = time.perf_counter() - start
        return result

    Flux2VAE.decode_packed_latents = decode
    before, after = GenerationContext.before_loop, GenerationContext.after_loop
    loop_start = 0

    def before_loop(self, latents, **kw):
        nonlocal loop_start
        before(self, latents, **kw)
        mx.eval(latents)
        mx.synchronize()
        loop_start = time.perf_counter()

    def after_loop(self, latents):
        mx.eval(latents)
        mx.synchronize()
        timings["sampling"] = time.perf_counter() - loop_start
        after(self, latents)

    GenerationContext.before_loop, GenerationContext.after_loop = before_loop, after_loop
    manifest = {"platform": platform.platform(), "python_version": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ("mflux", "mlx", "mlx-metal", "numpy")},
                "machine": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "runner_sha256": sha256(Path(__file__)),
                "thermal_before": command_output(["pmset", "-g", "therm"]),
                "model": str(args.model.resolve()), "precision": str(ModelConfig.precision), "quantization": "MLX affine 4-bit (not GGUF Q4_0/Q4_K_M)",
                "model_files": {str(p.relative_to(args.model)): sha256(p) for p in sorted(args.model.rglob("*.safetensors"))},
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "aligned_noise_sha256": sha256(args.aligned_inputs / "noise.f32") if args.aligned_inputs else None,
                "notes": "Text and VAE stages are explicitly evaluated; compiled denoiser uses stock MFLUX policy. No HS or cross-step cache. Aligned FP32 noise is cast to BF16. Weight quantization, activation precision and kernels still differ."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pinned = json.loads((ROOT / "benchmarks/mlx-model.json").read_text())
    for name, digest in manifest["model_files"].items():
        if pinned["files"].get(name) != digest:
            raise ValueError(f"model differs from pinned 4-bit checkpoint: {name}")
    start = time.perf_counter()
    model = Flux2Klein(model_path=str(args.model.resolve()))
    mx.eval(model.parameters())
    mx.synchronize()
    load = time.perf_counter() - start
    print(f"MLX load: {load:.3f}s", flush=True)
    runs = []
    previous = None
    for i in range(-args.warmup, args.runs):
        timings.clear()
        mx.reset_peak_memory()
        start = time.perf_counter()
        image = model.generate_image(seed=args.seed, prompt=args.prompt, num_inference_steps=args.steps,
                                     height=args.height, width=args.width, guidance=1.0)
        mx.synchronize()
        row = dict(timings, index=i, generation=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory())
        runs.append(row)
        pixels = image.image.tobytes()
        if previous is not None and previous != pixels:
            raise RuntimeError("identical resident MLX calls produced different pixels")
        previous = pixels
        if i >= 0:
            image.image.save(args.output / f"image_{i + 1:03}.png")
        print(json.dumps(row), flush=True)
        summary = {"load_seconds": load, "sigmas": schedule, "runs": runs,
                   "median_seconds": {k: statistics.median(r[k] for r in runs if r["index"] >= 0)
                                      for k in ("generation", "text", "sampling", "decode")} if i >= 0 else {}}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["thermal_after"] = command_output(["pmset", "-g", "therm"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
