#!/usr/bin/env python3
"""Reproducible resident ggml/native-MLX benchmark. Raw stage logs and provenance stay together."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FOX = "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def command_output(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def parse_log(text):
    runs = []
    current = None
    for line in text.splitlines():
        begin = re.search(r"BENCH_BEGIN (-?\d+)(?: ([\d.]+))?", line)
        if begin:
            current = {"index": int(begin[1])}
            if begin[2]:
                current["started_unix"] = float(begin[2])
        if current is None:
            continue
        for key, pattern in {
            "text": r"(?:get_learned_condition completed, taking|MLX text encoding:) ([\d.]+)s",
            "sampling": r"(?:sampling completed, taking|MLX sampling:) ([\d.]+)s",
            "decode": r"(?:decode_first_stage completed, taking|MLX VAE decoding:) ([\d.]+)s",
            "reference_encode": r"(?:encode_first_stage completed, taking|MLX reference encoding:) ([\d.]+)s",
        }.items():
            match = re.search(pattern, line)
            if match:
                current[key] = current.get(key, 0.0) + float(match[1])
        end = re.search(r"BENCH_END (-?\d+) ([\d.]+)", line)
        if end:
            if int(end[1]) != current["index"]:
                raise ValueError("unmatched benchmark markers")
            current["generation"] = float(end[2])
            runs.append(current)
            current = None
    if current is not None:
        raise ValueError("incomplete benchmark invocation")
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new result directory")
    p.add_argument("--backend", choices=("ggml", "mlx"), default="ggml")
    p.add_argument("--vae", type=Path)
    p.add_argument("--models", type=Path)
    p.add_argument("--runs", type=int, help="measured calls: default 3, or 1 with --trace")
    p.add_argument("--warmup", type=int, choices=(0, 1), help="default 1, or 0 with --trace")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, help="default 4, or 1 with --trace to bound capture cost")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--prompt", default=FOX)
    p.add_argument("--reference", type=Path, action="append", default=[])
    p.add_argument("--trace", action="store_true", help="separate instrumented run; excluded from ordinary timing claims")
    args = p.parse_args()
    if args.models is None:
        args.models = ROOT / ("models/mlx-flux2-klein-4b-4bit" if args.backend == "mlx" else "models/flux2-klein-4b")
    if args.vae is None:
        args.vae = ROOT / "models/flux2-klein-4b" / ("full_encoder_small_decoder.safetensors" if args.backend == "mlx" else "flux2-vae.safetensors")
    if args.runs is None:
        args.runs = 1 if args.trace else 3
    if args.warmup is None:
        args.warmup = 0 if args.trace else 1
    if args.steps is None:
        args.steps = 1 if args.trace else 4
    args.output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "zig-out/bin/pictor_klein_bench"
    models = [args.models / ("transformer" if args.backend == "mlx" else "flux-2-klein-4b-Q4_0.gguf"),
              args.models / ("text_encoder" if args.backend == "mlx" else "Qwen3-4B-Q4_K_M.gguf"), args.vae]
    weight_files = [f for path in models for f in (sorted(path.glob("*.safetensors")) if path.is_dir() else [path])]
    cmd = [str(binary), "flux-klein", "--backend", args.backend, "--diffusion-model", str(models[0].resolve()), "--text-encoder", str(models[1].resolve()),
           "--vae", str(models[2].resolve()), "--prompt", args.prompt, "--seed", str(args.seed), "--steps", str(args.steps),
           "--width", str(args.width), "--height", str(args.height), "--count", str(args.runs), "--output", str(args.output.resolve() / "image.png")]
    for ref in args.reference:
        cmd += ["--ref-image", str(ref.resolve())]
    env = dict(os.environ, PICTOR_BENCH_WARMUP=str(args.warmup))
    manifest = {
        "command": cmd, "platform": platform.platform(), "machine": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": command_output(["git", "rev-parse", "HEAD"]), "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=ROOT)).hexdigest(),
        "driver_sha256": sha256(ROOT / "benchmarks/klein.cpp"), "runner_sha256": sha256(Path(__file__)),
        "thermal_before": command_output(["pmset", "-g", "therm"]),
        "submodules": command_output(["git", "submodule", "status"]), "binary_sha256": sha256(binary),
        "backend_sha256": sha256(ROOT / "zig-out/lib/libstable-diffusion.dylib"),
        "runtime": args.backend,
        "mlx_runtime_sha256": sha256(ROOT / "zig-out/lib/libmlx.dylib") if args.backend == "mlx" else None,
        "models": [{"path": str(m.resolve()), "sha256": sha256(m)} for m in weight_files],
        "references": [{"path": str(r.resolve()), "sha256": sha256(r)} for r in args.reference],
        "warmup": args.warmup, "runs_requested": args.runs, "instrumented": args.trace,
        "environment": {k: v for k, v in env.items() if k.startswith(("GGML_", "PICTOR_BENCH_", "METAL_"))},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.trace:
        cmd = ["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--no-prompt", "--output",
               str(args.output.resolve() / "metal.trace"), "--env", f"PICTOR_BENCH_WARMUP={args.warmup}",
               "--target-stdout", "-", "--launch", "--"] + cmd
    start = time.perf_counter()
    with (args.output / "run.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    runs = parse_log((args.output / "run.log").read_text())
    measured = [r for r in runs if r["index"] >= 0]
    summary = {"returncode": result.returncode, "process_seconds": time.perf_counter() - start, "runs": runs,
               "median_seconds": {k: statistics.median(r[k] for r in measured if k in r)
                                  for k in sorted({k for r in measured for k in r} - {"index", "started_unix"})}}
    load = re.search(r"BENCH_LOAD ([\d.]+)", (args.output / "run.log").read_text())
    summary["load_seconds"] = float(load[1]) if load else None
    summary["thermal_after"] = command_output(["pmset", "-g", "therm"])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if result.returncode or len(measured) != args.runs or any(not {"text", "sampling", "decode"} <= r.keys() for r in measured):
        raise SystemExit("benchmark failed; inspect run.log")


if __name__ == "__main__":
    main()
