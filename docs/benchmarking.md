# Reproducible Klein performance experiments

Measured observations are recorded in [the experiment report](benchmark-results-2026-09-26.md).

These are developer experiments, separate from production presets. Keep HS off.
Run only one inference process at a time. Use AC power and a stable system state;
record thermal conditions and avoid concurrent builds/downloads. This host is a
fanless M4 MacBook Air, so repeated runs can drift even without a reported thermal
warning. Model loading, warmup and instrumented runs are separate from steady-state
generation timings. Never compare unrelated historical runs as an A/B speedup.

## Small Decoder and resident Metal benchmark

```sh
zig build benchmark-build -Dbackend=metal
bash scripts/download-small-decoder.sh
python3 scripts/benchmark-klein.py --output outputs/performance/full
python3 scripts/benchmark-klein.py --output outputs/performance/small \
  --vae models/flux2-klein-4b/full_encoder_small_decoder.safetensors
```

Python 3.11+ is required for the standard-library benchmark runner. It performs one
full warmup and three measured calls in one resident C++ session. Each uses exactly
the same prompt, seed, dimensions, weights and four-step Euler/discrete schedule.
The driver checks exact RGB equality across warmup and repeats. These are sequential
single-image calls, so native batch encoding reuse does not obscure stage timing.

Each result directory must be new. It includes model/binary hashes, git revision and
diff hash, command/environment, raw logs, stage samples and medians, and PNGs. Total
generation excludes model loading, queue wait and PNG writing. Stage values come
from the pinned backend's logs (rounded to 0.01 s), whereas total generation uses
pictor's monotonic timer. Do not sum nested VAE graph/image/stage logs: the parser
uses only the outer stage total. First-run allocation/compilation stays in warmup.

Useful options: `--runs 3 --warmup 1 --width 512 --height 512 --steps 4 --seed 666`,
`--prompt '...'`, and repeated `--reference path.png`. `--warmup 0 --runs 1` is a
functional/quality probe, not a steady-state speed measurement. Reference loading
occurs before model creation; reference VAE encoding occurs on every invocation.

The optional [official Small Decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder)
download is pinned by commit and SHA-256. It includes the original encoder for
reference editing. The existing `FluxKleinOptions::vae_path`, C options `vae_path`,
and CLI `--vae` already select it. No new ABI or default is needed. Compatibility
is verified at runtime; look for `vae decoder: ch = 96` (original: 128).
Use the same final latent, or the same seed/scheduler/encoder/denoiser, to isolate
decoder differences. Pixel MAE/PSNR quantify change; they do not certify quality.

## Metal System Trace

After ordinary timing runs finish:

```sh
python3 scripts/benchmark-klein.py --output outputs/performance/metal-trace \
  --trace --steps 1 --warmup 0 --runs 1
python3 scripts/analyze-metal-trace.py outputs/performance/metal-trace
open outputs/performance/metal-trace/metal.trace
```

Requires full Xcode with `xcrun xctrace` and the Metal System Trace template. Wait
for recording **and saving** to finish before export. Trace runs are instrumented
and cannot supply ordinary speedup claims. `--trace` defaults to one step, one call
and no warmup; explicit flags override these. The example uses one step at the same
512px geometry to bound capture cost; it is not a four-step end-to-end profile.
A full four-step capture on this host reached about 4 GB and postprocessing was
interrupted after nearly ten minutes without completion. Do not
start ordinary timed runs while that CPU/disk work is still active. The analyzer filters the target process,
unions overlapping GPU intervals, and reports active time and gaps within generation
windows. Raw XML remains alongside the trace.

GPU active fraction is not GPU occupancy, throughput efficiency, or an individual
kernel breakdown. The stock template here has Shader Timeline disabled; its encoder
labels alone cannot establish whether GEMM or attention dominates. Inspect the
Instruments GPU timeline and enable shader profiling in an appropriate Xcode
template for that next level. Likewise gaps can include CPU preparation, waits and
other apps competing for the GPU. Do not attribute all gaps to graph construction.

ggml also supports `GGML_METAL_CAPTURE_COMPUTE=N` plus `MTL_CAPTURE_ENABLED=1` for an
Xcode `.gputrace` (written under `/tmp/perf-metal-PID.gputrace`). Its counter is per
backend instance, so identify the captured component from logs instead of assuming
that global call N denotes denoiser step N. This is a diagnostic option, not a speed
flag. Metal fusion/concurrency/graph optimization are already enabled by default.

## MLX/MFLUX comparison

```sh
bash scripts/setup-mlx-benchmark.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 build/mlx-venv/bin/python \
  scripts/benchmark-mlx.py --output outputs/performance/mlx-native
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 build/mlx-venv/bin/python \
  scripts/benchmark-mlx.py --output outputs/performance/mlx-aligned \
  --aligned-inputs outputs/performance/full
```

The isolated environment and approximately 4.6 GB MLX weights do not affect pictor
build/runtime dependencies. Python dependencies and model revision/file hashes are
pinned under `benchmarks/`. The model is the distilled 4B variant, not base or 9B.
MFLUX's stock M4 compiled denoiser path is used with no HS or step caching. Explicit
MLX evaluation/synchronization at stage boundaries prevents lazy work from being
charged to the wrong stage. Warmup and model loading are recorded separately.

The native comparison measures an available implementation, **not backend-only
parity**. MLX affine 4-bit transformer/text weights and BF16 activations differ from
ggml Q4_0/Q4_K_M and its execution precision. Default random generators and timestep
sequences also differ, even for the same seed. The aligned experiment injects the
benchmark's exported Philox noise (NCHW float32, cast to MLX BF16) and the exact
pinned discrete sigma values. The schedule exporter mirrors the pinned backend's
float32 formula; changing backend/scheduler requires revalidation. Alignment does
not remove quantization/precision/tokenization/kernel differences and does not
imply pixel equality. This initial MLX driver covers text-to-image; it does not yet
establish cross-runtime reference-edit parity.

## Lightweight verification

```sh
python3 tests/benchmark_test.py
zig build test test-ffi -Dbackend=metal
```

The Python tests guard against VAE timing double-counting, incomplete runs and
overlapping GPU intervals. Model-dependent benchmarks remain opt-in.


## Native MLX backend

The same resident driver and stage accounting now support the integrated C++ MLX
backend. The runner keeps ggml/original-VAE defaults for reproducing older results;
select MLX explicitly to use the new Small Decoder default:

```sh
zig build benchmark-build
build/mlx-venv/bin/python scripts/benchmark-klein.py --backend mlx --output outputs/performance/pictor-mlx
# Same one-warmup/three-run policy; --reference and --trace also apply.
```

The manifest hashes each safetensors shard and the installed MLX runtime.
`--vae models/mlx-flux2-klein-4b-4bit/vae` selects the original MLX VAE.
Avoid concurrent GPU workloads when collecting timings. The 64px/one-step native
MLX runs in `outputs/mlx-port/native-benchmark` and `native-final-repeat` check the runner's accounting and exact resident repeats;
it is not a performance comparison with the published 512px measurements.
