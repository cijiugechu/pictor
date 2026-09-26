# Klein performance experiment — 2026-09-26

Host: fanless MacBook Air, Apple M4, 32 GiB, macOS 15.7.5. These are local
measurements, not general hardware promises. See [reproduction guide](benchmarking.md)
for commands, model revisions, environment setup and trace interpretation.

## Small Decoder

Text workload: distilled Klein 4B, 512×512, seed 666, four Euler/discrete steps,
CFG 1, diffusion flash attention, HS off. Each VAE uses a separate resident session,
one warmup and three measured sequential calls. All repeats within each session
produce identical RGB bytes. Prompt: “A small red fox sitting on a mossy rock in a
sunlit forest, detailed fur, soft natural light”.

| Median seconds | Original VAE | Small Decoder |
| --- | ---: | ---: |
| Text encoding | 5.95 | 5.79 |
| Sampling | 102.94 | 100.22 |
| Decode | 14.60 | 10.31 |
| Generation | 123.693 | 116.483 |

Decode time decreased 29.4% (1.42× throughput for this stage). Observed total time
decreased 5.8%, but sampling also drifted despite unchanged denoiser work. The
decoder difference alone accounts for approximately 4.29 seconds, or 3.5% of the
baseline total. Runs were sequential full-then-small, not randomized/interleaved;
thermal and system drift remain confounders. Do not attribute the entire total
difference to the decoder. Small run reported no thermal/performance warning;
absence of a warning does not establish identical chip temperature or clocks.

All 111 non-decoder tensors in the official full-encoder/small-decoder checkpoint
match the original VAE byte-for-byte, including dtype and shape. The existing
backend detects decoder width 96 instead of 128. Existing C++, C ABI and CLI VAE
path selection already support this checkpoint; production defaults are unchanged.

The original output exactly matches the earlier pictor baseline. The Small Decoder
output preserves the fox's composition, face and limbs, without the repeated
contour/stripe artifacts seen in the separate HS experiment. Fine texture differs.
RGB MAE is 1.8672/255, PSNR 37.66 dB; these describe differences, not perceptual
quality. This is a limited pilot, not proof of quality across prompts/resolutions.

Raw local artifacts (ignored by Git):

- `outputs/performance/full-512/{manifest,summary}.json`, `run.log`, `image-001.png`
- `outputs/performance/small-512/{manifest,summary}.json`, `image-difference.json`, `image-001.png`

Generation excludes model load and PNG writing. Warmups: 139.387 seconds (full),
120.527 seconds (small); model loads: 5.096 and 3.071 seconds. Neither is included
in the median above. Independent pinned-backend verification confirmed the exported
four-step sigmas `[1, 0.938505292, 0.792578936, 0.00756923715, 0]`.

Reference-edit functional probe: use the original 512px fox as reference, generate
256×256 at seed 666/four steps with “Turn the fox white and make the forest snowy,
keeping its pose and the rock.” Both VAEs produce a white fox in a snowy forest
while preserving its seated pose. The images remain visually close; RGB MAE
3.2462/255, PSNR 31.86 dB. Original/small decode took 3.79/2.53 seconds, total
81.778/78.462 seconds. These are single cold calls, not a repeated speed benchmark.
Artifacts: `outputs/performance/{full,small}-edit-256/`.

## Metal trace

The completed profile uses original VAE, 512×512, one Euler step, no warmup. It
covers full text encoding, one denoiser forward pass and decode at the target
geometry. Generation took 50.389 seconds with instrumentation. The analyzer
filtered PID 75496 and merged 6,297 target GPU intervals:

- GPU active interval union: 47.728 seconds, 94.72% of the generation window.
- Remaining gaps: 2.661 seconds; largest gap 0.306 seconds; 388 gaps over 1 ms.
- Capture plus saving: 219.992 seconds. Raw trace and exports remain in
  `outputs/performance/metal-512-one-step/`.

This suggests that host-side idle gaps alone cannot explain most of this run's
latency. It does **not** establish high shader occupancy, efficient matrix kernels,
or a compute-versus-bandwidth bottleneck. Active intervals can contain inefficient
GPU work. The stock template has Shader Timeline disabled, so the data cannot
rank GEMM versus attention. Shader counters/captures are a separate next step.
The one-step cold profile does not establish the same duty fraction for the
four-step warmed benchmark. Trace overhead and system scheduling are included.

A full four-step capture generated about 4 GB of raw data and its postprocessing
remained CPU-bound nearly ten minutes after target exit. Only the owned profiler
process was terminated; the incomplete capture remains in
`outputs/performance/metal-512/` and is excluded from profile conclusions. Its
failure does not invalidate the earlier ordinary four-step benchmarks. Prefer
bounded profiles for iteration on this machine.

## MLX/MFLUX

Separate Python 3.12.11 environment, MFLUX 0.20.0, MLX/MLX Metal 0.32.2. Pinned
`mlx-community/flux2-klein-4b-4bit` checkpoint, affine 4-bit groups of 64, BF16
activations; stock M4 compiled denoiser, no HS/step cache. Text sequence length
is 512 in both implementations. Each MLX configuration uses one warmup and three
measured resident calls; all repeats are RGB-identical within each MLX configuration.

| Median seconds | pictor original VAE | MLX native | MLX aligned noise/sigmas |
| --- | ---: | ---: | ---: |
| Text encoding | 5.950 | 5.669 | 5.645 |
| Sampling | 102.940 | 73.941 | 73.772 |
| Decode | 14.600 | 2.461 | 2.479 |
| Generation | 123.693 | 81.896 | 82.209 |

This available MLX configuration takes 33.8% less generation time (1.51× relative
throughput) in these measurements. It does **not** isolate a backend speedup: MLX
affine weights and BF16 execution differ from GGUF Q4_0/Q4_K_M and ggml execution;
native RNG and scheduler also differ. The MLX image is a clean seated fox on a
mossy rock, with a different composition. Its native four-step sigmas are
`[1, 0.958085358, 0.883981824, 0.717496574, 0]`, versus pictor's values above.

Native warmup: 86.519 seconds; model load: 1.956 seconds. MLX-reported peak
allocation is about 7.16 GB (6.67 GiB); this is not process RSS or a comparable
ggml memory measurement. The MLX generation timer includes final image conversion
but excludes PNG writing; stage boundaries explicitly evaluate/synchronize lazy
MLX work. Artifacts: `outputs/performance/mlx-native/`.

The aligned experiment reads the exported pictor Philox noise in NCHW FP32
layout, casts it to MLX BF16, and substitutes the exact exported sigma values.
Its first output has a frontal seated fox and background closer to the pictor
composition, but is not pixel-identical (MAE 15.4118/255, PSNR 19.78 dB). This is
an alignment sanity check and a limited visual inspection, not a perceptual quality
score or proof of equal model numerics. Weight quantization, execution precision,
and possible text-encoder/kernel numerical differences remain uncontrolled.

Aligned measured generations: 81.526, 82.678, 82.209 seconds; warmup 83.650 seconds,
load 1.704 seconds. The median remains about 1.50× faster than the original pictor
configuration. Changing RNG/sigmas did not remove the measured gap. Both MLX runs
were started after all profiler/export work finished, with no concurrent inference.
The earlier 64px MLX loading smoke overlapped profiler postprocessing and is
deliberately excluded. Artifacts: `outputs/performance/mlx-aligned/`.

## Initial experiment conclusion (before native integration)

Use Small Decoder through the existing optional VAE path when its image tradeoff
is acceptable. It is a modest end-to-end improvement here, with no new backend/API
needed. Keep the original VAE available for comparison.

MLX is worth further backend prototyping: the aligned experiment retains a large
gap in sampling and a particularly large gap in decoding. First obtain a shader
or per-operator profile of the existing VAE/convolution path and compare its
precision/layout with MLX; separately profile denoiser matrix operations. Do not
assume that host dispatch tuning or the experimental HS mode will reproduce these
gains. At this experiment stage, MLX had not yet been integrated into pictor. The subsequent integration is recorded below.

Scope: one 512px text prompt, one 256px reference edit for Small Decoder, and
text-only MLX comparisons. This does not certify 1024px/multi-reference/Anima
performance or broad image quality. A future backend decision needs those workload
and quality checks. Existing Metal tests and Zig/Rust FFI checks pass (56/56 build
steps); benchmark accounting tests pass (3 cases), with Python/shell/format checks.

Local comparison figures: `outputs/performance/decoder-comparison.png`,
`outputs/performance/runtime-comparison.png`, `outputs/performance/generation-times.png`.


## Subsequent native MLX integration

Pictor now contains a native C++ MLX backend and defaults Klein to it on Apple
Silicon, with Small Decoder and the aligned Philox/Euler/discrete policy. The
historical measurements above remain the original experiments, not measurements
of the newly integrated library.

Validation artifacts are under `outputs/mlx-port/`:

- `comparison.json`: native C++ vs pinned MFLUX tensors are exactly equal for
  text embeddings, transformer prediction, original VAE encode/decode and Small
  Decoder encode/decode. Four tokenizer fixtures (ASCII, Chinese/Unicode,
  whitespace/punctuation, special token) are also identical.
- `native-full-512.png`: all RGB bytes exactly equal
  `outputs/performance/mlx-aligned/image_001.png` for the fox prompt, 512²,
  four steps and seed 666. This checks the entire aligned sampling chain.
- `native-small-512.png`: final native default, same prompt/seed/settings, with
  Small Decoder. A cold call from a relocated installation took 77.862 seconds
  (text 5.983, sampling 69.870, decode 2.008). Model load was 0.763 seconds.
  This is a functional run, not a warmup/three-repeat performance claim. Earlier
  development runs varied materially with machine state; do not compare these
  single calls as controlled backend or decoder speedups.
- `native-edit-256.png`: native Small Decoder reference edit, 256²/four steps,
  turning the fox white and forest snowy while preserving pose/composition.
  Generated from the relocated installation with a different working directory.
- Native model smoke covers rectangular output, ordered reference edits,
  batch-vs-single RGB parity, CFG, explicit decoder tiling, callbacks, invalid
  UTF-8/error recovery, C/C++ RGB parity and borrowed/owned image lifetimes.
- Native benchmark runner completed a 64²/one-step run and captured all stage
  timings (`native-benchmark/`). Its manifest pins the runtime and model hashes.

A relocated `bin/lib` installation loads libpictor/libmlx/libstable-diffusion from
its own directory and generates without a Python runtime on PATH. MLX headers,
Python environments and the original build tree are not runtime dependencies.
Current checks: Metal test/FFI plus benchmark build 69/69 steps; CPU/no-MLX test/FFI 59/59;
benchmark accounting 4 cases. Broader 1024px quality and sustained performance
remain unmeasured for the native port.
