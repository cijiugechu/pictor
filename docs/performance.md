# Klein performance assessment — 2026-09-26

Functional baseline: commit `5b7a993`, Apple M4 / 32 GiB, pinned sd.cpp/ggml,
Klein Q4_0 + Qwen3 Q4_K_M + Flux2 VAE. This assessment does not change production
inference settings. The source and image tests are recorded in validation.md.

Follow-up experiments now have a [reproducible benchmark guide](benchmarking.md)
and [Small Decoder / Metal / MLX results](benchmark-results-2026-09-26.md).

## Where time goes

Earlier upstream reference runs, 512×512, four steps, CFG 1, Euler/discrete:

| Stage | Text-to-image | One-reference edit |
| --- | ---: | ---: |
| Reference VAE encoding | — | 6.85 s |
| Text conditioning | 3.77 s | 6.60 s |
| Denoising | 68.09 s | 120.09 s |
| VAE decode | 10.41 s | 7.84 s |
| Generation total | 82.41 s | 141.43 s |

Denoising accounts for 82.6% / 84.9% of these runs. This identifies the model
forward passes as the main target, but does **not** identify attention versus
linear/matrix operations as the dominant kernels; a GPU operator profile is needed.
Timings across the validation history vary substantially, so comparisons between
unrelated runs are not speedup measurements. Model loading is separate and is
already amortized by resident sessions and CLI batches.

## Priorities

1. **Measure kernels and attention configurations.** Current Klein enables only
   diffusion flash attention. Global `flash_attn` additionally enables Qwen3/VAE
   attention. Compare global, diffusion-only and disabled settings at the same
   prompt, seed, shape, step count and weights. Check both time and image changes,
   and include 1024/editing workloads before changing defaults. Flash attention
   is not guaranteed faster on Metal: [upstream performance guidance](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/performance.md).
   Use Metal capture to distinguish quantized matmul/dequantization, attention,
   normalization and dispatch overhead. The pinned backend already enables
   fusion, concurrency and graph optimization by default.

2. **Reuse conditioning in batches, then cache across requests.** At baseline commit `5b7a993`, Pictor
   implements `--count` as repeated calls with backend `batch_count=1`. Upstream
   `generate_image` already prepares reference latents/text embeds once before its
   native batch loop and advances seeds with `seed + b`. A Klein batch fast path
   is therefore a smaller first change than a persistent backend cache. It still
   samples images sequentially: the benefit is shared preparation, not parallel
   GPU inference. Handle all returned images, bound batch memory, and account for
   delayed first-image output (upstream decodes after sampling the whole batch),
   per-image timing and error/ownership semantics before wiring it into the API.
   Verify pixel parity for every seed, especially reference edits.

   For reuse across independent calls, cache text conditioning and reference
   encodings in the session. The
   current backend recomputes both on every `generate_image` call. This directly
   benefits seed sweeps and repeated edits of the same source. The cited logs
   put an ideal ceiling near 3.8 s for repeated text-only conditioning and
   13.5 s for repeated edit conditioning plus reference encoding, before lookup
   costs; neither is an end-to-end 2x opportunity. Klein's VAE encoding is
   deterministic (no Gaussian latent sampling), so cross-seed reuse is possible.
   Cache keys must include the model/session, input contents, effective resized
   dimensions, preprocessing, tiling and attention/precision settings. Cache
   owned values, not borrowed input pointers; bound memory with an LRU/limit.
   The public sd.cpp API exposes no prepared-conditioning handle, so cross-call reuse needs
   a contained backend change, not merely caching decoded PNGs in pictor (the
   CLI already decodes each reference once per batch).

3. **Profile and tune the dominant Metal kernels.** Q4_0 minimizes weight storage,
   not necessarily latency on every matrix shape. Compare quantized matmul to
   alternative layouts/precisions when memory permits, then tune or update the
   backend based on the measured hot operations. Converting existing Q4 weights
   to FP16 cannot recover original model precision. Do not promise gains from
   Q8/FP16 without measurements. Graphs are rebuilt for each forward; graph and
   allocator reuse should be considered only if profiling shows meaningful CPU
   overhead compared with GPU compute.

4. **Offer reference-resolution control.** Default auto-resize uses the output
   area (capped at 1024²) for *each* reference. At 1024 output, two references can
   therefore add about 8192 image tokens to the output's 4096 (plus text). A
   per-reference area cap could substantially reduce editing computation, but
   trades away source detail. This needs a separate setting: merely downscaling
   the input file is ineffective when auto-resize enlarges it again. Existing
   native-reference sizing is usable when inputs are prepared at multiples of 16.

5. **Treat approximate acceleration as an opt-in mode.** The original project's
   `flux2_sdnq_hs.py` pools image hidden states at stride 2 in single-stream blocks
   for the first three of four forwards and leaves the final forward exact.
   The experimental port now exposes a default-off session API and CLI switch.
   One 512px comparison reduced generation from 110.280 s to 92.168 s, but produced
   obvious repeated contours and striping: it is not a validated quality-preserving
   speed preset. See [the HS validation](validation.md#experimental-klein-hidden-state-compression-2026-09-26).
   This targets the expensive part of inference, especially at high
   resolutions, but its square-grid layout inference is insufficient for arbitrary
   aspect ratios and multiple references. Preserve separate text/output/reference
   ranges, rotary positions and image ordering, and evaluate quality on a varied
   set. Spectrum/other step caches and reducing four steps to three likewise
   require quality validation; the distilled [official example uses four steps
   and CFG 1](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

## Lower priority

- Changing Zig/C++ bindings, logging, PNG writes or the RGB copy does not address
  the measured dominant stages.
- CPU offload and VAE tiling primarily address memory pressure; there is no
  evidence of swapping in the recorded 32 GiB runs. Do not enable them as generic
  speed flags. Concurrent GPU requests are not a demonstrated latency win, and
  current process-wide backend callbacks require serialization.
- The pre-M5 tensor-API warning is intentional. The pinned ggml source explicitly
  notes no significant improvement on M4/M4 Max with that implementation; forcing
  `GGML_METAL_TENSOR_ENABLE` is not an established fix.
- Klein pads short text to 512 tokens in the pinned conditioner. Removing that
  padding is a numerical/model change, not automatically an equivalent shortcut.

## Fresh attention probe

Same installed sd-cli and weights, sequential processes, 512×512, four steps,
CFG 1, Euler/discrete, seed 666, the phase-1 fox prompt. The order is
diffusion-only → global → diffusion-only repeat, followed by disabled attention.
No other agent-run GPU inference was concurrent. Each process loads its own
context; generation times exclude loading. This small comparison is not a
controlled thermal/power benchmark, nor a high-resolution/editing benchmark.

| Configuration | Text | Sampling | Decode | Generation |
| --- | ---: | ---: | ---: | ---: |
| Diffusion FA (A1) | 5.37 s | 96.51 s | 12.92 s | 114.88 s |
| Global FA (B) | 4.86 s | 96.82 s | 12.61 s | 114.36 s |
| Diffusion FA (A2) | 5.14 s | 95.35 s | 12.71 s | 113.26 s |
| No FA | 7.38 s | 102.01 s | 13.29 s | 122.75 s |

Global FA falls inside the baseline variation, so there is no demonstrated
end-to-end speedup. It changes 3,531 RGB channels by one level (maximum delta 1,
mean absolute delta 0.00449 on the 0–255 scale). Both baseline runs are identical
to each other and the original text reference. Keep the current default unless
broader workloads establish a benefit and numerical changes are accepted.
Disabling FA did not help in this probe either. It changed 39,320 RGB channels
(maximum delta 5, mean absolute delta 0.05024/255). Its slower text encoding,
despite the text encoder having FA disabled in both baseline and no-FA cases,
also illustrates background variability: do not attribute the entire timing
difference to the diffusion setting.

## Native batch feasibility probe

Upstream `--batch-count 2` with two reference images, 64×64, one step, seeds
42/43 and the same prompt as the earlier CLI edit batch performed reference
encoding and text conditioning once (one log entry for each), then sampled both
seeds. Both RGB outputs match the earlier pictor per-image loop exactly. This
establishes feasibility for that case, not a universal parity guarantee or a
latency improvement measurement: it was not timed under the same system state
as the earlier wrapper run. Native total generation was 23.70 s, including
0.13 s reference encoding, 5.41 s text encoding, 17.66 s sampling and 0.48 s decode.

Recommended next implementation: add a repeatable stage-timing benchmark and a
bounded native batch path, validate larger text/edit batches and every seed,
then add session-level prepared-conditioning reuse for separate calls. Use the
resulting kernel profiles to choose the next single-image optimization. Keep
approximate token compression/reference reduction as explicit quality-speed modes.

## Implemented follow-up: bounded native batches

Anima and Klein now expose C++ `generate_batch`, with Klein `edit_batch`; the C ABI
has corresponding additive entry points. CLI `--count` uses chunks of at most
8 images and 16777216 total output pixels. Each native call prepares text and
reference conditions once, samples consecutive seeds, then decodes the results.
There is no cross-call cache. Counts beyond the CLI chunk limit encode again per
chunk; model weights remain resident. The sampling-index patch makes progress
independent of VAE tiling events.

`BatchResult::generation_seconds` / the C `batch_seconds` output measure the backend
batch invocation (including preparation, sampling and decoding; excluding model
load, queue wait and PNG writes). Per-image times are the amortized batch average.
Results return together, increasing first-image latency and retaining multiple
latents/decoded images. The output pixel cap limits batch accumulation, not total
backend/model memory. Existing single-image calls remain available.

Anima can share both positive and negative text conditions. In the real M4 256²
checks, text preparation was only 0.06–0.22 s, so eliminating one repeated encoding
did not produce a stable end-to-end speedup beyond variation. Klein text preparation
was several seconds. Its 256² native text batch encoded once in 4.96 s and matched
individual calls exactly; however sampling timings also changed between runs, so
the complete wall-time difference cannot be attributed to encoding reuse.
See [validation.md](validation.md) for reproduction commands and observations.

For the Klein 256² double-reference edit with VAE tiling, the batch reused one
3.24 s reference-encoding stage and one 6.03 s text-conditioning stage. Two-image
elapsed time was 178.796 s versus individual-call totals 189.717 s before and
190.771 s after (roughly 6% lower in this single observation). All RGB pixels
matched, including both subsequent individual requests.
