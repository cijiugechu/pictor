# Initial validation — 2026-09-26

## Native Anima public integration

- Additive C++/C backend selection/query and uncached request initialization
  preserve existing option/request layouts and explicit GGUF behavior. MLX-enabled
  Apple Silicon CLI builds now default to converted P3 BF16.
- Metal tests/FFI + Anima real-model smoke passed 74/74 build steps. CPU/no-MLX
  tests/FFI passed 60/60; updated Zig/Rust imports passed 31/31 CPU steps.
  Klein's actual C/C++ batch/edit smoke plus FFI passed 42/42. Final Anima smoke
  with explicit single/C batch callback checks passed 29/29.
- Anima checks cover batch/single and concurrent-call exact RGB, consecutive/random
  seeds, callbacks, C/C++ pixels, CFG0/CFG2, tiled edge sizes, malformed weights/
  UTF-8, recovery, image lifetime and the retained public INT4 checkpoint.
- Default 512×768/8-step CLI output is **RGB-exact** against
  `outputs/anima-mlx/p3-512-balanced-bf16/image-001.png`. Explicit ggml 256px/3-step
  output is **RGB-exact** against `outputs/anima-mlx/ggml-256-probe/image.png`.
  New PNGs/sidecars are under `outputs/anima-api/`. The 512px integration call took
  72.33s under uncontrolled load: a functional observation, not a new benchmark.
- `zig build download-model` verified/reused the source and BF16 export. Corrupt
  conversion data is rejected. A relocated install ran with `PATH=/usr/bin:/bin`
  without Python; dyld logs confirmed all project dylibs loaded from that install.
  CPU builds reject explicit MLX with a clear error.
- [Setup and API examples](anima-mlx.md) cover the new default, legacy preset
  migration, explicit ggml selection and approximate tiled decode.

## Follow-up: Klein reference-image editing

The same three Klein weight files now support reference editing, with the full
VAE encoder/decoder loaded in each Klein session. Existing C ABI structs and
entry points are unchanged; edit options, RGB views and image loading are additive.

- Metal and CPU `zig build test test-ffi` passed **51/51 steps**. Fixtures cover
  PNG alpha conversion, JPEG decoding, corrupted input, image ownership, invalid
  RGB lengths/dimensions, extreme aspect ratios, CLI reference ordering/limits,
  early missing-input failure, C edit errors and both Zig/Rust example modes.
- The test backend exercises multiple ordered RGB references, input immutability,
  output/reference aliasing in C++, edit-to-text reuse, callback isolation and
  C output lifetime after session destruction.
- `zig build smoke-klein-edit` passed **23/23 steps** on Metal. Real C++ and C ABI
  edits have identical RGB pixels and both match upstream sd-cli exactly:
  **0 changed RGB channels, maximum delta 0**. The source image is unchanged.
  Callbacks, invalid-buffer recovery and output ownership also pass.
- Text generation immediately after an edit, in the same resident C++ session,
  still matches the phase-1 text-to-image baseline exactly (0 changed channels).
  Previously compiled C/C++ consumers also continue to pass their API checks.
- A real CLI batch used two ordered 512×512 reference images, auto-resized for
  64×64 one-step output, with seeds 42/43. Both images succeeded with one model
  load and distinct outputs. Ordered absolute reference paths, edit mode, resize
  flag and seeds in JSON were verified; stdout contains exactly two image paths.
  This checks multi-reference integration, not image quality at that resolution.

Reference: phase-1 `outputs/klein-reference.png` (512×512 fox). Prompt:
“Change the fox's fur to white and the forest to a snowy winter scene. Keep the
fox's pose and composition.” All runs use seed 666, 512×512, four steps,
CFG 1, Euler/discrete, diffusion flash attention, auto-resized references,
no cache and no VAE tiling. Visual inspection found white fur and a snowy forest
with the seated pose/rock composition retained. The output is a model-generated
edit, not a pixel-preservation or masked-edit guarantee.

| Run | Generation |
| --- | ---: |
| Upstream reference edit | 141.43 s |
| C++ edit | 113.027 s |
| Text generation after edit | 61.062 s |
| C ABI edit | 114.195 s |

These M4 timings are functional observations with uncontrolled system load and
cache state. They do not establish a speedup. CPU inference and real editing
from the standalone Zig/Rust examples were not exercised; those examples were
compiled and their ABI initialization/link/error paths checked.

## Follow-up: Klein 4B distilled, phase 1

The existing pinned sd.cpp revision and Metal patch are unchanged. A separate
one-line patch makes split-file loading respect the installed progress callback
instead of writing stray separator newlines to stdout. It does not change model
loading or sampling calculations. Klein uses
the three pinned files in [klein.md](klein.md), all downloaded and SHA-256
verified, on the same Apple M4 / 32 GiB host.

- Metal and CPU `zig build test test-ffi` each passed 46/46 build steps. New
  checks cover split paths, model defaults/restrictions, additive C ABI sizes
  and errors, and both Zig/Rust example modes. CPU inference was not exercised.
- A deterministic backend fixture runs Anima and Klein from competing threads.
  It checks shared serialization, callback userdata isolation, model parameter
  routing, error recovery and image ownership.
- `zig build smoke-klein` passed 22/22 build steps. Two identical C++ requests
  reuse one resident context and produce identical RGB pixels. A fresh C ABI
  context rejects Spectrum, then generates the same image successfully. Progress
  callbacks (0..4), image access and PNG writing after session destruction pass.
- Both the C++ and C ABI PNGs match the pinned upstream `sd-cli` reference
  exactly: **0 changed RGB channels; maximum delta 0**. The reference image was
  visually inspected. This establishes parity only for the tested files,
  backend, build, device and request.
- Anima `zig build smoke-c` passed 22/22 build steps after the shared-runtime
  refactor. Its repeated 512×768, 3-step no-cache generation still matches the
  original upstream baseline exactly (0 changed channels). Load was 3.408 s;
  generation was 59.697 s and 49.674 s.
- Existing, previously compiled C11 and C++17 clients run against the new
  library. At that stage, the old C ABI layouts/signatures and Anima defaults remained intact;
  new Klein C and C++ symbols are exported. Zig/Rust Klein examples were checked
  without weights; real Klein FFI inference was exercised through the C ABI
  smoke test.

The 512×512 parity request uses the prompt “A small red fox sitting on a mossy
rock in a sunlit forest, detailed fur, soft natural light”, seed 666, 4 steps,
CFG 1, Euler/discrete, no cache, and diffusion-only flash attention. Neither
parameter CPU offload nor VAE tiling is enabled. Timings are development
observations with uncontrolled system load and cache state, not benchmarks.

| Run | Resolution | Load | Generation |
| --- | --- | ---: | ---: |
| Upstream sd-cli | 512×512 | included in 90.72 s wall time | 82.41 s |
| C++ first request | 512×512 | 8.912 s | 96.316 s |
| C++ resident repeat | 512×512 | reused | 90.858 s |
| C ABI fresh context | 512×512 | not recorded | 93.198 s |
| CLI | 1024×1024 | 7.631 s | 331.786 s |

The upstream reference's maximum process RSS was 8,182,153,216 bytes. This is
process accounting, not a complete measurement of GPU/unified memory use.
The 1024×1024 CLI run took 339.99 s wall time with maximum RSS 8,721,498,112 bytes
and peak physical footprint 8,172,723,064 bytes. Its image was visually inspected;
PNG dimensions and JSON model paths, settings, seed and timings were checked.
That run exposed two stray stdout newlines from the upstream split-file loader,
which prompted the progress-callback patch described above.
After rebuilding both backends with that patch, Metal and CPU tests again passed
46/46 steps. A real 64×64, 1-step Klein CLI run then passed strict stdout checking
(exactly one PNG path plus its newline), stderr severity/progress checks and
PNG/JSON validation. Patch preparation was also run twice to verify idempotence.
Image editing, base/9B checkpoints, other quantizations, CUDA, SDNQ/Quanto
loading and MLX/hidden-state accelerations remain outside this phase.

## Follow-up: C ABI and explicit C++ errors

Pictor's C++ APIs now return `Status` with output parameters. The C++ types remain;
callers must migrate away from the earlier throwing constructor/generate API.
Production project code, CLI and test consumers compile with `-fno-exceptions`.
Only the private `src/backend.cpp` boundary retains catch handlers for the
unchanged upstream dependency. spdlog/fmt use their no-exception modes.

- Metal `zig build test test-ffi`: 41/41 build steps passed. Tests cover C header
  compilation, struct-size rejection, invalid values, NULLs, optional error
  buffers, explicit C++ errors, CLI behavior, and Zig/Rust layout/link checks.
- Final combined `zig build test test-ffi smoke-c reference`: 48/48 build steps
  passed. C ABI real inference generated two identical 512×768, 3-step, no-cache
  images with seed 666 from one resident session. Callback userdata, step 0..3,
  invalid-request recovery, occupied-handle rejection, image lifetime after
  session destruction, and PNG error recovery all passed.
- The C ABI PNG matches the original upstream reference exactly: 0 changed RGB
  channels, maximum delta 0. Observed load was 3.523 s, generation 51.543 s and
  51.920 s; warmed OS caches and uncontrolled system load make these functional
  timings rather than benchmarks.
- CPU `zig build test test-ffi -Dbackend=cpu --prefix /tmp/pictor-cabi-cpu`:
  41/41 build steps passed. Inference remains tested on Metal.
- A dedicated backend fault fixture injects `bad_alloc`, ordinary exceptions,
  unknown exceptions, NULL returns and oversized messages. A no-exception C++
  caller receives the expected error codes and bounded messages.
- Independent Apple Clang C11 and C++17 (`-fno-exceptions`) clients linked the
  installed headers/library and passed API error checks. The host's default
  deployment target is older than Zig's native 15.7.5 target, producing only
  a deployment-target linker warning.
- Export inspection confirms unmangled `pictor_*` functions alongside C++
  exports. Runtime dependencies remain system libc++, the backend, and libSystem;
  the installed library uses `@loader_path`, without build-cache rpaths.
- The Zig 0.16.0 and Rust 1.98.1 examples each generated a real 512×768 PNG via
  the C ABI, including progress userdata and pixel-buffer access. Both outputs
  match the upstream reference exactly (0 changed channels, maximum delta 0).
- The migrated CLI generated a 64×64, 1-step PNG with seed 666. PNG dimensions,
  JSON parameters and stdout/stderr separation passed. The C++ smoke executable
  also rebuilt and returned its expected error for a deliberately missing model.

Fatal allocation failures inside STL/spdlog or backend assertions are not
converted into recoverable errors; see the README for this boundary.

## Follow-up: spdlog and Zig project targets

The root CMakeLists/CTest definitions have been removed. Zig now directly builds
the project library, CLI, tests, and reference CLI and installs them. Upstream
CMake builds only `libstable-diffusion` (including static ggml/embedded Metal).
spdlog 1.17.0 is pinned in a submodule and used header-only.

- `zig build test reference --summary all`: 29/29 build steps succeeded. Checks
  cover options, help, invalid arguments, missing models, named spdlog severity,
  literal braces in messages, and stdout/stderr routing.
- `zig build test -Dbackend=cpu --prefix /tmp/pictor-spdlog-cpu --summary all`:
  26/26 steps succeeded. Model-dependent inference remains validated on Metal.
- Installed executable/library inspection shows system libc++ and only relative
  `@loader_path` rpaths, with no dependency on build-cache paths.
- A fresh installation ran from `/tmp`; an Apple Clang C++ consumer linked the
  public API and handled the missing-model exception correctly.
- A real Metal batch generated two 512×768 images, 3 steps, no cache, seeds
  666/667, with one model load. Its first image matches the original baseline's
  RGB pixels exactly (0 changed channels).
- stdout contained exactly two image paths. stderr contained named application
  info logs and backend warning/debug logs under `--verbose`. JSON seeds and
  settings were checked.
- Observed times: load 17.15 s; generation 51.72 s and 46.12 s. These remain
  functional checks, not performance benchmarks.

The following records describe the initial CMake-based implementation before
this build/logging migration.

Host: Apple M4, 32 GiB unified memory, macOS. Zig 0.16.0; CMake Release build
using Apple Clang. sd.cpp and ggml revisions and the model checksum are pinned
in the README. This is integration validation, not a controlled performance benchmark.

## Checks completed

- Metal build and installation succeeded.
- CPU configuration also built successfully and passed the same 3/3 CTest checks;
  model inference was exercised on Metal only.
- `zig build test`: 3/3 CTest checks passed. These cover preset/override parsing,
  invalid numeric values, seed overflow, output naming, help, and missing-model errors.
- `zig build smoke`: one resident model context generated three images. The
  first two used the same prompt, seed 666, 512×768, 3 steps, and no cache. Their
  decoded RGB pixels are identical. A third request changed prompt, seed, and
  cache mode and produced a different image. The first PNG was visually inspected.
- The pinned upstream `sd-cli` produced an image with exactly the same decoded
  RGB pixels as the first smoke image: 0 changed channels, maximum delta 0.
- The installed CLI runs outside the repository, and an independent C++ program
  compiles/links against the installed header/library and handles a missing model.
- The installed CLI generated a two-image Balanced batch from `/tmp`, with an
  output directory containing spaces. It loaded the model once, wrote 512×768
  PNGs and valid JSON sidecars, and recorded seeds 42 and 43 and the expected
  sampler/scheduler/cache settings. Existing output was rejected before loading.
- Download completed and matched the pinned SHA-256. Model file: 1,792,252,512 bytes.

## Observed timings

All generated images below are 512×768. Cache state and system load were not controlled.

| Run | Settings | Load | Generation |
| --- | --- | ---: | ---: |
| Resident session, first request | 3 steps, no cache | 16.48 s | 57.23 s |
| Same session, identical request | 3 steps, no cache | reused | 58.63 s |
| Same session, changed prompt/seed | 3 steps, Spectrum | reused | 48.73 s |
| Upstream reference CLI | 3 steps, no cache | included in wall time | 51.27 s |
| Installed CLI, batch image 1 | 8 steps, Spectrum | 22.83 s | 87.30 s |
| Installed CLI, batch image 2 | 8 steps, Spectrum | reused | 108.37 s |

Upstream reference wall time was 71.95 s; its denoising took 39.53 s and VAE
decoding 11.58 s. `/usr/bin/time -l` reported maximum RSS 3,528,163,328 bytes
and peak physical footprint 3,143,699,696 bytes. Those measurements are for the
reference process, not a complete GPU-memory accounting.

The installed CLI's two-image batch took 219.33 s wall time. Spectrum skipped
2/8 steps for each image. Maximum RSS was 3,529,474,048 bytes and peak physical
footprint 3,144,387,752 bytes. Variation between requests reinforces that these
are functional validation runs rather than controlled latency measurements.

The wrapper preserves outputs and supports resident weights; these measurements
do not establish a denoising speedup. The original project's M2 Max timings do
not describe this M4 run.

## Boundaries

- Exact pixel parity is established only for the tested checkpoint, backend,
  device, build, and request. Different devices or revisions may differ.
- No HTTP, CUDA, Windows, other checkpoints, split weights, or external LoRAs
  were tested. High-resolution memory/performance tuning is deferred.
- The upstream Metal warning about the tensor API being disabled before M5/A19
  is expected on this M4; the normal Metal backend was used successfully.
- The dependency emits existing compiler warnings. No upstream model code was
  modified beyond the original bundled Metal patch.

## Native batch follow-up (2026-09-26)

- Added Anima/Klein C++ batches, C ABI batches and CLI chunking. Bounds: 8 images
  and 16777216 total output pixels. Existing single-image APIs/layouts remain.
- Stand-in tests exercise consecutive/random/overflow seeds, one backend call per
  batch, reference order and borrowed-input lifetime, invalid final images,
  no partial outputs, C handle ownership after session destruction and callback
  indices excluding VAE events. CLI checks cover 9 images split 8+1, metadata,
  filenames and reference batches. ASan/UBSan stand-in run passes. CPU build,
  API tests and Zig/Rust consumers pass all 53/53 build steps; the final Metal
  run also passes 53/53 with the updated batch FFI bindings.
- Real Metal Anima, 256x256, seeds 42/43, same resident session: both Fast
  (3 steps, CFG 1, Spectrum) and CFG 2 + negative prompt + cache disabled match
  individual calls and subsequent repeats exactly in RGB bytes. Text conditioning
  takes about 0.06–0.22 s here; batch reuse removes one repeat but does not establish
  an end-to-end speedup above run variation.

| Anima workload | Two single calls before | Native batch of 2 | Two single calls after |
| --- | ---: | ---: | ---: |
| Fast / CFG 1 / Spectrum | 27.578 s | 26.638 s | 25.268 s |
| CFG 2 / negative prompt / no cache | 35.564 s | 37.160 s | 35.171 s |

These are development observations, not controlled benchmarks (other build work
may run on the host). Model loading is excluded. Reproduce with
`zig build smoke-batch -- --anima` and `zig build smoke-batch -- --klein`.

Klein text-to-image, 256x256, 4 steps, CFG 1, seeds 42/43: exact RGB parity for
the batch and subsequent individual calls. Timings: serial-before 124.430 s,
batch 81.705 s, serial-after 99.612 s. The batch encodes text once (4.96 s),
but sampling itself also varied substantially; do not attribute the entire
wall-time difference to encoding reuse.

Klein double-reference editing uses the two text outputs as ordered references,
with VAE tiling enabled. The native batch matches the individual baseline images
exactly; sampling callbacks report the correct indices despite VAE tile events.
Reference VAE preparation and text conditioning each execute once per native
batch (3.24 s and 6.03 s respectively in this run).

The two subsequent individual edits also match exactly, completing the same-session
reuse check. Two-image times: serial-before 189.717 s, native batch 178.796 s,
serial-after 190.771 s (about 6% less elapsed time in this particular 256² example).
This is one development run, not a general speedup guarantee. Both model smoke
commands completed successfully.

## Experimental Klein hidden-state compression (2026-09-26)

- Default off; additive C++/C session setter, Zig/Rust declaration checks and CLI
  flag. Existing public C ABI v1 layouts/signatures stay unchanged. Applies to
  text generation, reference editing and native batches; Anima rejects opt-in.
- Metal and CPU tests/FFI pass 56/56 build steps each. Scalar-reference tests cover
  pooling, rotary-position selection, residual restoration, odd rectangular grids
  and independent references. A copy of the same graph on Metal matches every
  contiguous F32 node from CPU exactly. This verifies the compression primitives,
  not equivalence of full model inference between backends.
- Real Metal checks: one-step on/off pixel equality; multi-step off/on/off restores
  exact pixels; 80x112 output with 80x64 and 112x80 references, native reference
  sizing and tiled VAE preserves two-seed batch/individual equality. Both inputs
  remain unchanged. Each image resets the compression schedule.
- 512x512, four Euler steps, CFG 1, seed 666, resident Q4 Klein model: first three
  steps reduce image tokens 1024 to 256; last step stays full. Off output matches
  the pre-change `outputs/klein-reference.png` in every RGB byte.

| 512px fox comparison | HS off | HS on |
| --- | ---: | ---: |
| Generation, excluding model loading | 110.280 s | 92.168 s |
| Sampling | 91.02 s | 71.80 s |
| VAE decode | 13.97 s | 14.71 s |

One paired run on Apple M4; about 16.4% less generation time, not a general speedup
guarantee. No concurrent build ran during this pair. Prompt: `A small red fox
sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light`.
Outputs: `outputs/hs-smoke/exact.png` and `outputs/hs-smoke/hs.png`.

**Visual quality is not acceptable in this test:** HS introduces obvious repeated
facial/body contours and striping. RGB MAE 29.7478 and PSNR 15.458 dB measure the
pixel difference only. The original Python algorithm's settings were chosen for
2K images; the ggml port's high-resolution quality is unvalidated. Numerical graph
parity and code inspection found no discrepancy in the pooling/restoration
primitives, but do not establish the cause of all model-level artifacts. This
feature remains explicitly experimental, disabled by default, and unsuitable as
a general quality-preserving preset. `smoke-hs` success is a functional check, not
a visual-quality approval.

Clean pinned-source replay of all three sd.cpp patches reproduces the working
backend byte-for-byte, and each patch's reverse check remains idempotent.


## Native MLX integration

Apple Silicon default Klein now uses native MLX and Small Decoder; ggml remains
selectable. Real `zig build smoke-mlx` validates C++/C pixel parity, batch-vs-single
seeds, reference batches/multiple references, aliased input/output lifetime,
Unicode, CFG, tiled decoding and error recovery. `scripts/check-mlx-parity.py`
compares all text/transformer/original-VAE/Small-VAE tensors and token fixtures
with pinned MFLUX; all elements match. A 512²/four-step original-VAE generation
also matches the earlier aligned MFLUX image exactly. A relocated bin/lib tree
successfully performs text generation and reference editing without Python.

Metal tests/FFI plus benchmark build: 69/69 steps. CPU with MLX disabled: 59/59 steps. Native MLX and
Small Decoder workflows, numerical fixtures and limits are in [mlx.md](mlx.md).
