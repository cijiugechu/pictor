# pictor

Local image generation on Apple Silicon — text-to-image and reference editing
for **Anima P3 Turbo AIO Q4** and **FLUX.2-klein-4B**, exposed as a C++17
library, a stable C ABI (v1) for Zig/Rust/other FFI consumers, and a CLI.
Both models default to a native MLX backend on Apple Silicon; the pinned
stable-diffusion.cpp/ggml backend remains explicitly selectable.
No Python runtime and no HTTP server.

## Quickstart

Requirements: Zig **0.16.0**, CMake, Ninja, Git, Xcode Command Line Tools,
and `uv` (first MLX dependency preparation only).

```sh
zig build
zig build download-model            # Anima weights
zig build download-klein-model      # Klein weights

zig-out/bin/pictor anima -p "anime landscape, a cottage by a lake" \
  --seed 666 -o outputs/cottage.png
zig-out/bin/pictor flux-klein -p "A red fox in a sunlit forest" \
  --seed 666 -o outputs/fox.png
zig-out/bin/pictor flux-klein -r outputs/fox.png \
  -p "Change the scene to snowy winter. Keep the subject and composition." \
  --seed 666 -o outputs/winter-fox.png
```

PNG paths go to stdout; status/progress/backend diagnostics go to stderr.
Every PNG gets a JSON sidecar recording effective settings, seed, model
paths and timings. Invalid arguments exit with 2; inference/I/O errors with 1.
Existing outputs require `--overwrite`.

## Models

### Anima P3 Turbo

`zig build download-model` fetches the pinned GGUF (1.79 GB) and, on MLX
builds, stream-converts it to **5.14 GiB BF16** under
`models/anima-p3-mlx-bf16`, verifying and reusing a complete conversion.
CPU/no-MLX builds prepare only the GGUF. Preparation is explicit and
Python-assisted; builds and inference never download or convert weights.

| Item | Pinned value |
| --- | --- |
| Repository | [n-Arno/Anima-P3-Turbo-AIO-Q4_K](https://huggingface.co/n-Arno/Anima-P3-Turbo-AIO-Q4_K) |
| Revision | `ddb1225449828d4e5d69208c008b53f838e626c5` |
| File | `Anima-P3-Turbo-AIO-Q4_K.gguf` |
| SHA-256 | `3290dc9abad9cf98cf1b39b491a464bd4e7bba200ed508dcf0a7e9cd8fdb9168` |

The AIO contains DiT, text encoder, VAE and merged Turbo LoRA. The pinned
public `xocialize/anima-mlx` checkpoint is an optional slower detail/style
alternative. Arbitrary Anima revisions and LoRAs are not covered.
See [Anima MLX setup and API](docs/anima-mlx.md).

### FLUX.2-klein-4B

`zig build download-klein-model` fetches the pinned MLX 4-bit checkpoint on
MLX builds (GGUF split weights on ggml builds), plus the official Small
Decoder either way.
Klein sessions load the full VAE so `generate()` and `edit()` interleave
without reloading. Base/9B variants, masked inpainting, LoRAs and SDNQ
weights are unsupported. Pinned revisions and checksums:
[Klein usage and API](docs/klein.md), [native MLX backend](docs/mlx.md).

## CLI usage

```sh
zig-out/bin/pictor anima  --preset balanced --steps N --cache none|spectrum
zig-out/bin/pictor flux-klein --ref-image a.png -r b.jpg   # 1-4 ordered refs
```

- Backends: `--backend auto|mlx|ggml`. Auto picks MLX for weight directories,
  ggml for files; MLX-enabled Apple Silicon builds default both models to MLX.
- Anima defaults: 512×768, CFG 1, `er_sde`/`smoothstep`, flash attention.
  Presets `fast`/`balanced`/`quality` = 3/8/16 steps; ggml fast/balanced use
  Spectrum cache, MLX always runs uncached (explicit Spectrum is rejected).
- Klein defaults: 512×512, 4 steps, CFG 1, Euler/discrete, diffusion flash
  attention, no cache. `--preset`/`--model` are Anima-only; Klein takes
  `--diffusion-model`, `--text-encoder`/`--llm`, `--vae`.
- `--count N` (1..64) generates consecutive seeds in bounded native batches
  (≤8 images and ≤16 MP each), reusing text and reference encodings per
  batch. Files are named `name-001.png`...; each image's recorded time is the
  batch average.
- Other flags: `--negative-prompt`, `--cfg-scale`, `--seed` (-1 random),
  `--width`/`-W`, `--height`/`-H`, `--vae-tiling` (approximate tiled decode,
  off by default), `--threads` (ggml only), `--disable-auto-resize-ref-image`,
  `--hs-compression` (see below), `--verbose`. `pictor --help` lists all.

## Library

`pictor/anima.hpp`, `pictor/flux_klein.hpp` (C++) and `pictor/pictor.h`
(C ABI v1) are installed under `zig-out/include/pictor`. Link `libpictor`
and keep `libstable-diffusion` (plus `libmlx`/`mlx.metallib` on MLX builds)
alongside it. No public interface exposes sd.cpp types.

```cpp
std::unique_ptr<pictor::AnimaSession> session;
auto status = pictor::AnimaSession::create({"models/anima-p3-mlx-bf16"}, session);
if (!status) { std::fprintf(stderr, "%s\n", status.message); return 1; }
auto request = pictor::anima_request(pictor::Preset::balanced);
request.prompt = "anime landscape, a cottage by a lake";
request.seed = 666;
pictor::Image image;
status = session->generate(request, image);
status = pictor::write_png("cottage.png", image);
```

All public calls are `noexcept` and return `Status` (`ErrorCode` + fixed
512-byte message). Sessions are created via `create(options, output)` or
`create(options, backend, output)`; directories select MLX, files ggml.
`generate_batch`/`edit_batch` take ≤8 images within the pixel cap; images own
their pixels independently of the session. Klein adds `edit`,
`FluxKleinEditRequest` with 1..4 borrowed RGB8 reference views, and
`set_hidden_state_compression`.

The C API mirrors this with `*_init` struct sizing, explicit error buffers
and opaque handles; contracts (ownership, callbacks, serialization, batch
rules) are documented inline in [`pictor/pictor.h`](include/pictor/pictor.h).
Working consumers: [`examples/anima.zig`](examples/anima.zig),
[`examples/anima.rs`](examples/anima.rs) — `zig build test-ffi` builds and
runs both without weights (needs `rustc`), and both accept Anima or Klein
(`--klein`, `--mlx`) model arguments for real inference.

```sh
# Standalone C linking on macOS (no weights needed for the API test):
cc -std=c11 -Izig-out/include tests/c_api_test.c -Lzig-out/lib -lpictor \
  -Wl,-rpath,"$PWD/zig-out/lib" -o /tmp/pictor-c-client
```

### Optional Klein hidden-state compression

Experimental, ggml-only, **off by default** — known to produce visible
contour/stripe artifacts at 512px. Enable per session with
`set_hidden_state_compression(true)` /
`pictor_flux_klein_session_set_hidden_state_compression(session, 1, &error)`,
or pass `--hs-compression` to `pictor flux-klein`. See
[the Klein guide](docs/klein.md#optional-hidden-state-compression-hs).

## Verification

```sh
zig build test                  # no weights: C/C++ API, backend errors, CLI, logging
zig build test-ffi              # no weights: Zig/Rust consumers (requires rustc)
zig build smoke-c               # Anima weights: C callbacks, reuse, ownership, PNG
zig build smoke-anima-mlx       # Native Anima public API/batch/CFG/tiling
zig build smoke-mlx             # Klein MLX weights: C++/C parity, batches, edits
zig build smoke-klein           # Klein ggml weights: resident C++/C ABI parity
zig build smoke-klein-edit      # edit parity/reuse (+ reference PNG)
zig build smoke-hs              # Klein ggml weights: HS geometry and A/B
zig build smoke-batch -- --anima # Anima native batch parity/timing
zig build smoke-batch -- --klein # Klein text + two-reference batch parity
zig build smoke                 # weights required: repeated in-process generation
zig build reference             # optional upstream sd-cli, without the HTTP server
```

Same-seed parity is checked against decoded RGB pixels for the tested
backend/build/device; it is not a guarantee across versions or GPUs.
For resident timing, Metal System Trace and MLX comparison workflows, see
[the reproducible benchmark guide](docs/benchmarking.md).

## Internals

- `build.zig` builds the C++ library, CLI, tests and installs them; only the
  upstream sd.cpp/ggml dependency uses CMake/Ninja (into
  `.zig-cache/backend/<backend>-<type>`, serialized by `cmake/BuildNative.cmake`).
- Options: `-Djobs=N` (upstream parallelism, default 4), `-Doptimize=`
  (default ReleaseFast), `-Dbackend=cpu` (MLX off), `-Dmlx=false`
  (sd.cpp-only), `--prefix` (install path), `-j` (Zig parallelism).
- macOS builds compile project C++ against the SDK's libc++ to match Apple
  Clang consumers; Zig's bundled libc++ is not mixed into the public ABI.
  MLX sources are private C++20 with exceptions; public wrappers are
  C++17/`-fno-exceptions` and translate failures into `Status`.
- Logging uses bundled spdlog 1.17.0 (header-only, with fmt). Backend
  info/debug requires `--verbose`; warnings/errors are always shown.
- Inference is synchronous; backend calls are serialized process-wide.
  Callbacks run on the generating thread and must not re-enter pictor.
- First build initializes the sd.cpp/ggml submodules and applies the bundled
  patches (may access GitHub). Dependency pins:

  | Dependency | Pin |
  | --- | --- |
  | sd.cpp | `90e87bc846f17059771efb8aaa31e9ef0cab6f78` |
  | ggml | `404fcb9d7c96989569e68c9e7881ee3465a05c50` |
  | spdlog | 1.17.0 (`79524ddd08a4ec981b7fea76afd08ee05f83755d`) |

  Patches: `patches/anima-ggml-metal-im2col3d-pad.patch` (ggml),
  `patches/sd-model-progress-callback.patch`,
  `patches/sd-sampling-image-index.patch`,
  `patches/sd-flux-hidden-state-compression.patch` (sd.cpp). Preparation
  verifies all three revisions and all four patches; mismatches fail
  explicitly without resetting modified checkouts. See
  `THIRD_PARTY_NOTICES.md` for attribution.

The initial target is macOS/Apple Silicon; CPU builds are for development.
CUDA, Windows packaging and cross compilation are out of scope.
