# pictor

C++ inference library with a C ABI and CLI for **Anima P3 Turbo AIO Q4** and
**FLUX.2-klein-4B**. Apple Silicon defaults to native MLX for Klein and a pinned
stable-diffusion.cpp backend for Anima; Klein also supports explicit ggml selection. Zig directly builds the C++ library, CLI, and tests
and owns installation. Only the upstream sd.cpp/ggml dependency uses CMake/Ninja
and the platform toolchain for its C++/Metal code. No Python runtime or HTTP server.

## Build

On Apple Silicon, install Zig **0.16.0**, CMake, Ninja, Git, and Xcode Command Line
Tools (or Xcode), plus `uv` for the first MLX dependency preparation. Metal is enabled by default on macOS.

```sh
zig build
zig build test
zig build run -- --help
```

The first build initializes missing sd.cpp/ggml submodules and applies the
bundled Metal and progress-output patches. It may access GitHub. Subsequent builds use local sources;
model weights are **not** downloaded by building or testing.

Outputs are `zig-out/bin/pictor`, `zig-out/lib/libpictor.dylib`, and
`zig-out/lib/libstable-diffusion.dylib` on macOS, plus public headers in
`zig-out/include/pictor`. Keep `bin` and the libraries together when
moving an installation. MLX builds also install `libmlx.dylib` and `mlx.metallib`; keep both in `lib`. sd.cpp Metal shaders are embedded.

```sh
zig build -Djobs=8                         # upstream CMake jobs (default: 4)
zig build -j8                             # Zig project build parallelism
zig build -Doptimize=Debug                 # default: ReleaseFast
zig build -Dbackend=cpu                    # separate build, MLX disabled by default
zig build -Dmlx=false                      # Metal sd.cpp-only build
zig build --prefix /absolute/install/path
```

The project has no root `CMakeLists.txt`. `build.zig` defines C++ modules with
`addCSourceFiles`, libraries with `addLibrary`, executables with `addExecutable`,
and tests/install steps using Zig's build API. Project outputs are cached by Zig.
Upstream build files live under `.zig-cache/backend/<backend>-<CMake build type>`;
the small `cmake/BuildNative.cmake` helper only serializes upstream Ninja builds.

On macOS, Zig compiles project C++ against the SDK's C++ headers and system
`libc++`, matching the upstream Apple Clang build and native C++ consumers.
It does not mix Zig's bundled libc++ into the public library ABI.

The initial target is macOS/Apple Silicon. CPU is available for development;
CUDA, Windows packaging, and cross compilation are outside this first version.

## Model

Download the original project's exact checkpoint, or supply an existing file:

```sh
zig build download-model
# Optional custom location:
bash scripts/download-model.sh /absolute/path/Anima-P3-Turbo-AIO-Q4_K.gguf
```

The download is **1,792,252,512 bytes** (1.79 GB / 1.67 GiB), resumes via a `.part`
file, and verifies SHA-256 before renaming. It is independent of the build.

| Item | Pinned value |
| --- | --- |
| Repository | [n-Arno/Anima-P3-Turbo-AIO-Q4_K](https://huggingface.co/n-Arno/Anima-P3-Turbo-AIO-Q4_K) |
| Revision | `ddb1225449828d4e5d69208c008b53f838e626c5` |
| File | `Anima-P3-Turbo-AIO-Q4_K.gguf` |
| SHA-256 | `3290dc9abad9cf98cf1b39b491a464bd4e7bba200ed508dcf0a7e9cd8fdb9168` |

This AIO contains DiT, text encoder, VAE, and merged Turbo LoRA. The presets below
are specific to it. Other Anima revisions, split weights, and arbitrary LoRAs
are not covered by this release. Weights retain their upstream model license;
see the model repository for terms. Weights are not committed to this repository.

## FLUX.2-klein-4B

Klein defaults to native MLX 4-bit diffusion/text weights and the official Small
Decoder on Apple Silicon. Philox noise and Euler/discrete sampling preserve the
validated aligned mode. C++ and C ABI generation, reference editing and batch
calls all support MLX. See [native MLX setup, API and fallback](docs/mlx.md).
Explicit GGUF paths or `--backend ggml` retain the sd.cpp path.

```sh
zig build download-klein-model
zig build run -- flux-klein --prompt "A red fox in a sunlit forest" \
  --seed 666 --output outputs/fox.png
zig build run -- flux-klein --ref-image outputs/fox.png \
  --prompt "Change the scene to snowy winter. Keep the subject and composition." \
  --seed 666 --output outputs/winter-fox.png
```

Defaults are 512×512, 4 steps, CFG 1, Euler/discrete, diffusion flash attention,
and no cache. Custom paths use `--diffusion-model`, `--text-encoder` / `--llm`,
and `--vae`. `--count` reuses loaded weights. C++ uses `FluxKleinSession`;
the additive C ABI creation API works with the existing image and session calls.
See [Klein usage, pinned weights and API examples](docs/klein.md).
Repeat `--ref-image` / `-r` for up to four ordered PNG/JPEG inputs. Editing defaults
to aspect-preserving reference resizing; output dimensions remain explicit.
The same session supports both `generate()` and `edit()`. The additive C edit API
accepts borrowed RGB8 buffers, usable from Zig/Rust without file I/O.
Masked inpainting, 9B/base variants and SDNQ weights are outside this scope.

## Generate

From the project directory:

```sh
zig-out/bin/pictor anima \
  --prompt "masterpiece, best quality, anime landscape, a cottage by a lake" \
  --preset balanced --seed 666 --output outputs/cottage.png
```

Defaults: 512×768, CFG 1, `er_sde`, `smoothstep`, flash attention. `--model` defaults
to `models/Anima-P3-Turbo-AIO-Q4_K.gguf` relative to the current directory.

| Preset | Steps | Cache |
| --- | ---: | --- |
| `fast` | 3 | Spectrum |
| `balanced` (default) | 8 | Spectrum |
| `quality` | 16 | none |

`--steps` and `--cache` override presets regardless of argument order. Spectrum
is an approximation; use `--cache none` for comparisons. CFG 1 normally skips
the negative/unconditional pass. `--vae-tiling` can reduce decoding memory.

Generate multiple images with one model load and shared conditioning per batch:

```sh
zig-out/bin/pictor anima --prompt "anime landscape, sunset" \
  --seed 666 --count 2 --output outputs/sunset.png
```

This writes `sunset-001.png` and `sunset-002.png` with seeds 666 and 667. With a
random seed (`-1`), the CLI resolves one base seed and increments it. Every PNG
has a JSON sidecar recording effective settings, seed, model path, backend
revision, load time, batch time and amortized per-image generation time. Existing outputs require `--overwrite`.
Anima reuses text conditioning (including the negative prompt when CFG requires it).
Klein additionally reuses reference-image encodings for editing. Sampling still runs
serially; this does not cache conditioning across requests or accelerate a single image.
The CLI splits `--count` into batches of at most 8 images and 16 megapixels of total
output. Each batch returns/saves all images after decoding, so the first PNG arrives
later than with individual calls. Batches retain multiple latents and decoded images;
the pixel cap bounds accumulation, not total model/backend memory.
Sidecars identify `generation_seconds_kind: "batch_average"`, `batch_count` and
`batch_generation_seconds`; `generation_seconds` is that batch time divided by its count.
PNG paths go to stdout; status/progress/backend diagnostics go to stderr.

Logging uses pinned **spdlog 1.17.0** (header-only, bundled fmt). Application
messages use the `pictor` logger; sd.cpp/ggml callbacks use the `sd.cpp` logger and
preserve severity. Backend info/debug messages require `--verbose`; warnings and
errors always remain visible. Each step is a separate log line. Color is enabled
only for terminals, and backend trailing newlines are trimmed to avoid blank lines.
The library does not change an embedding application's default spdlog logger.

`--help` lists all flags. Invalid arguments exit with 2; inference/I/O errors with 1.
There is no in-process cancellation API in the pinned backend. A CLI process can
be interrupted normally; the library does not promise cooperative cancellation.

## Library

Both interfaces are installed: `pictor/anima.hpp` and `pictor/flux_klein.hpp` for
C++, and `pictor/pictor.h` for C, Zig, Rust, or other C FFI consumers. Link `libpictor` and keep
`libstable-diffusion` alongside it. Neither public interface exposes sd.cpp types.

### C++ (explicit errors, no exceptions)

```cpp
#include <pictor/anima.hpp>
#include <cstdio>

int main() {
    std::unique_ptr<pictor::AnimaSession> session;
    auto status = pictor::AnimaSession::create({"/absolute/path/model.gguf"}, session);
    if (!status) { std::fprintf(stderr, "%s\n", status.message); return 1; }
    auto request = pictor::preset_request(pictor::Preset::balanced);
    request.prompt = "anime landscape, a cottage by a lake";
    request.seed = 666;
    pictor::Image image;
    status = session->generate(request, image);
    if (!status) { std::fprintf(stderr, "%s\n", status.message); return 1; }
    status = pictor::write_png("cottage.png", image);
    if (!status) { std::fprintf(stderr, "%s\n", status.message); return 1; }
    // Reuse session for further requests; image owns its pixels independently.
}
```

`Status` contains `ErrorCode` and a fixed 512-byte message; `status.ok()` / its
boolean conversion means success. Creation uses a factory because constructors
cannot return errors. This updates the earlier throwing API: callers must migrate
construction to `create`, pass an output `Image` to `generate`, and check statuses.
The C++ session/request/image types remain available. Callbacks are now a
`noexcept` function pointer plus `void* userdata`, instead of `std::function`.

For a batch, both session types expose `generate_batch(request, count, result)`;
Klein also exposes `edit_batch(edit_request, count, result)`. Single-image methods
remain available. `BatchResult` owns its images and whole-batch `generation_seconds`:

```cpp
pictor::BatchResult batch;
auto status = session->generate_batch(request, 2, batch);
if (!status) { std::fprintf(stderr, "%s\n", status.message); return 1; }
// batch.images[0/1] own RGB8 pixels and carry seeds request.seed + 0/1.
// Image::generation_seconds is the batch average, not measured image latency.
```

Batch calls accept 1..8 images and at most 16777216 output pixels in total. Seed
`-1` resolves one random starting seed; sequence overflow is rejected. Outputs are
cleared on failure. `BatchProgressCallback` reports the zero-based image index and
sampling step; it excludes reference encoding and VAE decoding/tiling events.

Public wrappers, CLI, and test consumers compile with `-fno-exceptions`;
spdlog/fmt use their no-exception modes there. The private sd.cpp boundary
(`src/backend.cpp`) and native MLX implementation contain runtime exceptions and
convert them to status codes before returning to the public API. The upstream
backend and optional upstream CLI retain their exception support. Public-path
filesystem I/O, CLI numeric parsing and OS random-seed generation report errors
explicitly.

This is explicit handling of recoverable errors, not a guarantee of recovery from
all allocation failures: retained STL containers can terminate on OOM, spdlog's
fatal failures abort in no-exception mode, and backend assertions remain fatal.
Detected handle allocation failures and caught upstream `bad_alloc` return
`out_of_memory`. Backend cleanup errors are logged; destructors never throw.

### C ABI / Zig / Rust

`pictor/pictor.h` documents ABI v1. The API provides preset initialization,
validation, session creation/generation/destruction, image inspection and PNG output.

- Status codes are fixed-width integers: `PICTOR_OK`, `PICTOR_INVALID_ARGUMENT`,
  `PICTOR_IO_ERROR`, `PICTOR_BACKEND_ERROR`, `PICTOR_OUT_OF_MEMORY`.
- Initialize options/requests with `*_init(..., sizeof(value), ...)`. Check
  `pictor_abi_version()` against `PICTOR_ABI_VERSION` before using the ABI.
  V1 layouts are fixed; size mismatches are rejected rather than over-read/written.
- `pictor_error` is an optional caller-owned buffer, cleared on success and filled
  on error with a NUL-terminated, possibly truncated message. No global last-error
  state, cross-language allocator, or exception handling is required.
- Strings are borrowed UTF-8 C strings for the duration of the call. A NULL
  negative prompt means empty. Callers must pass valid pointers, sizes and handles.
- Initialize output handles to NULL; create/generate require an empty slot and
  leave it NULL on failure. Reusing an occupied slot is rejected without freeing it.
- Images outlive their sessions. `pictor_image_get_info` gives a borrowed view of
  tightly packed RGB8 pixels; free the image with `pictor_image_destroy`, never
  free its pixel pointer yourself. Destroying NULL handles is safe.
- Inference is synchronous. Callbacks run on the generating thread and receive
  caller userdata. They must not re-enter pictor, throw, longjmp, or unwind a Rust
  panic. Backend calls are serialized across sessions; do not destroy a handle
  while another call is using it. Do not mix direct sd.cpp calls with pictor.

The additive ABI v1 batch entry points are `pictor_session_generate_batch` (both
models) and `pictor_flux_klein_session_edit_batch`. Pass an array of `count` NULL
image handles and a `double*` for whole-batch seconds. On failure the slots remain
unchanged and seconds is zero; on success destroy every returned image separately.
The existing request layouts, image functions and single-image calls are unchanged.

```c
pictor_image* images[2] = {NULL, NULL};
double batch_seconds = 0;
pictor_status status = pictor_session_generate_batch(
    session, &request, 2, NULL, NULL, images, &batch_seconds, &error);
if (status != PICTOR_OK) { fprintf(stderr, "%s\n", error.message); return 1; }
/* Inspect or write images[0] and images[1] using existing image functions. */
for (int i = 0; i < 2; ++i) pictor_image_destroy(images[i]);
```

Working consumers are in [`examples/anima.zig`](examples/anima.zig) (`@cImport`)
and [`examples/anima.rs`](examples/anima.rs) (`repr(C)` / `extern "C"`, no crates).
Both include progress userdata, pixel access, error handling, and handle cleanup.

```sh
zig build test-ffi              # builds/runs Zig and Rust consumers; needs rustc
# No arguments exercise linking, layouts, validation and a missing-model error.
zig-out/bin/pictor_zig_example
zig-out/bin/pictor_rust_example
# A model argument runs real inference and writes outputs/zig.png or rust.png:
zig-out/bin/pictor_zig_example models/Anima-P3-Turbo-AIO-Q4_K.gguf
zig-out/bin/pictor_rust_example models/Anima-P3-Turbo-AIO-Q4_K.gguf
```

Standalone C linking on macOS (the test itself needs no weights):

```sh
cc -std=c11 -Izig-out/include tests/c_api_test.c -Lzig-out/lib -lpictor \
  -Wl,-rpath,"$PWD/zig-out/lib" -o /tmp/pictor-c-client
/tmp/pictor-c-client
```

For an external Zig build, add the installed include path and library path, link
`pictor` and libc, and configure a runtime library search path. Rust consumers do
the same with a native library search path and `#[link(name = "pictor")]`; the
example contains the declarations needed for its flow. Copy both dynamic libraries
when packaging either consumer. C++ consumers must use a compatible C++ runtime;
C ABI consumers do not need C++ headers.

For Small Decoder, resident timing, Metal System Trace and MLX comparison experiments,
see [the reproducible benchmark guide](docs/benchmarking.md).

### Optional Klein hidden-state compression

HS is **experimental and off by default**. A 512px test showed obvious repeated
contours/striping, so this is not a quality-preserving speed preset. Enable it for an existing
C++ Klein session with `set_hidden_state_compression(true)`, or call
`pictor_flux_klein_session_set_hidden_state_compression(session, 1, &error)` from
C/Zig/Rust. Disable with `false` / `0`. The setting applies to subsequent text,
reference-edit and batch calls; Anima does not support it. Existing ABI layouts
and default generation paths are unchanged.

CLI: add `--hs-compression` to `pictor flux-klein`. The fixed mode pools 2x2 image
tokens in single-stream blocks for the first N-1 Euler steps, leaving the last
step full. It handles each rectangular reference grid independently and preserves
full-resolution residuals. A one-step request stays exact. See
[the Klein guide](docs/klein.md#optional-hidden-state-compression-hs) for API examples,
session/concurrency semantics and quality limitations.

## Verification

```sh
zig build test                  # no weights: C/C++ API, backend errors, CLI, logging
zig build test-ffi              # no weights: Zig/Rust consumers (requires rustc)
zig build smoke-c               # Anima weights: C callbacks, reuse, ownership, PNG
zig build smoke-klein           # Klein weights: resident C++/C ABI parity
zig build smoke-klein-edit      # Klein weights + outputs/klein-reference.png: edit parity/reuse
zig build smoke-hs              # Klein weights: HS geometry, toggling and visual A/B
zig build smoke-batch -- --anima # Anima native batch parity/timing
zig build smoke-batch -- --klein # Klein text + two-reference batch parity/timing
zig build smoke                 # weights required: repeated in-process generation
zig build smoke -- /path/model.gguf outputs/smoke
zig build reference             # optional upstream sd-cli, without the HTTP server
```

The C ABI smoke also verifies image lifetime after session destruction and recovery
from invalid requests and PNG I/O failures. The C++ smoke test compares raw pixels for two identical requests in one session,
then changes prompt, seed, and cache mode. Images are written to `outputs/smoke`.
The reference executable is `zig-out/bin/sd-cli`. It can run the original command for parity:

```sh
zig-out/bin/sd-cli \
  -m models/Anima-P3-Turbo-AIO-Q4_K.gguf --fa \
  --steps 3 --cfg-scale 1 -W 512 -H 768 \
  --sampling-method er_sde --scheduler smoothstep --seed 666 \
  -p "masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people" \
  -o outputs/reference.png
```

Compare decoded pixels rather than PNG bytes (the upstream CLI embeds metadata).
Same-seed identity is tested for this backend/build/device; it is not a guarantee
across versions or different GPU backends. See `docs/validation.md` for measured results.

## Dependency maintenance

- sd.cpp: `90e87bc846f17059771efb8aaa31e9ef0cab6f78`
- ggml: `404fcb9d7c96989569e68c9e7881ee3465a05c50`
- Local patch: `patches/anima-ggml-metal-im2col3d-pad.patch`
- Callback output patch: `patches/sd-model-progress-callback.patch` (suppresses
  split-file loader separator newlines when a progress callback is installed)
- spdlog 1.17.0: `79524ddd08a4ec981b7fea76afd08ee05f83755d`

Preparation checks all three dependency revisions and whether both patches are already applied.
Mismatches fail explicitly; it does not reset modified checkouts. A modified ggml
and sd.cpp submodule after building is expected because patches are kept separately.
When upgrading, update the pins, check patch compatibility and API ownership,
then rerun inference parity and resident-session tests.

The starting configuration and Metal patch come from
[ultra-fast-image-gen](https://github.com/newideas99/ultra-fast-image-gen).
Upstream dependency license files remain in the submodules; see
`THIRD_PARTY_NOTICES.md` for attribution.
