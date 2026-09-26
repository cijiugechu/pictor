# Native MLX Klein backend

Apple Silicon Metal builds now default the Klein CLI to **native MLX + Small
Decoder**. Anima also defaults to native MLX; see [Anima's setup/API guide](anima-mlx.md).
Klein's C++ session and C ABI expose the same
text, reference-edit and batch operations with either backend. Inference never
starts Python: libpictor links libmlx directly.

## Build and install

```sh
zig build                       # MLX enabled on Apple Silicon with Metal
zig build -Dmlx=false            # sd.cpp-only build
zig build -Dbackend=cpu          # defaults to MLX disabled
zig build download-klein-model  # default backend checkpoint + Small Decoder
```

The first MLX build needs `uv` and network access to prepare pinned MLX/MLX-metal
**0.32.2** development files (Python 3.12.11 is used for package installation).
An existing matching benchmark environment can supply these files. Subsequent
builds reuse `build/mlx-native`. MFLUX is not a build or runtime dependency.

Deploy the whole `zig-out/bin` and `zig-out/lib` layout: `libpictor.dylib`,
`libstable-diffusion.dylib`, `libmlx.dylib`, and **mlx.metallib**. MLX loads the
shader library beside its dylib. No venv, Python executable, build directory or
absolute build rpath is needed at runtime. Include the installed license notices.
MLX uses C++20 and exceptions privately; public C++ wrappers remain C++17 with
exceptions disabled. The native boundary translates failures into `Status`/C
error codes.

## Weights and defaults

The supported MLX checkpoint is `mlx-community/flux2-klein-4b-4bit`, revision
`860e87183ceb29e39627c0612ebd66d8ea66e68c`: affine 4-bit, group size 64, BF16
activations. `benchmarks/mlx-model.json` pins each file's SHA-256. It is also used
by `scripts/download-mlx-model.sh`; GGUF files cannot be loaded by MLX.

Default paths:

- Diffusion: `models/mlx-flux2-klein-4b-4bit/transformer`
- Text: `models/mlx-flux2-klein-4b-4bit/text_encoder`
- VAE: `models/flux2-klein-4b/full_encoder_small_decoder.safetensors`

Small Decoder is the official full-encoder checkpoint, 249,519,092 bytes,
revision `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a`, SHA-256
`ea4273f02d1fafbf8e1d1c2cf6018ed8748652eb0bf34f2dd91171f16f15ab62`.
It is the CLI default for **both** Klein backends. `--vae` always overrides it.
The downloader also retains the original MLX VAE for numerical comparisons.

```sh
zig-out/bin/pictor flux-klein -p "A red fox in a sunlit forest" --seed 666 -o outputs/fox.png
zig-out/bin/pictor flux-klein -r outputs/fox.png -p "Turn the scene into winter" --seed 666 -o outputs/edit.png

# Explicit ggml fallback; download its separate weights first.
bash scripts/download-klein-model.sh
zig-out/bin/pictor flux-klein --backend ggml -p "A red fox" --seed 666 -o outputs/ggml.png

# Original MLX VAE, for comparison:
zig-out/bin/pictor flux-klein --backend mlx --vae models/mlx-flux2-klein-4b-4bit/vae \
  -p "A red fox" --seed 666 -o outputs/full.png
```

The CLI accepts `--backend auto|mlx|ggml`. Auto selects MLX by default in enabled
Apple Silicon builds; explicit diffusion/text directories select MLX and file
paths select ggml. Existing explicit GGUF callers continue using ggml. It never
silently converts weights or retries failed MLX inference on ggml.

## C++ and C ABI

The existing C++ options layout and `create(options, output)` remain intact.
An additive `create(options, backend, output)` overload selects a backend. Automatic
library selection uses the diffusion path: a directory selects MLX, a file
selects ggml. The library requires callers to provide all three paths.

```cpp
pictor::FluxKleinOptions options{
    "models/mlx-flux2-klein-4b-4bit/transformer",
    "models/mlx-flux2-klein-4b-4bit/text_encoder",
    "models/flux2-klein-4b/full_encoder_small_decoder.safetensors"};
std::unique_ptr<pictor::FluxKleinSession> session;
auto status = pictor::FluxKleinSession::create(options, pictor::KleinBackend::mlx, session);
if (!status) { /* handle status.code and status.message */ }
// session->generate / edit / generate_batch / edit_batch use this backend.
```

C ABI v1 structure layouts and existing functions are unchanged. New additive
functions select/query a backend:

```c
pictor_status status = pictor_flux_klein_session_create_with_backend(
    &options, PICTOR_KLEIN_BACKEND_MLX, &session, &error);
pictor_klein_backend backend;
if (status == PICTOR_OK)
    status = pictor_flux_klein_session_backend(session, &backend, &error);
```

The existing `pictor_flux_klein_session_create()` uses AUTO. The Zig and Rust
examples now support `--mlx models/mlx-flux2-klein-4b-4bit [reference.png]`.
Public errors, image ownership, progress events, batch seed increments and shared
Anima/Klein serialization are unchanged. Callbacks must not re-enter pictor.

## Sampling and supported scope

The default always uses pictor's **Philox noise + Euler/discrete FLUX.2 schedule**,
including the resolution-dependent flow shift. Noise starts as FP32 NCHW and is
cast to BF16 before packing. Euler updates use the verified MFLUX fused BF16
policy. This aligns noise and sampling with the earlier comparison; it does not
make MLX affine quantization identical to ggml's Q4 variants.

Qwen3 conditioning uses layers 9/18/27 and 512 tokens. Text and reference
encodings are reused once per batch. References retain pictor's aspect-ratio,
nearest-neighbour resize and ordered conditioning semantics. Both original and
Small VAE retain the full encoder. CFG above 1 supports a negative prompt.

`--vae-tiling` explicitly enables overlapping decoder tiles. It is approximate:
per-tile GroupNorm can alter colour and detail. It is never enabled automatically
and does not tile reference encoding. `--threads` controls ggml CPU threads;
MLX manages its own scheduling. HS compression remains default-off and is
supported only by ggml; enabling it on MLX returns a clear error. No implicit
cross-step cache, LoRA, base/9B checkpoint, or mixed quantization support is added.

## Reproducible checks

```sh
zig build test test-ffi
zig build smoke-mlx             # real weights: C++/C, batches, references, errors
bash scripts/setup-mlx-benchmark.sh  # optional Python reference environment
build/mlx-venv/bin/python scripts/mlx-reference-fixture.py
zig build mlx-probe-build
zig-out/bin/pictor_mlx_probe models/mlx-flux2-klein-4b-4bit outputs/mlx-port
build/mlx-venv/bin/python scripts/check-mlx-parity.py
```

The 64px stage fixture compares token IDs, all text embeddings, transformer
predictions, and original/Small VAE encode/decode tensors. Full 512px/4-step generation with the
original MLX VAE is also pixel-identical to the previously validated aligned
MFLUX image. See the native integration section in the dated benchmark results
for measured runs and their limits.
