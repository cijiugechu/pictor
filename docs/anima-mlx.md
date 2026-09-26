# Native Anima MLX backend

Apple Silicon builds with MLX enabled use the validated **P3 BF16 MLX** route by
default. Anima's public C++ sessions, C ABI and CLI support both MLX and ggml.
Inference runs entirely in native C++; Python is used only for model preparation
and independent numerical checks.

## Setup and selection

```sh
zig build
zig build download-model
zig-out/bin/pictor anima -p 'anime landscape, a cottage by a lake' --seed 666 -o outputs/cottage.png

# Retain the original GGUF/Spectrum route:
zig-out/bin/pictor anima --backend ggml -p 'anime landscape' -o outputs/ggml.png
```

`download-model` verifies the pinned 1.79 GB GGUF and streams its conversion using
the pinned ggml dequantizer to `models/anima-p3-mlx-bf16` (5.14 GiB). The source is
retained for ggml. A complete existing conversion is verified and reused; a bad
hash or incomplete conversion is rejected without overwriting it. Preparation
requires `uv`/NumPy if an existing compatible conversion environment is unavailable.
Ordinary builds and inference do not download or convert weights implicitly.

`-Dmlx=false` and the default `-Dbackend=cpu` build keep ggml defaults. Explicit MLX
selection in a build without MLX returns an error. There is no silent runtime
fallback after an MLX loading/generation failure.

| Selection | Backend / default weights |
| --- | --- |
| No CLI overrides, MLX build | MLX / `models/anima-p3-mlx-bf16` |
| `--backend ggml` | ggml / `models/Anima-P3-Turbo-AIO-Q4_K.gguf` |
| `--model path/to/file.gguf` | automatic → ggml |
| `--model path/to/directory` | automatic → MLX |

MLX directories contain `text_encoder-bf16.safetensors`,
`llm_adapter-bf16.safetensors`, `vae-bf16.safetensors` and a transformer. The loader
prefers `transformer-int4.safetensors` when present, otherwise
`transformer-bf16.safetensors`. Computation uses BF16 activations and FP32 sampler/
latent arithmetic, preserving the measured P3 route. Reference-compute experiments
remain available through the separate benchmark executable.

The public checkpoint remains optional, with its own recipe:

```sh
python3 scripts/download-anima-mlx.py
zig-out/bin/pictor anima --backend mlx --model models/xocialize-anima-mlx \
  -p 'anime landscape, a cottage by a lake' --steps 16 --cfg-scale 4.5 \
  --negative-prompt 'worst quality, low quality, blurry, deformed' \
  --seed 666 -o outputs/public-anima.png
```

P3 presets remain 3/8/16 steps, CFG1, ER-SDE/SmoothStep and Philox. MLX presets use
no cache. ggml retains Spectrum for Fast/Balanced. Explicit Spectrum on MLX fails
before inference; it is never silently ignored. CFG values other than 1 use
negative conditioning, including CFG0 and fractional CFG. The standalone Python
upstream's RNG/schedule differ; these native routes retain pictor's sampler.

## C++ and C API

The existing `SessionOptions`, `GenerationRequest`, opaque handles and ABI v1
struct layouts remain unchanged. Existing GGUF callers retain ggml behavior.
Automatic API selection uses the supplied model path; API callers still supply
that path explicitly. The CLI supplies the platform's default directory/file.

```cpp
std::unique_ptr<pictor::AnimaSession> session;
auto status = pictor::AnimaSession::create(
    {"models/anima-p3-mlx-bf16"}, pictor::AnimaBackend::mlx, session);
if (!status) { /* handle status.message */ return; }
auto request = pictor::anima_request(pictor::Preset::balanced);
request.prompt = "anime landscape";
request.seed = 666;
pictor::BatchResult result;
status = session->generate_batch(request, 2, result);
// Check status before using result.images; query session->backend() if needed.
```

The original two-argument `create(options, output)` remains available and uses
automatic directory/file selection. The new `anima_request()` initializes cache
to none; legacy `preset_request()` intentionally retains its Spectrum behavior.
Use the new initializer or explicitly set `request.cache = CacheMode::none` when
migrating legacy requests to MLX.

```c
pictor_error error = {{0}};
pictor_session_options options;
pictor_request request;
pictor_session* session = NULL;
pictor_image* image = NULL;
pictor_status status = pictor_session_options_init(&options, sizeof(options), &error);
if (status != PICTOR_OK) return;
options.model_path = "models/anima-p3-mlx-bf16";
status = pictor_anima_request_init(&request, sizeof(request), PICTOR_PRESET_BALANCED, &error);
if (status != PICTOR_OK) return;
request.prompt = "anime landscape";
request.seed = 666;
status = pictor_anima_session_create_with_backend(&options, PICTOR_ANIMA_BACKEND_MLX, &session, &error);
if (status != PICTOR_OK) return;
status = pictor_session_generate(session, &request, NULL, NULL, &image, &error);
/* Check status, then inspect/write the owned image. */
pictor_image_destroy(image);
pictor_session_destroy(session);
```

`pictor_session_create()` still performs automatic Anima selection.
`pictor_anima_session_backend()` reports the selected backend and rejects Klein
handles. `pictor_anima_request_init()` gives uncached P3 presets; existing
`pictor_request_init()` is unchanged. Both Zig and Rust examples import/use the
additive APIs and accept either Anima GGUF files or MLX directories.

## Batches, ownership and tiling

Anima reuses positive/negative text and adapter context within a batch, samples
consecutive Philox seeds serially, then decodes the results. Seed -1 resolves one
random starting seed. Existing limits (8 images / 16 megapixels) and overflow
checks apply. Each image reports amortized batch time. Progress covers sampling
only, with image indices and steps 0..N. Returned RGB8 pixels are owned by the
image and survive session destruction; failures return no partial result.

Model loading, generation and destruction use the same process-wide lock as
Klein and ggml. Concurrent calls serialize. Callbacks must not re-enter pictor,
throw, or unwind across the API. Native runtime exceptions become Status/C
errors; malformed UTF-8 is rejected without changing valid prompt normalization.

`--vae-tiling` / `request.vae_tiling` enables approximate overlapping 32×32 latent
tiles (256 pixels, 64-pixel overlap). Small images use the exact full decoder.
Larger tiled outputs differ because attention/convolution context is local to
each tile. Tiling is off by default and only bounds decoder intermediates; it
does not reduce transformer memory or promise artifact-free large images.

## Verification

```sh
zig build test test-ffi
zig build smoke-anima-mlx    # Needs both prepared P3 and public Anima weights
zig build smoke-mlx          # Klein regression, requires its existing weights
zig build test test-ffi -Dbackend=cpu -Dmlx=false --prefix build/cpu-validation
```

Real Anima checks cover C/C++ exact RGB, batch/single and concurrent-call
equivalence, seed progression, callbacks, CFG0/CFG2, negative/Unicode prompts,
tiled edge sizes, malformed weights/input, recovery and image lifetime. The
production 512×768 default and explicit ggml route are also compared with retained
pre-integration images. Integration smoke times are not new controlled benchmarks.

Deployment uses the existing native MLX library/metallib layout described in
[the MLX packaging guide](mlx.md#build-and-install). Inference has no Python
runtime dependency.
