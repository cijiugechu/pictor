# FLUX.2-klein-4B generation and reference editing

Pictor supports the **distilled 4B checkpoint**. Apple Silicon defaults to native
MLX with aligned Philox/Euler/discrete sampling and the official Small Decoder.
The C++ and C ABI remain exception-free at the public boundary. Both backends
support text, reference editing and batches. See [MLX setup and API selection](mlx.md).
Explicit GGUF paths or `--backend ggml` select the retained sd.cpp backend.
Base/9B variants, masked inpainting, LoRAs and SDNQ/Quanto weights are unsupported.

## Weights

`zig build download-klein-model` downloads the configured default backend. On
Apple Silicon that is the pinned MLX manifest plus Small Decoder described in
[mlx.md](mlx.md). The following table records the retained **ggml comparison
weights**, including the original VAE. Its downloader also fetches Small Decoder.


```sh
bash scripts/download-klein-model.sh
# Optional ggml directory:
bash scripts/download-klein-model.sh /absolute/path/klein
```

The three comparison files total **5,293,871,164 bytes** (~5.3 GB / 4.93 GiB). Downloads resume
from `.part` files and verify SHA-256 before renaming; existing files are verified
and are never silently replaced. Builds and ordinary tests do not download weights.
The sizes below describe disk files, not peak inference memory.

| Component | Repository / pinned revision | File | Bytes |
| --- | --- | --- | ---: |
| Diffusion | [leejet/FLUX.2-klein-4B-GGUF](https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF), `3b1f5a9dc3abb32238b053aeb3d823c30afdacbd` | `flux-2-klein-4b-Q4_0.gguf` | 2,460,378,560 |
| Text encoder | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF), `22c9fc8a8c7700b76a1789366280a6a5a1ad1120` | `Qwen3-4B-Q4_K_M.gguf` | 2,497,281,312 |
| VAE | [Comfy-Org/vae-text-encorder-for-flux-klein-4b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b), `5f526678002e43af5551dadb73ce2e8c91b43afe` | `split_files/vae/flux2-vae.safetensors` | 336,211,292 |

SHA-256, in table order:

```text
d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4
f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a
868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3
```

Weights retain their source licenses. They are not bundled in source releases.
The supplied manifest targets Klein 4B distilled; the loader does not authenticate
checkpoint identity for arbitrary user-supplied paths. Other combinations are unvalidated.

## CLI

```sh
zig build
zig-out/bin/pictor flux-klein --prompt "A red fox in a sunlit forest" \
  --seed 666 --output outputs/fox.png

zig-out/bin/pictor flux-klein --prompt "A red fox in a sunlit forest" \
  --diffusion-model /path/flux-2-klein-4b-Q4_0.gguf \
  --text-encoder /path/Qwen3-4B-Q4_K_M.gguf \
  --vae /path/flux2-vae.safetensors \
  --width 1024 --height 1024 --count 2 --seed 666 --output outputs/fox.png
```

Defaults: 512×512, 4 steps, CFG 1, Euler, discrete scheduler (including sd.cpp's
FLUX.2 flow shift), fast attention, no cache. ggml enables diffusion flash
attention but not text-encoder flash attention. MLX uses its fast SDPA kernels.
`--llm` aliases `--text-encoder`; backend-specific default paths are in [mlx.md](mlx.md).
`--steps`, `--cfg-scale`, `--threads`, `--vae-tiling`, `--verbose`, `--count`,
`--overwrite` and output naming follow the existing CLI. `--preset` and `--model`
are Anima-only. Klein currently rejects `--cache spectrum`.

A batch loads the three components once and increments the seed. PNG/JSON files
use the existing output policy. JSON records the model family, all three absolute
paths, effective sampler/scheduler/cache/attention settings, seed and timings.
CFG 1 normally omits the unconditional pass; a negative prompt does not add an
extra guidance pass at that default. Increasing steps is not a guaranteed quality
improvement for this distilled checkpoint.

## Reference editing

### CLI

```sh
zig-out/bin/pictor flux-klein \
  --ref-image outputs/fox.png \
  --prompt "Change the fox's fur to white and the forest to a snowy winter scene. Keep the fox's pose and composition." \
  --width 512 --height 512 --seed 666 --output outputs/winter-fox.png

# Multiple references are ordered; describe their roles in the prompt.
zig-out/bin/pictor flux-klein -r subject.png -r background.jpg \
  --prompt "Place the subject from image 1 into the scene from image 2." \
  --seed 42 --output outputs/combined.png
```

`--ref-image` / `-r` selects reference editing and may be repeated **1–4 times**.
This is the wrapper's current input limit. Files are decoded and validated before
loading the model, and loaded once per batch. The output size remains 512×512 by
default, independently of input dimensions; set `--width` / `--height` explicitly.
The default sampler, four steps and CFG 1 remain unchanged.

By default pictor preserves each reference's aspect ratio while resizing to
approximately `min(output_width * output_height, 1024 * 1024)` pixels, rounding
each dimension to a multiple of 16. `--disable-auto-resize-ref-image` keeps native
dimensions, which must then be multiples of 16. Inputs must be 1–4096 pixels per
dimension; aspect ratios yielding a resized dimension outside 16–4096 are rejected.
All references are conditioned together in the listed order. Their number and
resolution increase inference time and memory use.

PNG/JPEG input becomes tightly packed RGB8. Alpha is discarded, EXIF orientation
is not applied, and no color-profile conversion is performed; prepare inputs
accordingly. The edit is guided by the prompt and references, with no mask or
denoising-strength parameter. It does not guarantee exact preservation of regions.
JSON records `mode`, ordered absolute `reference_images` and `auto_resize_reference`.

## C++

### Generation

```cpp
#include <pictor/flux_klein.hpp>

pictor::Status generate_fox() {
    std::unique_ptr<pictor::FluxKleinSession> session;
    auto status = pictor::FluxKleinSession::create({
        "/path/flux-2-klein-4b-Q4_0.gguf",
        "/path/Qwen3-4B-Q4_K_M.gguf",
        "/path/full_encoder_small_decoder.safetensors",
    }, session);
    if (!status) return status;
    auto request = pictor::flux_klein_request();
    request.prompt = "A red fox in a sunlit forest";
    request.seed = 666;
    pictor::Image image;
    status = session->generate(request, image);
    if (!status) return status;
    return pictor::write_png("fox.png", image);
}
```

`GenerationRequest`, `Image`, `ProgressCallback`, validation helpers and PNG output
live in `pictor/types.hpp`, still included transitively by `anima.hpp`. Existing
Anima source and ABI entry points retain their behavior. Use `flux_klein_request()`
instead of the Anima defaults; `validate_flux_klein_request()` also checks Klein's
cache restriction. Generation clears the output image on error.

Anima, ggml Klein and MLX Klein share **one process-wide backend mutex**. Concurrent calls serialize, and global sd.cpp callbacks remain
scoped to the active call. Callbacks must not re-enter pictor. Returned images own
their storage independently of the session. No model is reloaded between requests.

### Editing

```cpp
pictor::Status edit_fox(pictor::FluxKleinSession& session, pictor::Image& output) {
    pictor::Image source;
    auto status = pictor::read_image("fox.png", source);
    if (!status) return status;
    pictor::FluxKleinEditRequest request;
    request.generation.prompt = "Change the scene to snowy winter. Keep the subject and composition.";
    request.generation.seed = 666;
    request.reference_images.push_back({source.width, source.height,
                                       source.pixels.data(), source.pixels.size()});
    return session.edit(request, output);
}
```

`ImageView` is a borrowed RGB8 view with an exact `width * height * 3` byte length.
You can pass application-owned memory instead of loading a file. Keep its bytes
valid and unchanged until `edit()` returns. The backend reads but does not modify
them. `validate_flux_klein_edit_request()` checks the request without loading a
model. `auto_resize` defaults to true. The output owns its pixels and is cleared
on failure, following `generate()` semantics.

Klein sessions now load the **full VAE (encoder + decoder)** so `generate()` and
`edit()` can be interleaved without reloading. This increases resident VAE memory
compared with the original text-only implementation. Anima remains decode-only.

## C ABI, Zig and Rust

The C ABI remains version 1, with unchanged existing layouts. New symbols are:

- `pictor_flux_klein_options_init`
- `pictor_flux_klein_request_init`
- `pictor_flux_klein_session_create`
- `pictor_flux_klein_edit_options_init`
- `pictor_flux_klein_session_edit`
- `pictor_image_load`

The new `pictor_flux_klein_options` contains three borrowed UTF-8 paths, threads
and verbose, plus the usual `struct_size`. Creation returns the existing opaque
`pictor_session*`; use `pictor_session_generate`, image access, progress callbacks
and destroy functions as before. Input strings need only live through the call.
The shared request validator checks common fields; generation also validates
model-specific restrictions. Existing `pictor_session_create` and
`pictor_request_init` continue to select Anima.

For editing, initialize a normal Klein `pictor_request` and a separate
`pictor_flux_klein_edit_options`. Supply an array of `pictor_image_view`, each with
`struct_size = sizeof(pictor_image_view)`, dimensions, RGB bytes and exact length.
Set the array pointer/count on the options and call `pictor_flux_klein_session_edit`
with the same callback/userdata/output/error arguments used by generation.
Anima sessions are rejected by this call. Existing structs have not gained fields.

`pictor_image_load(path, &image, &error)` creates an owned image without a model;
use `pictor_image_get_info` to construct a borrowed reference view. Keep the loaded
image alive through the edit, then destroy it normally. The generated image is
independent of both the source and the session. All new functions preserve the
existing NULL-output-slot and explicit-error contracts.

The existing language examples now accept `--klein`:

```sh
zig build test-ffi
zig-out/bin/pictor_zig_example --klein   # no weights: ABI/error checks
zig-out/bin/pictor_rust_example --klein
zig-out/bin/pictor_zig_example --klein models/flux2-klein-4b
zig-out/bin/pictor_rust_example --klein models/flux2-klein-4b
zig-out/bin/pictor_zig_example --klein models/flux2-klein-4b fox.png
zig-out/bin/pictor_rust_example --klein models/flux2-klein-4b fox.png
```

With a directory argument they generate `outputs/zig-klein.png` or
`outputs/rust-klein.png`. Without `--klein`, previous Anima invocation forms work.
An optional reference path selects editing and writes `outputs/zig-klein-edit.png`
or `outputs/rust-klein-edit.png`, using a winter-scene prompt.

## Verification

```sh
zig build test test-ffi
zig build smoke-klein
zig build smoke-klein -- /path/model-directory outputs/klein-smoke
zig build smoke-klein-edit -- models/flux2-klein-4b outputs/fox.png outputs/edit-smoke
```

The model-dependent test compares two identical C++ generations in one resident
session, then compares a fresh C ABI session's pixels against them. It exercises
callbacks, invalid-cache recovery and image lifetime after session destruction.
A no-weight concurrent test verifies shared serialization, callback userdata and
model parameter routing against an explicit test backend.
The edit smoke compares C++ and C ABI edits, checks input immutability, progress,
error recovery and output lifetime, then also exercises text generation after an
edit in the same session. It needs an existing reference image (default
`outputs/klein-reference.png`). Image tests cover PNG alpha, JPEG, corrupted input,
RGB lengths and reference bounds without weights.

For the numerical reference, use the pinned `sd-cli` with the same split weights,
`--diffusion-fa --steps 4 --cfg-scale 1 --sampling-method euler --scheduler discrete`,
matching dimensions, prompt and seed. Compare decoded RGB pixels. Different
quantization formats/frameworks are not expected to agree pixel-for-pixel.
For reference editing, add `-r source.png` (repeat for multiple references). Match
`--disable-auto-resize-ref-image` when native reference sizing is selected.

## Batches

`--count` uses bounded native batches (8 images / 16777216 output pixels per batch),
sharing text conditioning and all reference latents. C++ exposes `generate_batch`
and `edit_batch`; C exposes `pictor_session_generate_batch` and
`pictor_flux_klein_session_edit_batch`. Images use consecutive seeds and return
together after decoding. The batch timing is explicit; each image's timing is an
amortized average. This does not add a cross-request cache. See README for ownership.

## Optional hidden-state compression (HS)

**ggml-only, experimental and disabled by default.** This is an approximate acceleration mode for Klein, not a
pixel-equivalent optimization. It can change fine details, text and reference
fidelity. Treat visual inspection of your workload as part of choosing this mode.

The current 512x512, four-step Q4 test produced **obvious repeated contours and
striping**, despite passing numerical and API tests. Its generation time fell
from 110.280 s to 92.168 s on an M4, but that result does not establish an acceptable
quality/speed tradeoff. The original Python settings were selected for 2K work;
larger-resolution quality in this ggml port has not yet been validated. Keep HS
off for final-quality work unless your own comparisons justify enabling it.

```cpp
// On an existing FluxKleinSession; returns Status, like generation.
auto status = session->set_hidden_state_compression(true);
if (!status) { std::fprintf(stderr, "%s\n", status.message); return 1; }
// generate(), edit(), generate_batch() and edit_batch() now use HS.
// To restore the exact path:
status = session->set_hidden_state_compression(false);
```

```c
pictor_status status = pictor_flux_klein_session_set_hidden_state_compression(
    session, 1, &error); /* 0 disables; reject any other value, NULL, or Anima */
if (status != PICTOR_OK) { fprintf(stderr, "%s\n", error.message); return 1; }
/* Existing generate/edit/batch calls apply the selected mode. */
```

Zig can call the same function by importing `pictor/pictor.h` with `@cImport`/`@cInclude`; Rust's
updated example declares the function with `enabled: u32`. All existing C ABI v1
struct layouts and generation signatures remain unchanged.

The setting belongs to the session and persists until changed. Setter and inference
use the shared backend lock. A setter waits for an active generation; it never
changes an in-flight batch. Setting and then generating are two separate calls;
callers sharing one session must coordinate if they require per-request modes.
Callbacks must not call the setter or re-enter pictor.

```sh
zig-out/bin/pictor flux-klein -p "a red fox in a forest" \
  --backend ggml --seed 666 --steps 4 --hs-compression --count 2 -o outputs/hs.png
```

CLI metadata records `hidden_state_compression`; verbose logs show compressed/full
steps and the number of image tokens. Anima rejects this flag. A one-step request
still uses the full path even when HS is enabled.

Algorithm: for each Euler step except the last, pool 2x2 image tokens separately
for the output and each reference in single-stream transformer blocks. Text tokens
remain full length. Run the block on reduced tokens, expand only its image update,
and add it to the original features. Text outputs come from the reduced block.
Use representative existing rotary position embeddings, retaining each reference's
positional identity. Double-stream blocks and the final Euler step remain full.
The final full step does not undo the approximations in earlier steps.

Unlike the original square-grid heuristic, this port uses actual latent dimensions.
Non-square outputs, ordered references of different dimensions, and odd token-grid
edges are supported. Partial 2x2 edge cells average their valid entries by repeating
the final row/column; no padding from another image is included. Each batch image
uses its own step schedule. There is no persistent activation cache.

`zig build test` includes a scalar-reference comparison of pooling, position
selection and residual expansion; Metal builds also compare that graph with CPU
numerically. `zig build smoke-hs` checks real Metal inference,
one-step equivalence, off/on/off recovery, odd rectangular multi-reference batch
parity, and writes a 512x512 four-step comparison to `outputs/hs-smoke/`.
Successful smoke execution verifies functionality, not visual quality.
