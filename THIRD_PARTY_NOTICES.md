# Third-party components

- **spdlog 1.17.0**, Gabi Melman and contributors, MIT. Pinned source and license:
  `vendor/spdlog/LICENSE`. Its header-only distribution includes fmt, whose license
  is also reproduced in that file.

- **stable-diffusion.cpp**, Copyright (c) 2023 leejet, MIT.
  Source and license: `vendor/stable-diffusion.cpp/LICENSE`.
  Pictor additionally applies `patches/sd-model-progress-callback.patch` to keep
  split-file loader output quiet when a caller installs a progress callback.
  `patches/sd-sampling-image-index.patch` adds a thread-local sampling-index query
  so batch progress can distinguish sampling from VAE tiling without parsing logs.
  `patches/sd-flux-hidden-state-compression.patch` adds opt-in Klein compression:
  the ggml implementation follows the 2x2 residual-update algorithm in
  `ultra-fast-image-gen/flux2_sdnq_hs.py`, with explicit per-image rectangular grids
  and partial edge cells. It does not change upstream defaults or existing ABI layouts.
- **ggml**, license in `vendor/stable-diffusion.cpp/ggml/LICENSE`.
- **stb_image_write**, used from the pinned sd.cpp checkout. Its MIT/public-domain
  dual-license text is embedded in `vendor/stable-diffusion.cpp/thirdparty/stb_image_write.h`.
- **stb_image**, PNG/JPEG decoding from the same pinned checkout; MIT/public-domain
  dual-license text is in `vendor/stable-diffusion.cpp/thirdparty/stb_image.h`.
- The Anima Metal patch is copied unchanged from
  `ultra-fast-image-gen/patches/anima-ggml-metal-im2col3d-pad.patch`, as are the
  original runtime pin and preset values. Project:
  https://github.com/newideas99/ultra-fast-image-gen.
- Anima model weights are downloaded separately and retain their own license:
  https://huggingface.co/n-Arno/Anima-P3-Turbo-AIO-Q4_K.
- Klein diffusion, Qwen3 text encoder and Flux2 VAE weights are also downloaded
  separately. Their source repositories, pinned revisions and checksums are listed
  in [docs/klein.md](docs/klein.md); they retain their respective source licenses.

The native library links upstream dependencies. Preserve their license notices
when distributing binaries. This file does not relicense any source or model.

## MLX and MFLUX

Native Klein on Apple Silicon links MLX 0.32.2 (Apple, MIT license); its runtime
license is installed alongside pictor. The C++ Qwen3, Klein and Flux2 VAE
implementations in `src/mlx/` are adapted from MFLUX 0.20.0 (Filip Strand, MIT license), with native tokenization, reference preparation and
pictor sampling integration. See `licenses/mflux.txt`. Python/MFLUX is used only
for reference tests; it is not loaded by the inference library.
