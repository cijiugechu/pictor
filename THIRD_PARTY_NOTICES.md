# Third-party components

- **spdlog 1.17.0**, Gabi Melman and contributors, MIT. Pinned source and license:
  `vendor/spdlog/LICENSE`. Its header-only distribution includes fmt, whose license
  is also reproduced in that file.

- **stable-diffusion.cpp**, Copyright (c) 2023 leejet, MIT.
  Source and license: `vendor/stable-diffusion.cpp/LICENSE`.
- **ggml**, license in `vendor/stable-diffusion.cpp/ggml/LICENSE`.
- **stb_image_write**, used from the pinned sd.cpp checkout. Its MIT/public-domain
  dual-license text is embedded in `vendor/stable-diffusion.cpp/thirdparty/stb_image_write.h`.
- The Anima Metal patch is copied unchanged from
  `ultra-fast-image-gen/patches/anima-ggml-metal-im2col3d-pad.patch`, as are the
  original runtime pin and preset values. Project:
  https://github.com/newideas99/ultra-fast-image-gen.
- Anima model weights are downloaded separately and retain their own license:
  https://huggingface.co/n-Arno/Anima-P3-Turbo-AIO-Q4_K.

The native library links upstream dependencies. Preserve their license notices
when distributing binaries. This file does not relicense any source or model.
