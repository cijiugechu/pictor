#pragma once

#include "pictor/status.hpp"
#include "stable-diffusion.h"

namespace pictor::backend {
// Only backend.cpp enables exceptions. They cannot escape these entry points.
Status create(const sd_ctx_params_t& params, sd_ctx_t*& output) noexcept;
Status generate(sd_ctx_t* context, const sd_img_gen_params_t& params, sd_image_t*& output) noexcept;
Status destroy(sd_ctx_t* context) noexcept;
} // namespace pictor::backend
