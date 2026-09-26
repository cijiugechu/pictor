// Deliberately throwing upstream stand-ins, compiled separately with exceptions.
#include "stable-diffusion.h"
#include <stdexcept>
#include <new>
#include <string>

extern "C" {
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params) {
    if (params->n_threads == 1) throw std::bad_alloc();
    if (params->n_threads == 2) throw 42;
    if (params->n_threads == 3) throw std::runtime_error(std::string(1024, 'x'));
    return nullptr;
}
sd_image_t* generate_image(sd_ctx_t*, const sd_img_gen_params_t*) {
    throw std::runtime_error("injected generation failure");
}
sd_image_t* generate_image_with_hs(sd_ctx_t* ctx, const sd_img_gen_params_t* params) { return generate_image(ctx, params); }
void free_sd_ctx(sd_ctx_t*) { throw std::runtime_error("injected destruction failure"); }
}
