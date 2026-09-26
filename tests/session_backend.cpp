// Deterministic sd.cpp stand-in: exercises real pictor session routing and locking.
#include "stable-diffusion.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

struct sd_ctx_t { bool klein; };
static sd_progress_cb_t callback;
static void* callback_data;
static std::atomic<int> active{0};
static std::atomic<int> overlaps{0};
extern "C" int pictor_test_backend_overlaps() { return overlaps.load(); }

extern "C" {
void sd_set_log_callback(sd_log_cb_t, void*) {}
void sd_set_progress_callback(sd_progress_cb_t function, void* data) { callback = function; callback_data = data; }
void sd_ctx_params_init(sd_ctx_params_t* params) { *params = {}; params->vae_decode_only = true; }
void sd_img_gen_params_init(sd_img_gen_params_t* params) { *params = {}; }
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params) {
    const bool klein = params->diffusion_model_path != nullptr;
    if (params->free_params_immediately || params->vae_decode_only == klein) return nullptr;
    if (klein && (!params->llm_path || !params->vae_path || !params->diffusion_flash_attn || params->flash_attn)) return nullptr;
    if (!klein && (!params->model_path || !params->flash_attn)) return nullptr;
    auto* context = static_cast<sd_ctx_t*>(std::malloc(sizeof(sd_ctx_t)));
    if (context) context->klein = klein;
    return context;
}
void free_sd_ctx(sd_ctx_t* context) { std::free(context); }
sd_image_t* generate_image(sd_ctx_t* context, const sd_img_gen_params_t* params) {
    if (params->sample_params.sample_method != (context->klein ? EULER_SAMPLE_METHOD : ER_SDE_SAMPLE_METHOD) ||
        params->sample_params.scheduler != (context->klein ? DISCRETE_SCHEDULER : SMOOTHSTEP_SCHEDULER) ||
        (context->klein && params->cache.mode != SD_CACHE_DISABLED) || params->batch_count != 1) return nullptr;
    if (active.fetch_add(1) != 0) overlaps++;
    // Pause to expose accidental per-model locks and callback replacement.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const int steps = context->klein ? 4 : 3;
    if (callback) callback(steps, steps, 0.1f, callback_data);
    active--;
    auto* image = static_cast<sd_image_t*>(std::calloc(1, sizeof(sd_image_t)));
    if (!image) return nullptr;
    image->width = params->width;
    image->height = params->height;
    image->channel = 3;
    const auto size = static_cast<size_t>(image->width) * image->height * 3;
    image->data = static_cast<uint8_t*>(std::malloc(size));
    int color = context->klein ? 22 : 11;
    for (int i = 0; i < params->ref_images_count; ++i)
        color += (i + 1) * params->ref_images[i].data[0];
    if (image->data) std::memset(image->data, color, size);
    return image;
}
}
