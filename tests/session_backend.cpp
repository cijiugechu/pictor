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
static thread_local int sampling_index = -1;
static int generation_calls = 0;
static thread_local bool hs_enabled = false;
extern "C" int pictor_test_generation_calls() { return generation_calls; }
static std::atomic<int> active{0};
static std::atomic<int> overlaps{0};
extern "C" int pictor_test_backend_overlaps() { return overlaps.load(); }

extern "C" {
int sd_get_sampling_image_index(void) { return sampling_index; }
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
        (context->klein && params->cache.mode != SD_CACHE_DISABLED) || (params->batch_count < 1 || params->batch_count > 8)) return nullptr;
    ++generation_calls;
    if (active.fetch_add(1) != 0) overlaps++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // Non-sampling events from VAE tiling must not acquire a sampling index.
    if (callback) callback(1, 1, 0.1f, callback_data);
    const int steps = context->klein ? 4 : 3;
    for (int i = 0; i < params->batch_count; ++i) {
        sampling_index = i;
        if (callback) {
            callback(0, steps, 0, callback_data);
            callback(steps, steps, 0.1f, callback_data);
        }
    }
    sampling_index = -1;
    if (callback) callback(1, 1, 0.1f, callback_data);
    active--;
    if (std::strcmp(params->prompt, "fail") == 0) return nullptr;
    auto* images = static_cast<sd_image_t*>(std::calloc(params->batch_count, sizeof(sd_image_t)));
    if (!images) return nullptr;
    for (int i = 0; i < params->batch_count; ++i) {
        auto& image = images[i];
        image.width = params->width;
        image.height = params->height;
        image.channel = 3;
        const auto size = static_cast<size_t>(image.width) * image.height * 3;
        if (i == params->batch_count - 1 && std::strcmp(params->prompt, "null-last") == 0) continue;
        image.data = static_cast<uint8_t*>(std::malloc(size));
        int color = context->klein ? 22 : 11;
        if (hs_enabled && params->sample_params.sample_steps > 1) color += 100;
        for (int j = 0; j < params->ref_images_count; ++j)
            color += (j + 1) * params->ref_images[j].data[0];
        if (std::strcmp(params->prompt, "seed-colors") == 0) color += (params->seed + i) % 100;
        if (image.data) std::memset(image.data, color, size);
        if (i == params->batch_count - 1 && std::strcmp(params->prompt, "bad-last") == 0) image.width++;
    }
    return images;
}
}

extern "C" sd_image_t* generate_image_with_hs(sd_ctx_t* context, const sd_img_gen_params_t* params) {
    if (!context->klein) return nullptr;
    hs_enabled = true;
    auto* output = generate_image(context, params);
    hs_enabled = false;
    return output;
}
