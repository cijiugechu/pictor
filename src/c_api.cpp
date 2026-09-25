#include "pictor/pictor.h"
#include "pictor/anima.hpp"

#include <new>

struct pictor_session { std::unique_ptr<pictor::AnimaSession> value; };
struct pictor_image { pictor::Image value; };

namespace {
using pictor::ErrorCode;
using pictor::Status;
using pictor::failure;
static_assert(static_cast<int>(ErrorCode::ok) == PICTOR_OK);
static_assert(static_cast<int>(ErrorCode::invalid_argument) == PICTOR_INVALID_ARGUMENT);
static_assert(static_cast<int>(ErrorCode::io_error) == PICTOR_IO_ERROR);
static_assert(static_cast<int>(ErrorCode::backend_error) == PICTOR_BACKEND_ERROR);
static_assert(static_cast<int>(ErrorCode::out_of_memory) == PICTOR_OUT_OF_MEMORY);

pictor_status finish(Status status, pictor_error* error) noexcept {
    if (error) std::memcpy(error->message, status.message, sizeof(error->message));
    return static_cast<pictor_status>(status.code);
}

Status convert_request(const pictor_request* input, pictor::GenerationRequest& output) noexcept {
    if (!input || input->struct_size != sizeof(*input))
        return failure(ErrorCode::invalid_argument, "invalid request pointer or struct_size");
    if (!input->prompt) return failure(ErrorCode::invalid_argument, "prompt must not be NULL");
    if (input->cache != PICTOR_CACHE_NONE && input->cache != PICTOR_CACHE_SPECTRUM)
        return failure(ErrorCode::invalid_argument, "unsupported cache mode");
    if (input->vae_tiling > 1) return failure(ErrorCode::invalid_argument, "vae_tiling must be 0 or 1");
    output = {input->prompt, input->negative_prompt ? input->negative_prompt : "",
              input->width, input->height, input->steps, input->cfg_scale, input->seed,
              input->cache == PICTOR_CACHE_NONE ? pictor::CacheMode::none : pictor::CacheMode::spectrum,
              input->vae_tiling != 0};
    return pictor::validate_request(output);
}

struct Callback {
    pictor_progress_callback function;
    void* userdata;
    static void invoke(const pictor::Progress& progress, void* data) noexcept {
        const auto& self = *static_cast<const Callback*>(data);
        self.function(progress.step, progress.steps, progress.seconds, self.userdata);
    }
};
} // namespace

extern "C" {
uint32_t pictor_abi_version(void) noexcept { return PICTOR_ABI_VERSION; }

pictor_status pictor_session_options_init(pictor_session_options* output, size_t size, pictor_error* error) noexcept {
    if (!output || size != sizeof(*output))
        return finish(failure(ErrorCode::invalid_argument, "invalid options pointer or size"), error);
    *output = {sizeof(*output), nullptr, 0, 0};
    return finish({}, error);
}

pictor_status pictor_request_init(pictor_request* output, size_t size, int32_t preset, pictor_error* error) noexcept {
    if (!output || size != sizeof(*output))
        return finish(failure(ErrorCode::invalid_argument, "invalid request pointer or size"), error);
    *output = {};
    if (preset < PICTOR_PRESET_FAST || preset > PICTOR_PRESET_QUALITY)
        return finish(failure(ErrorCode::invalid_argument, "unsupported preset"), error);
    const auto value = pictor::preset_request(static_cast<pictor::Preset>(preset));
    *output = {sizeof(*output), nullptr, nullptr, value.width, value.height, value.steps,
               value.cfg_scale, value.seed, value.cache == pictor::CacheMode::none ? PICTOR_CACHE_NONE : PICTOR_CACHE_SPECTRUM, 0};
    return finish({}, error);
}

pictor_status pictor_request_validate(const pictor_request* request, pictor_error* error) noexcept {
    pictor::GenerationRequest value;
    return finish(convert_request(request, value), error);
}

pictor_status pictor_session_create(const pictor_session_options* options, pictor_session** output, pictor_error* error) noexcept {
    if (!output || *output)
        return finish(failure(ErrorCode::invalid_argument, "session output must point to NULL"), error);
    if (!options || options->struct_size != sizeof(*options) || !options->model_path || options->verbose > 1)
        return finish(failure(ErrorCode::invalid_argument, "invalid session options"), error);
    auto session = std::unique_ptr<pictor_session>(new (std::nothrow) pictor_session);
    if (!session) return finish(failure(ErrorCode::out_of_memory, "cannot allocate session handle"), error);
    const auto status = pictor::AnimaSession::create({options->model_path, options->threads, options->verbose != 0}, session->value);
    if (status) *output = session.release();
    return finish(status, error);
}

pictor_status pictor_session_load_seconds(const pictor_session* session, double* output, pictor_error* error) noexcept {
    if (output) *output = 0;
    if (!session || !output) return finish(failure(ErrorCode::invalid_argument, "session and output must not be NULL"), error);
    *output = session->value->load_seconds();
    return finish({}, error);
}

pictor_status pictor_session_generate(pictor_session* session, const pictor_request* request,
    pictor_progress_callback progress, void* userdata, pictor_image** output, pictor_error* error) noexcept {
    if (!output || *output)
        return finish(failure(ErrorCode::invalid_argument, "image output must point to NULL"), error);
    if (!session) return finish(failure(ErrorCode::invalid_argument, "session must not be NULL"), error);
    pictor::GenerationRequest value;
    if (const auto status = convert_request(request, value); !status) return finish(status, error);
    auto image = std::unique_ptr<pictor_image>(new (std::nothrow) pictor_image);
    if (!image) return finish(failure(ErrorCode::out_of_memory, "cannot allocate image handle"), error);
    Callback callback{progress, userdata};
    const auto status = session->value->generate(value, image->value, progress ? Callback::invoke : nullptr, &callback);
    if (status) *output = image.release();
    return finish(status, error);
}

void pictor_session_destroy(pictor_session* session) noexcept { delete session; }

pictor_status pictor_image_get_info(const pictor_image* image, pictor_image_info* output, size_t size, pictor_error* error) noexcept {
    if (!output || size != sizeof(*output))
        return finish(failure(ErrorCode::invalid_argument, "invalid image info pointer or size"), error);
    *output = {};
    if (!image) return finish(failure(ErrorCode::invalid_argument, "image must not be NULL"), error);
    const auto& value = image->value;
    *output = {value.width, value.height, value.channels, value.pixels.data(), value.pixels.size(), value.seed, value.generation_seconds};
    return finish({}, error);
}

pictor_status pictor_image_write_png(const pictor_image* image, const char* path, pictor_error* error) noexcept {
    if (!image || !path) return finish(failure(ErrorCode::invalid_argument, "image and path must not be NULL"), error);
    return finish(pictor::write_png(path, image->value), error);
}

void pictor_image_destroy(pictor_image* image) noexcept { delete image; }
} // extern "C"
