#include "pictor/pictor.h"
#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"

#include <new>

struct pictor_session {
    // Exactly one owner is set by a successful factory; handles are opaque to callers.
    std::unique_ptr<pictor::AnimaSession> value;
    std::unique_ptr<pictor::FluxKleinSession> klein;
    double load_seconds() const noexcept { return value ? value->load_seconds() : klein->load_seconds(); }
    pictor::Status generate(const pictor::GenerationRequest& request, pictor::Image& image,
                           pictor::ProgressCallback progress, void* userdata) noexcept {
        return value ? value->generate(request, image, progress, userdata) : klein->generate(request, image, progress, userdata);
    }
    pictor::Status generate_batch(const pictor::GenerationRequest& request, int count, pictor::BatchResult& output,
                                 pictor::BatchProgressCallback progress, void* userdata) noexcept {
        return value ? value->generate_batch(request, count, output, progress, userdata)
                     : klein->generate_batch(request, count, output, progress, userdata);
    }
};
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

Status convert_edit(const pictor_request* request, const pictor_flux_klein_edit_options* options,
                    pictor::FluxKleinEditRequest& value) noexcept {
    if (!options || options->struct_size != sizeof(*options) || options->auto_resize > 1 ||
        !options->reference_images || options->reference_images_count < 1 || options->reference_images_count > 4)
        return failure(ErrorCode::invalid_argument, "invalid Klein edit options; expected 1..4 references");
    if (const auto status = convert_request(request, value.generation); !status) return status;
    value.auto_resize = options->auto_resize != 0;
    for (size_t i = 0; i < options->reference_images_count; ++i) {
        const auto& ref = options->reference_images[i];
        if (ref.struct_size != sizeof(ref))
            return failure(ErrorCode::invalid_argument, "invalid reference image struct_size");
        value.reference_images.push_back({ref.width, ref.height, ref.pixels, ref.pixels_len});
    }
    return pictor::validate_flux_klein_edit_request(value);
}

struct BatchCallback {
    pictor_batch_progress_callback function;
    void* userdata;
    static void invoke(const pictor::BatchProgress& value, void* data) noexcept {
        const auto& self = *static_cast<const BatchCallback*>(data);
        self.function(value.image_index, value.image_count, value.sampling.step, value.sampling.steps,
                      value.sampling.seconds, self.userdata);
    }
};

pictor_status generate_batch(pictor_session* session, const pictor_request* request,
    const pictor_flux_klein_edit_options* options, bool edit, int32_t count,
    pictor_batch_progress_callback progress, void* userdata, pictor_image** outputs,
    double* batch_seconds, pictor_error* error) noexcept {
    if (batch_seconds) *batch_seconds = 0;
    if (!session || !outputs || !batch_seconds || count < 1 || count > pictor::max_batch_count)
        return finish(failure(ErrorCode::invalid_argument, "invalid batch session, outputs, timing pointer or count"), error);
    for (int i = 0; i < count; ++i)
        if (outputs[i]) return finish(failure(ErrorCode::invalid_argument, "all batch image slots must be NULL"), error);
    if (edit && !session->klein)
        return finish(failure(ErrorCode::invalid_argument, "reference editing requires a Klein session"), error);
    pictor::FluxKleinEditRequest value;
    const auto converted = edit ? convert_edit(request, options, value) : convert_request(request, value.generation);
    if (!converted) return finish(converted, error);
    if (const auto status = pictor::validate_batch_request(value.generation, count); !status) return finish(status, error);
    std::unique_ptr<pictor_image> handles[pictor::max_batch_count];
    for (int i = 0; i < count; ++i) {
        handles[i].reset(new (std::nothrow) pictor_image);
        if (!handles[i]) return finish(failure(ErrorCode::out_of_memory, "cannot allocate batch image handle"), error);
    }
    pictor::BatchResult result;
    BatchCallback callback{progress, userdata};
    const auto function = progress ? BatchCallback::invoke : nullptr;
    const auto status = edit ? session->klein->edit_batch(value, count, result, function, &callback)
                             : session->generate_batch(value.generation, count, result, function, &callback);
    if (!status) return finish(status, error);
    for (int i = 0; i < count; ++i) {
        handles[i]->value = std::move(result.images[i]);
        outputs[i] = handles[i].release();
    }
    *batch_seconds = result.generation_seconds;
    return finish({}, error);
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
    return pictor_anima_session_create_with_backend(options, PICTOR_ANIMA_BACKEND_AUTO, output, error);
}

pictor_status pictor_anima_request_init(pictor_request* output, size_t size, int32_t preset, pictor_error* error) noexcept {
    const auto status = pictor_request_init(output, size, preset, error);
    if (status == PICTOR_OK) output->cache = PICTOR_CACHE_NONE;
    return status;
}

pictor_status pictor_anima_session_backend(const pictor_session* session, pictor_anima_backend* output, pictor_error* error) noexcept {
    if (output) *output = PICTOR_ANIMA_BACKEND_AUTO;
    if (!session || !session->value || !output)
        return finish(failure(ErrorCode::invalid_argument, "expected an Anima session and backend output"), error);
    *output = static_cast<pictor_anima_backend>(session->value->backend());
    return finish({}, error);
}

pictor_status pictor_anima_session_create_with_backend(const pictor_session_options* options, pictor_anima_backend backend,
                                                      pictor_session** output, pictor_error* error) noexcept {
    if (backend < PICTOR_ANIMA_BACKEND_AUTO || backend > PICTOR_ANIMA_BACKEND_MLX)
        return finish(failure(ErrorCode::invalid_argument, "invalid Anima backend"), error);
    if (!output || *output)
        return finish(failure(ErrorCode::invalid_argument, "session output must point to NULL"), error);
    if (!options || options->struct_size != sizeof(*options) || !options->model_path || options->verbose > 1)
        return finish(failure(ErrorCode::invalid_argument, "invalid session options"), error);
    auto session = std::unique_ptr<pictor_session>(new (std::nothrow) pictor_session);
    if (!session) return finish(failure(ErrorCode::out_of_memory, "cannot allocate session handle"), error);
    const auto status = pictor::AnimaSession::create({options->model_path, options->threads, options->verbose != 0},
        static_cast<pictor::AnimaBackend>(backend), session->value);
    if (status) *output = session.release();
    return finish(status, error);
}

pictor_status pictor_flux_klein_options_init(pictor_flux_klein_options* output, size_t size, pictor_error* error) noexcept {
    if (!output || size != sizeof(*output))
        return finish(failure(ErrorCode::invalid_argument, "invalid Klein options pointer or size"), error);
    *output = {sizeof(*output), nullptr, nullptr, nullptr, 0, 0};
    return finish({}, error);
}

pictor_status pictor_flux_klein_request_init(pictor_request* output, size_t size, pictor_error* error) noexcept {
    if (!output || size != sizeof(*output))
        return finish(failure(ErrorCode::invalid_argument, "invalid request pointer or size"), error);
    const auto value = pictor::flux_klein_request();
    *output = {sizeof(*output), nullptr, nullptr, value.width, value.height, value.steps,
               value.cfg_scale, value.seed, PICTOR_CACHE_NONE, 0};
    return finish({}, error);
}

pictor_status pictor_flux_klein_session_create(const pictor_flux_klein_options* options, pictor_session** output, pictor_error* error) noexcept {
    return pictor_flux_klein_session_create_with_backend(options, PICTOR_KLEIN_BACKEND_AUTO, output, error);
}

pictor_status pictor_flux_klein_session_backend(const pictor_session* session, pictor_klein_backend* output, pictor_error* error) noexcept {
    if (output) *output = PICTOR_KLEIN_BACKEND_AUTO;
    if (!session || !session->klein || !output)
        return finish(failure(ErrorCode::invalid_argument, "expected a Klein session and backend output"), error);
    *output = static_cast<pictor_klein_backend>(session->klein->backend());
    return finish({}, error);
}

pictor_status pictor_flux_klein_session_create_with_backend(const pictor_flux_klein_options* options, pictor_klein_backend backend,
                                                          pictor_session** output, pictor_error* error) noexcept {
    if (backend < PICTOR_KLEIN_BACKEND_AUTO || backend > PICTOR_KLEIN_BACKEND_MLX)
        return finish(failure(ErrorCode::invalid_argument, "invalid Klein backend"), error);
    if (!output || *output)
        return finish(failure(ErrorCode::invalid_argument, "session output must point to NULL"), error);
    if (!options || options->struct_size != sizeof(*options) || !options->diffusion_model_path ||
        !options->text_encoder_path || !options->vae_path || options->verbose > 1)
        return finish(failure(ErrorCode::invalid_argument, "invalid Klein session options"), error);
    auto session = std::unique_ptr<pictor_session>(new (std::nothrow) pictor_session);
    if (!session) return finish(failure(ErrorCode::out_of_memory, "cannot allocate session handle"), error);
    const auto status = pictor::FluxKleinSession::create({options->diffusion_model_path, options->text_encoder_path,
        options->vae_path, options->threads, options->verbose != 0}, static_cast<pictor::KleinBackend>(backend), session->klein);
    if (status) *output = session.release();
    return finish(status, error);
}

pictor_status pictor_session_load_seconds(const pictor_session* session, double* output, pictor_error* error) noexcept {
    if (output) *output = 0;
    if (!session || !output) return finish(failure(ErrorCode::invalid_argument, "session and output must not be NULL"), error);
    *output = session->load_seconds();
    return finish({}, error);
}

pictor_status pictor_flux_klein_edit_options_init(pictor_flux_klein_edit_options* output, size_t size, pictor_error* error) noexcept {
    if (!output || size != sizeof(*output))
        return finish(failure(ErrorCode::invalid_argument, "invalid edit options pointer or size"), error);
    *output = {sizeof(*output), nullptr, 0, 1};
    return finish({}, error);
}

pictor_status pictor_flux_klein_session_set_hidden_state_compression(pictor_session* session,
    uint32_t enabled, pictor_error* error) noexcept {
    if (!session || !session->klein || enabled > 1)
        return finish(failure(ErrorCode::invalid_argument, "expected a Klein session and enabled=0 or 1"), error);
    return finish(session->klein->set_hidden_state_compression(enabled != 0), error);
}

pictor_status pictor_flux_klein_session_edit(pictor_session* session, const pictor_request* request,
    const pictor_flux_klein_edit_options* options, pictor_progress_callback progress, void* userdata,
    pictor_image** output, pictor_error* error) noexcept {
    if (!output || *output)
        return finish(failure(ErrorCode::invalid_argument, "image output must point to NULL"), error);
    if (!session || !session->klein)
        return finish(failure(ErrorCode::invalid_argument, "reference editing requires a Klein session"), error);
    pictor::FluxKleinEditRequest value;
    if (const auto status = convert_edit(request, options, value); !status) return finish(status, error);
    auto image = std::unique_ptr<pictor_image>(new (std::nothrow) pictor_image);
    if (!image) return finish(failure(ErrorCode::out_of_memory, "cannot allocate image handle"), error);
    Callback callback{progress, userdata};
    const auto status = session->klein->edit(value, image->value, progress ? Callback::invoke : nullptr, &callback);
    if (status) *output = image.release();
    return finish(status, error);
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
    const auto status = session->generate(value, image->value, progress ? Callback::invoke : nullptr, &callback);
    if (status) *output = image.release();
    return finish(status, error);
}

pictor_status pictor_session_generate_batch(pictor_session* session, const pictor_request* request,
    int32_t count, pictor_batch_progress_callback progress, void* userdata,
    pictor_image** outputs, double* batch_seconds, pictor_error* error) noexcept {
    return generate_batch(session, request, nullptr, false, count, progress, userdata, outputs, batch_seconds, error);
}
pictor_status pictor_flux_klein_session_edit_batch(pictor_session* session, const pictor_request* request,
    const pictor_flux_klein_edit_options* options, int32_t count,
    pictor_batch_progress_callback progress, void* userdata,
    pictor_image** outputs, double* batch_seconds, pictor_error* error) noexcept {
    return generate_batch(session, request, options, true, count, progress, userdata, outputs, batch_seconds, error);
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

pictor_status pictor_image_load(const char* path, pictor_image** output, pictor_error* error) noexcept {
    if (!output || *output || !path)
        return finish(failure(ErrorCode::invalid_argument, "image output must point to NULL and path must not be NULL"), error);
    auto image = std::unique_ptr<pictor_image>(new (std::nothrow) pictor_image);
    if (!image) return finish(failure(ErrorCode::out_of_memory, "cannot allocate image handle"), error);
    const auto status = pictor::read_image(path, image->value);
    if (status) *output = image.release();
    return finish(status, error);
}
} // extern "C"
