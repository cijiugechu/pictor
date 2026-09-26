#include "session.hpp"
#include "pictor/flux_klein.hpp"
#include "backend.hpp"
#include "logging.hpp"

#include <chrono>
#include <cstdlib>
#include <new>
#include <mutex>
#include <string_view>

namespace pictor::detail {
namespace {
using Clock = std::chrono::steady_clock;
std::mutex backend_mutex;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Callbacks {
    bool verbose;
    ProgressCallback progress;
    void* userdata;

    explicit Callbacks(bool verbose_value, ProgressCallback callback = nullptr, void* data = nullptr)
        : verbose(verbose_value), progress(callback), userdata(data) {
        sd_set_log_callback([](sd_log_level_t level, const char* text, void* data) noexcept {
            auto& self = *static_cast<Callbacks*>(data);
            if (!self.verbose && level < SD_LOG_WARN) return;
            auto severity = spdlog::level::info;
            switch (level) {
            case SD_LOG_DEBUG: severity = spdlog::level::debug; break;
            case SD_LOG_INFO: severity = spdlog::level::info; break;
            case SD_LOG_WARN: severity = spdlog::level::warn; break;
            case SD_LOG_ERROR: severity = spdlog::level::err; break;
            }
            std::string_view message = text ? text : "";
            while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
                message.remove_suffix(1);
            if (!message.empty()) logging::backend().log(severity, "{}", message);
        }, this);
        sd_set_progress_callback([](int step, int steps, float seconds, void* data) noexcept {
            auto& self = *static_cast<Callbacks*>(data);
            if (self.progress) self.progress({step, steps, seconds}, self.userdata);
        }, this);
    }

    ~Callbacks() {
        sd_set_progress_callback(nullptr, nullptr);
        sd_set_log_callback(nullptr, nullptr);
    }
};

struct ImageDeleter {
    void operator()(sd_image_t* images) const {
        if (images) {
            std::free(images[0].data); // Requests always contain exactly one image.
            std::free(images);
        }
    }
};
} // namespace

Session::~Session() {
    if (!context_) return;
    std::lock_guard<std::mutex> lock(backend_mutex);
    Callbacks callbacks(verbose_);
    const auto status = backend::destroy(context_);
    if (!status) logging::backend().error("{}", status.message);
}

namespace {
Status check_file(const std::filesystem::path& path, const char* label) noexcept {
    if (path.empty()) return failure(ErrorCode::invalid_argument, std::string(label) + " path must not be empty");
    std::error_code ec;
    const bool regular = std::filesystem::is_regular_file(path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
        return failure(ErrorCode::io_error, std::string("cannot inspect ") + label + ": " + ec.message());
    if (!regular) return failure(ErrorCode::invalid_argument, std::string(label) + " file not found: " + path.string());
    return {};
}
}

Status Session::create(Model model, const ModelFiles& files, int threads, bool verbose,
                       std::unique_ptr<Session>& output) noexcept {
    if (output) return failure(ErrorCode::invalid_argument, "session output must be empty");
    if (threads < 0) return failure(ErrorCode::invalid_argument, "threads must be nonnegative");
    if (const auto status = check_file(files.model, "model"); !status) return status;
    if (model == Model::flux_klein) {
        if (const auto status = check_file(files.text_encoder, "text encoder"); !status) return status;
        if (const auto status = check_file(files.vae, "VAE"); !status) return status;
    }
    auto session = std::unique_ptr<Session>(new (std::nothrow) Session(model, verbose));
    if (!session) return failure(ErrorCode::out_of_memory, "cannot allocate session");
    session->model_path_ = files.model.string();
    session->text_encoder_path_ = files.text_encoder.string();
    session->vae_path_ = files.vae.string();
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        Callbacks callbacks(verbose);
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        if (model == Model::anima) {
            params.model_path = session->model_path_.c_str();
            params.flash_attn = true;
        } else {
            params.diffusion_model_path = session->model_path_.c_str();
            params.llm_path = session->text_encoder_path_.c_str();
            params.vae_path = session->vae_path_.c_str();
            params.diffusion_flash_attn = true;
            params.vae_decode_only = false; // Both generate() and edit() share this context.
        }
        params.free_params_immediately = false;
        if (threads > 0) params.n_threads = threads;
        const auto start = Clock::now();
        const auto status = backend::create(params, session->context_);
        session->load_seconds_ = elapsed(start);
        if (!status) return status;
    }
    output = std::move(session);
    return {};
}

Status Session::generate(const GenerationRequest& request, Image& image,
                         ProgressCallback progress, void* userdata, const FluxKleinEditRequest* edit) noexcept {
    image = {};
    if (edit) {
        if (model_ != Model::flux_klein) return failure(ErrorCode::invalid_argument, "reference editing requires Klein");
        if (const auto status = validate_flux_klein_edit_request(*edit); !status) return status;
    }
    const auto validation = model_ == Model::anima ? validate_request(request) : validate_flux_klein_request(request);
    if (!validation) return validation;
    std::int64_t seed;
    if (const auto status = resolve_seed(request.seed, seed); !status) return status;
    std::lock_guard<std::mutex> lock(backend_mutex);
    Callbacks callbacks(verbose_, progress, userdata);
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = request.prompt.c_str();
    params.negative_prompt = request.negative_prompt.c_str();
    params.width = request.width;
    params.height = request.height;
    params.seed = seed;
    params.batch_count = 1;
    params.sample_params.sample_steps = request.steps;
    params.sample_params.guidance.txt_cfg = request.cfg_scale;
    params.sample_params.sample_method = model_ == Model::anima ? ER_SDE_SAMPLE_METHOD : EULER_SAMPLE_METHOD;
    params.sample_params.scheduler = model_ == Model::anima ? SMOOTHSTEP_SCHEDULER : DISCRETE_SCHEDULER;
    params.cache.mode = request.cache == CacheMode::spectrum ? SD_CACHE_SPECTRUM : SD_CACHE_DISABLED;
    params.vae_tiling_params.enabled = request.vae_tiling;

    std::vector<sd_image_t> references;
    if (edit) {
        for (const auto& ref : edit->reference_images)
            references.push_back({static_cast<uint32_t>(ref.width), static_cast<uint32_t>(ref.height), 3,
                                  const_cast<uint8_t*>(ref.pixels)}); // sd.cpp copies into float tensors.
        params.ref_images = references.data();
        params.ref_images_count = static_cast<int>(references.size());
        params.auto_resize_ref_image = edit->auto_resize;
    }

    const auto start = Clock::now();
    sd_image_t* raw_output = nullptr;
    const auto status = backend::generate(context_, params, raw_output);
    std::unique_ptr<sd_image_t, ImageDeleter> output(raw_output);
    const auto seconds = elapsed(start);
    if (!status) return status;
    if (output->width != static_cast<unsigned>(request.width) ||
        output->height != static_cast<unsigned>(request.height) || output->channel != 3)
        return failure(ErrorCode::backend_error, "backend returned unexpected image dimensions or channels");
    const auto size = static_cast<std::size_t>(output->width) * output->height * output->channel;
    image = {request.width, request.height, 3, {output->data, output->data + size}, seed, seconds};
    return {};
}

} // namespace pictor::detail
