#include "pictor/anima.hpp"
#include "backend.hpp"
#include "logging.hpp"

#include <chrono>
#include <cstdlib>
#include <new>
#include <mutex>
#include <string_view>

namespace pictor {
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

struct AnimaSession::Impl {
    sd_ctx_t* context = nullptr;
    std::string model_path;
    bool verbose = false;
    double load_seconds = 0;

    ~Impl() {
        std::lock_guard<std::mutex> lock(backend_mutex);
        Callbacks callbacks(verbose);
        if (context) {
            const auto status = backend::destroy(context);
            if (!status) logging::backend().error("{}", status.message);
        }
    }
};

AnimaSession::AnimaSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

Status AnimaSession::create(const SessionOptions& options, std::unique_ptr<AnimaSession>& output) noexcept {
    if (output) return failure(ErrorCode::invalid_argument, "session output must be empty");
    if (options.model_path.empty()) return failure(ErrorCode::invalid_argument, "model path must not be empty");
    if (options.threads < 0) return failure(ErrorCode::invalid_argument, "threads must be nonnegative");
    std::error_code ec;
    const bool regular = std::filesystem::is_regular_file(options.model_path, ec);
    if (ec && ec != std::errc::no_such_file_or_directory)
        return failure(ErrorCode::io_error, "cannot inspect model file: " + ec.message());
    if (!regular)
        return failure(ErrorCode::invalid_argument, "model file not found: " + options.model_path.string() +
                       "; run zig build download-model or pass --model PATH");
    auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!impl) return failure(ErrorCode::out_of_memory, "cannot allocate session");
    impl->model_path = options.model_path.string();
    impl->verbose = options.verbose;
    {
        std::lock_guard<std::mutex> lock(backend_mutex);
        Callbacks callbacks(options.verbose);
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        params.model_path = impl->model_path.c_str();
        params.free_params_immediately = false;
        params.flash_attn = true;
        if (options.threads > 0) params.n_threads = options.threads;
        const auto start = Clock::now();
        const auto status = backend::create(params, impl->context);
        impl->load_seconds = elapsed(start);
        if (!status) return status;
    }
    auto* session = new (std::nothrow) AnimaSession(std::move(impl));
    if (!session) return failure(ErrorCode::out_of_memory, "cannot allocate session handle");
    output.reset(session);
    return {};
}

AnimaSession::~AnimaSession() = default;

double AnimaSession::load_seconds() const noexcept { return impl_->load_seconds; }

Status AnimaSession::generate(const GenerationRequest& request, Image& image,
                              ProgressCallback progress, void* userdata) noexcept {
    image = {};
    if (const auto status = validate_request(request); !status) return status;
    std::int64_t seed;
    if (const auto status = resolve_seed(request.seed, seed); !status) return status;
    std::lock_guard<std::mutex> lock(backend_mutex);
    Callbacks callbacks(impl_->verbose, progress, userdata);
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
    params.sample_params.sample_method = ER_SDE_SAMPLE_METHOD;
    params.sample_params.scheduler = SMOOTHSTEP_SCHEDULER;
    params.cache.mode = request.cache == CacheMode::spectrum ? SD_CACHE_SPECTRUM : SD_CACHE_DISABLED;
    params.vae_tiling_params.enabled = request.vae_tiling;

    const auto start = Clock::now();
    sd_image_t* raw_output = nullptr;
    const auto status = backend::generate(impl_->context, params, raw_output);
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

} // namespace pictor
