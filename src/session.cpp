#include "session.hpp"
#include "inference_lock.hpp"
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
    void* userdata;
    BatchProgressCallback progress;
    int image_count;

    explicit Callbacks(bool verbose_value, BatchProgressCallback callback = nullptr, void* data = nullptr, int count = 1)
        : verbose(verbose_value), userdata(data), progress(callback), image_count(count) {
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
            const int index = sd_get_sampling_image_index();
            if (self.progress && index >= 0 && index < self.image_count)
                self.progress({index, self.image_count, {step, steps, seconds}}, self.userdata);
        }, this);
    }

    ~Callbacks() {
        sd_set_progress_callback(nullptr, nullptr);
        sd_set_log_callback(nullptr, nullptr);
    }
};

struct ImageDeleter {
    int count;
    void operator()(sd_image_t* images) const {
        if (images) {
            for (int i = 0; i < count; ++i) std::free(images[i].data);
            std::free(images);
        }
    }
};
} // namespace

std::mutex& inference_mutex() noexcept { return backend_mutex; }

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

Status Session::set_hidden_state_compression(bool enabled) noexcept {
    if (model_ != Model::flux_klein)
        return failure(ErrorCode::invalid_argument, "hidden-state compression requires Klein");
    std::lock_guard<std::mutex> lock(backend_mutex);
    hidden_state_compression_ = enabled;
    return {};
}

Status Session::generate(const GenerationRequest& request, Image& image,
                         ProgressCallback progress, void* userdata, const FluxKleinEditRequest* edit) noexcept {
    image = {};
    struct Adapter { ProgressCallback function; void* userdata; } adapter{progress, userdata};
    const auto callback = [](const BatchProgress& value, void* data) noexcept {
        auto& adapter = *static_cast<Adapter*>(data);
        adapter.function(value.sampling, adapter.userdata);
    };
    BatchResult result;
    const auto status = generate_batch(request, 1, result, progress ? callback : static_cast<BatchProgressCallback>(nullptr), &adapter, edit);
    if (status) image = std::move(result.images.front());
    return status;
}

Status Session::generate_batch(const GenerationRequest& request, int count, BatchResult& result,
                               BatchProgressCallback progress, void* userdata, const FluxKleinEditRequest* edit) noexcept {
    result = {};
    if (const auto status = validate_batch_request(request, count); !status) return status;
    if (edit) {
        if (model_ != Model::flux_klein) return failure(ErrorCode::invalid_argument, "reference editing requires Klein");
        if (const auto status = validate_flux_klein_edit_request(*edit); !status) return status;
    }
    const auto validation = model_ == Model::anima ? validate_request(request) : validate_flux_klein_request(request);
    if (!validation) return validation;
    std::int64_t seed;
    if (const auto status = resolve_seed(request.seed, seed); !status) return status;
    std::lock_guard<std::mutex> lock(backend_mutex);
    Callbacks callbacks(verbose_, progress, userdata, count);
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = request.prompt.c_str();
    params.negative_prompt = request.negative_prompt.c_str();
    params.width = request.width;
    params.height = request.height;
    params.seed = seed;
    params.batch_count = count;
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
    const auto status = backend::generate(context_, params, raw_output, hidden_state_compression_);
    std::unique_ptr<sd_image_t, ImageDeleter> output(raw_output, ImageDeleter{count});
    const auto seconds = elapsed(start);
    if (!status) return status;
    // Validate the entire array before publishing any images.
    for (int i = 0; i < count; ++i) {
        const auto& item = raw_output[i];
        if (!item.data || item.width != static_cast<unsigned>(request.width) ||
            item.height != static_cast<unsigned>(request.height) || item.channel != 3)
            return failure(ErrorCode::backend_error, "backend returned invalid image data, dimensions or channels");
    }
    for (int i = 0; i < count; ++i) {
        const auto& item = raw_output[i];
        const auto size = static_cast<std::size_t>(item.width) * item.height * item.channel;
        result.images.push_back({request.width, request.height, 3, {item.data, item.data + size}, seed + i, seconds / count});
    }
    result.generation_seconds = seconds;
    return {};
}

} // namespace pictor::detail
