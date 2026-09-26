// Exception-enabled native boundary; public interfaces remain noexcept + Status.
#include "anima_mlx_session.hpp"
#include "denoiser.hpp"
#include "inference_lock.hpp"
#include "mlx/runtime.hpp"
#include "model.hpp"
#include "rng_philox.hpp"
#include <chrono>
#include <mutex>
#include <optional>

namespace pictor::detail {
using namespace anima_mlx;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
} // namespace
struct AnimaMlxSession::Impl {
    anima_mlx::Model model;
    bool verbose;
    double seconds = 0;
    Impl(const SessionOptions &options, bool int4) : model(options.model_path, int4, true), verbose(options.verbose) {}
};
AnimaMlxSession::AnimaMlxSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
AnimaMlxSession::~AnimaMlxSession() {
    std::lock_guard<std::mutex> lock(inference_mutex());
    impl_.reset();
}
Status AnimaMlxSession::create(const SessionOptions &options, std::unique_ptr<AnimaMlxSession> &output) noexcept {
    if (output)
        return failure(ErrorCode::invalid_argument, "session output must be empty");
    if (options.model_path.empty() || options.threads < 0)
        return failure(ErrorCode::invalid_argument, "Anima MLX needs a weights directory and nonnegative threads");
    try {
        const auto &directory = options.model_path;
        const bool int4 = std::filesystem::is_regular_file(directory / "transformer-int4.safetensors");
        for (const auto *name :
             {int4 ? "transformer-int4.safetensors" : "transformer-bf16.safetensors", "text_encoder-bf16.safetensors",
              "llm_adapter-bf16.safetensors", "vae-bf16.safetensors"})
            if (!std::filesystem::is_regular_file(directory / name))
                return failure(ErrorCode::invalid_argument,
                               "Anima MLX weights not found: " + (directory / name).string() +
                                   "; convert P3 with scripts/convert-anima-p3.py or select --backend ggml");
        std::lock_guard<std::mutex> lock(inference_mutex());
        RuntimeScope runtime;
        TokenizerLogScope log;
        const auto start = Clock::now();
        auto impl = std::make_unique<Impl>(options, int4);
        impl->seconds = elapsed(start);
        output.reset(new AnimaMlxSession(std::move(impl)));
        return {};
    } catch (...) {
        return exception_status();
    }
}
double AnimaMlxSession::load_seconds() const noexcept { return impl_->seconds; }
Status AnimaMlxSession::generate_batch(const GenerationRequest &r, int count, BatchResult &output,
                                       BatchProgressCallback progress, void *userdata) noexcept {
    output = {};
    if (auto status = validate_batch_request(r, count); !status)
        return status;
    if (r.cache != CacheMode::none)
        return failure(ErrorCode::invalid_argument, "Anima MLX requires cache=none; Spectrum requires ggml");
    std::int64_t seed;
    if (auto status = resolve_seed(r.seed, seed); !status)
        return status;
    try {
        std::lock_guard<std::mutex> lock(inference_mutex());
        RuntimeScope runtime;
        TokenizerLogScope log;
        const auto start = Clock::now();
        auto stage = start;
        auto context = impl_->model.context(r.prompt);
        mx::eval(context);
        std::optional<array> negative;
        // Match pinned sd.cpp, including unconditional/fractional CFG requests.
        if (r.cfg_scale != 1) {
            negative = impl_->model.context(r.negative_prompt);
            mx::eval(*negative);
        }
        if (impl_->verbose)
            logging::mlx().info("MLX text encoding: {:.3f}s", elapsed(stage));
        const int h = r.height / 8, w = r.width / 8;
        DiscreteFlowDenoiser denoiser(3.0f);
        const auto sigmas = denoiser.get_sigmas(r.steps, 0, SMOOTHSTEP_SCHEDULER, VERSION_ANIMA);
        std::vector<array> sampled;
        stage = Clock::now();
        for (int i = 0; i < count; ++i) {
            auto rng = std::make_shared<PhiloxRNG>(static_cast<std::uint64_t>(seed + i));
            sd::Tensor<float> x({w, h, 16, 1}, rng->randn(w * h * 16));
            if (progress)
                progress({i, count, {0, r.steps, 0}}, userdata);
            denoise_cb_t predict = [&](const sd::Tensor<float> &input, float sigma, int step) {
                const auto tick = Clock::now();
                auto latent = array(input.data(), {1, 16, h, w});
                auto v = mx::astype(impl_->model.predict(latent, sigma, context), mx::float32);
                if (negative) {
                    auto u = mx::astype(impl_->model.predict(latent, sigma, *negative), mx::float32);
                    v = u + r.cfg_scale * (v - u);
                }
                auto d = latent - sigma * v;
                mx::eval(d);
                if (progress)
                    progress({i, count, {step, r.steps, float(elapsed(tick))}}, userdata);
                return sd::Tensor<float>({w, h, 16, 1},
                                         std::vector<float>(d.data<float>(), d.data<float>() + d.size()));
            };
            x = sample_er_sde(predict, std::move(x), sigmas, rng, true, 1.0f);
            sampled.push_back(array(x.data(), {1, 16, h, w}));
            mx::eval(sampled.back());
        }
        if (impl_->verbose)
            logging::mlx().info("MLX sampling: {:.3f}s", elapsed(stage));
        stage = Clock::now();
        BatchResult result;
        for (int i = 0; i < count; ++i) {
            auto pixels = r.vae_tiling ? impl_->model.decode_tiled(sampled[i]) : impl_->model.decode(sampled[i]);
            mx::eval(pixels);
            auto bytes = mx::astype(mx::clip(pixels * 255.0f, array(0.0f), array(255.0f)), mx::uint8);
            mx::eval(bytes);
            const auto *data = bytes.data<std::uint8_t>();
            result.images.push_back({r.width, r.height, 3, {data, data + bytes.size()}, seed + i, 0});
        }
        if (impl_->verbose)
            logging::mlx().info("MLX VAE decoding: {:.3f}s", elapsed(stage));
        result.generation_seconds = elapsed(start);
        for (auto &image : result.images)
            image.generation_seconds = result.generation_seconds / count;
        output = std::move(result);
        return {};
    } catch (...) {
        return exception_status();
    }
}
} // namespace pictor::detail
