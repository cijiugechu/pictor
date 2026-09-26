// Native MLX boundary: translate runtime exceptions to the public Status API.
#include "inference_lock.hpp"
#include "logging.hpp"
#include "mlx_session.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "text_encoder.hpp"
#include "transformer.hpp"
#include "vae.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>

namespace pictor::detail {
using namespace mlx_backend;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct RuntimeScope {
    mx::Device previous = mx::default_device();
    RuntimeScope() { mx::set_default_device(mx::Device(mx::Device::gpu)); }
    ~RuntimeScope() { mx::set_default_device(previous); }
};
struct TokenizerLogScope {
    TokenizerLogScope() {
        sd_set_log_callback(
            [](sd_log_level_t level, const char *text, void *) {
                if (level >= SD_LOG_WARN && text)
                    logging::mlx().warn("{}", text);
            },
            nullptr);
    }
    ~TokenizerLogScope() { sd_set_log_callback(nullptr, nullptr); }
};
Status exception_status() noexcept {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return failure(ErrorCode::out_of_memory, "MLX allocation failed");
    } catch (const std::filesystem::filesystem_error &e) {
        return failure(ErrorCode::io_error, e.what());
    } catch (const std::invalid_argument &e) {
        return failure(ErrorCode::invalid_argument, e.what());
    } catch (const std::exception &e) {
        return failure(ErrorCode::backend_error, e.what());
    } catch (...) {
        return failure(ErrorCode::backend_error, "unknown MLX runtime failure");
    }
}
array reference_pixels(const ImageView &ref, const GenerationRequest &r, bool resize) {
    int w = ref.width, h = ref.height;
    if (resize) {
        const auto area = std::min(1024 * 1024, r.width * r.height);
        const double width = std::sqrt(double(area) * ref.width / ref.height);
        w = int(std::round(width / 16)) * 16;
        h = int(std::round(width * ref.height / ref.width / 16)) * 16;
    }
    std::vector<float> values(std::size_t(w) * h * 3);
    // Match pictor's existing sd.cpp nearest-neighbour reference resizing.
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                const int sy = int(double(y) * ref.height / h), sx = int(double(x) * ref.width / w);
                values[(std::size_t(y) * w + x) * 3 + c] =
                    float(ref.pixels[(std::size_t(sy) * ref.width + sx) * 3 + c]) / 255.0f * 2.0f - 1.0f;
            }
    return mx::astype(array(values.data(), {1, h, w, 3}), mx::bfloat16);
}
array pack(const array &nchw) {
    return mx::transpose(mx::reshape(nchw, {1, 128, nchw.shape(2) * nchw.shape(3)}), {0, 2, 1});
}
} // namespace
struct MlxSession::Impl {
    TextEncoder text;
    Transformer transformer;
    Vae vae;
    bool verbose;
    double seconds = 0;
    explicit Impl(const FluxKleinOptions &o)
        : text(o.text_encoder_path), transformer(o.diffusion_model_path), vae(o.vae_path), verbose(o.verbose) {}
};
MlxSession::MlxSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
}
MlxSession::~MlxSession() {
    std::lock_guard<std::mutex> lock(inference_mutex());
    impl_.reset();
}
Status MlxSession::create(const FluxKleinOptions &options, std::unique_ptr<MlxSession> &output) noexcept {
    if (output)
        return failure(ErrorCode::invalid_argument, "session output must be empty");
    if (options.threads < 0)
        return failure(ErrorCode::invalid_argument, "threads must be nonnegative");
    if (options.diffusion_model_path.empty() || options.text_encoder_path.empty() || options.vae_path.empty())
        return failure(ErrorCode::invalid_argument, "MLX model, text encoder and VAE paths must not be empty");
    try {
        for (const auto &path : {options.diffusion_model_path, options.text_encoder_path, options.vae_path})
            if (!std::filesystem::exists(path))
                return failure(ErrorCode::invalid_argument, "MLX weights not found: " + path.string());
        std::lock_guard<std::mutex> lock(inference_mutex());
        RuntimeScope runtime;
        TokenizerLogScope log;
        const auto start = Clock::now();
        auto impl = std::make_unique<Impl>(options);
        impl->seconds = elapsed(start);
        output.reset(new MlxSession(std::move(impl)));
        return {};
    } catch (...) {
        return exception_status();
    }
}
double MlxSession::load_seconds() const noexcept {
    return impl_->seconds;
}
Status MlxSession::generate_batch(const GenerationRequest &r, int count, BatchResult &output,
                                  BatchProgressCallback progress, void *userdata,
                                  const FluxKleinEditRequest *edit) noexcept {
    output = {};
    if (auto status = validate_batch_request(r, count); !status)
        return status;
    if (auto status = validate_flux_klein_request(r); !status)
        return status;
    if (edit)
        if (auto status = validate_flux_klein_edit_request(*edit); !status)
            return status;
    std::int64_t seed;
    if (auto status = resolve_seed(r.seed, seed); !status)
        return status;
    try {
        std::lock_guard<std::mutex> lock(inference_mutex());
        RuntimeScope runtime;
        TokenizerLogScope log;
        const auto start = Clock::now();
        auto stage = start;
        const auto text = impl_->text.encode(r.prompt);
        std::optional<array> negative;
        if (r.cfg_scale != 1.0f)
            negative = impl_->text.encode(r.negative_prompt);
        mx::eval(text);
        if (negative)
            mx::eval(*negative);
        if (impl_->verbose)
            logging::mlx().info("MLX text encoding: {:.3f}s", elapsed(stage));
        const int h = r.height / 16, w = r.width / 16, n = h * w;
        std::vector<array> refs, ids{grid_ids(h, w, 0)};
        if (edit) {
            stage = Clock::now();
            for (std::size_t i = 0; i < edit->reference_images.size(); ++i) {
                const auto encoded =
                    impl_->vae.encode(reference_pixels(edit->reference_images[i], r, edit->auto_resize));
                refs.push_back(pack(encoded));
                ids.push_back(grid_ids(encoded.shape(2), encoded.shape(3), int(i + 1) * 10));
            }
            mx::eval(refs);
            if (impl_->verbose)
                logging::mlx().info("MLX reference encoding: {:.3f}s", elapsed(stage));
        }
        const auto image_ids = mx::concatenate(ids, 0);
        const auto sigmas = discrete_sigmas(r.width, r.height, r.steps);
        // MFLUX compiles the Euler update; dt is explicitly cast to latent precision.
        static auto step =
            mx::compile([](const std::vector<array> &a) { return std::vector<array>{a[0] + a[1] * a[2]}; });
        std::vector<array> sampled;
        stage = Clock::now();
        for (int i = 0; i < count; ++i) {
            PhiloxRNG rng(static_cast<std::uint64_t>(seed + i));
            const auto noise = rng.randn(128 * h * w);
            auto latents = pack(mx::astype(array(noise.data(), {1, 128, h, w}), mx::bfloat16));
            if (progress)
                progress({i, count, {0, r.steps, 0}}, userdata);
            for (int s = 0; s < r.steps; ++s) {
                const auto tick = Clock::now();
                auto all = refs;
                all.insert(all.begin(), latents);
                const auto input = mx::concatenate(all, 1);
                const array t(sigmas[s] * 1000.0f);
                auto prediction = part(impl_->transformer.predict(input, text, t, image_ids), 1, 0, n);
                if (negative) {
                    const auto uncond = part(impl_->transformer.predict(input, *negative, t, image_ids), 1, 0, n);
                    prediction = uncond + array(r.cfg_scale, uncond.dtype()) * (prediction - uncond);
                }
                latents = step({latents, prediction, array(sigmas[s + 1] - sigmas[s], mx::bfloat16)})[0];
                mx::eval(latents);
                if (progress)
                    progress({i, count, {s + 1, r.steps, float(elapsed(tick))}}, userdata);
            }
            sampled.push_back(mx::reshape(mx::transpose(latents, {0, 2, 1}), {1, 128, h, w}));
        }
        if (impl_->verbose)
            logging::mlx().info("MLX sampling: {:.3f}s", elapsed(stage));
        stage = Clock::now();
        BatchResult result;
        for (int i = 0; i < count; ++i) {
            const auto decoded = impl_->vae.decode(sampled[i], r.vae_tiling);
            const auto normalized = mx::clip(decoded / array(2.0f, decoded.dtype()) + array(0.5f, decoded.dtype()),
                                             array(0.0f, decoded.dtype()), array(1.0f, decoded.dtype()));
            const auto pixels = mx::astype(
                mx::round(mx::astype(mx::transpose(normalized, {0, 2, 3, 1}), mx::float32) * 255.0f), mx::uint8);
            mx::eval(pixels);
            const auto *data = pixels.data<std::uint8_t>();
            result.images.push_back({r.width, r.height, 3, {data, data + pixels.size()}, seed + i, 0});
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
