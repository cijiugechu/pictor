#include "anima_mlx/model.hpp"
#include "cli.hpp"
#include "denoiser.hpp"
#include "inference_lock.hpp"
#include "rng_philox.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
using namespace pictor::anima_mlx;
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void check(pictor::Status s) {
    if (!s)
        throw std::runtime_error(s.message);
}
int main(int argc, char **argv) try {
    pictor::cli::Options o;
    if (!std::getenv("PICTOR_ANIMA_FIXTURE")) {
        check(pictor::cli::parse({argv + 1, argv + argc}, o));
        if (o.help) {
            std::puts("Experimental Anima MLX: same Anima arguments; --count repeats the same seed. Set "
                      "PICTOR_ANIMA_WEIGHTS; --cache none required.");
            return 0;
        }
        if (o.model != pictor::cli::Model::anima || o.request.vae_tiling)
            throw std::invalid_argument("Anima text generation without tiling required");
        if (o.request.cache != pictor::CacheMode::none)
            throw std::invalid_argument("experimental Anima MLX benchmark requires --cache none");
        if (o.request.seed < 0)
            throw std::invalid_argument("fixed seed required");
        for (int i = 0; i < o.count; ++i) {
            std::filesystem::path path;
            check(pictor::cli::output_path(o, i, path));
            if (std::filesystem::exists(path) && !o.overwrite)
                throw std::invalid_argument("output exists");
        }
    }
    const char *weights = std::getenv("PICTOR_ANIMA_WEIGHTS");
    if (!weights)
        throw std::invalid_argument("set PICTOR_ANIMA_WEIGHTS to the converted model directory");
    std::lock_guard<std::mutex> lock(pictor::detail::inference_mutex());
    const bool int4 = std::getenv("PICTOR_ANIMA_INT4"), bf16 = std::getenv("PICTOR_ANIMA_BF16");
    auto start = now();
    Model model(weights, int4, bf16);
    std::fprintf(stderr, "BENCH_LOAD %.6f\n", now() - start);
    if (const auto fixture = std::getenv("PICTOR_ANIMA_FIXTURE")) {
        auto p = std::filesystem::path(fixture);
        auto inputs = mx::load_safetensors((p / "inputs.safetensors").string()).first;
        auto qi = inputs.at("qwen_ids"), ti = inputs.at("t5_ids");
        mx::eval(qi, ti);
        std::vector<int> q(qi.data<int>(), qi.data<int>() + qi.size()), t(ti.data<int>(), ti.data<int>() + ti.size());
        auto text = model.text(q);
        mx::eval(text);
        auto context = model.adapt(text, t, {});
        mx::eval(context);
        auto noise = model.predict(inputs.at("latent"), 0.5f, inputs.at("context"));
        mx::eval(noise);
        auto decoded = model.decode(inputs.at("latent"));
        mx::eval(decoded);
        mx::save_safetensors((p / "native.safetensors").string(),
                             {{"text", text}, {"context", context}, {"noise", noise}, {"decoded", decoded}});
        return 0;
    }
    int warm = 1;
    if (auto env = std::getenv("PICTOR_BENCH_WARMUP")) {
        if (std::strcmp(env, "0") == 0)
            warm = 0;
        else if (std::strcmp(env, "1") != 0)
            throw std::invalid_argument("PICTOR_BENCH_WARMUP must be 0 or 1");
    }
    pictor::Image previous;
    for (int i = -warm; i < o.count; ++i) {
        std::fprintf(stderr, "BENCH_BEGIN %d\n", i);
        std::fflush(stderr);
        start = now();
        auto text_start = now();
        auto ctx = model.context(o.request.prompt);
        mx::eval(ctx);
        std::optional<array> unc;
        if (o.request.cfg_scale != 1) {
            unc = model.context(o.request.negative_prompt);
            mx::eval(*unc);
        }
        std::fprintf(stderr, "MLX text encoding: %.6fs\n", now() - text_start);
        auto sampling = now();
        auto rng = std::make_shared<PhiloxRNG>(uint64_t(o.request.seed));
        const int h = o.request.height / 8, w = o.request.width / 8;
        sd::Tensor<float> x({w, h, 16, 1}, rng->randn(w * h * 16));
        DiscreteFlowDenoiser denoiser(3.0f);
        auto sigmas = denoiser.get_sigmas(o.request.steps, 0, SMOOTHSTEP_SCHEDULER, VERSION_ANIMA);
        denoise_cb_t predict = [&](const sd::Tensor<float> &input, float sigma, int step) {
            auto latent = array(input.data(), {1, 16, h, w});
            auto v = mx::astype(model.predict(latent, sigma, ctx), mx::float32);
            if (unc) {
                auto u = mx::astype(model.predict(latent, sigma, *unc), mx::float32);
                v = u + o.request.cfg_scale * (v - u);
            }
            auto d = latent - sigma * v;
            mx::eval(d);
            std::fprintf(stderr, "MLX step %d/%d\n", step, o.request.steps);
            std::fflush(stderr);
            return sd::Tensor<float>({w, h, 16, 1}, std::vector<float>(d.data<float>(), d.data<float>() + d.size()));
        };
        x = sample_er_sde(predict, std::move(x), sigmas, rng, true, 1.0f);
        std::fprintf(stderr, "MLX sampling: %.6fs\n", now() - sampling);
        auto decode = now();
        auto pixels = model.decode(array(x.data(), {1, 16, h, w}));
        mx::eval(pixels);
        auto bytes = mx::astype(mx::clip(pixels * 255.0f, array(0.0f), array(255.0f)), mx::uint8);
        mx::eval(bytes);
        std::fprintf(stderr, "MLX VAE decoding: %.6fs\n", now() - decode);
        pictor::Image image;
        image.channels = 3;
        image.width = o.request.width;
        image.height = o.request.height;
        image.seed = o.request.seed;
        image.pixels.assign(bytes.data<uint8_t>(), bytes.data<uint8_t>() + bytes.size());
        image.generation_seconds = now() - start;
        std::fprintf(stderr, "BENCH_END %d %.6f\n", i, image.generation_seconds);
        std::fflush(stderr);
        if (!previous.pixels.empty() && previous.pixels != image.pixels)
            throw std::runtime_error("resident RGB mismatch");
        if (i >= 0) {
            std::filesystem::path path;
            check(pictor::cli::output_path(o, i, path));
            check(pictor::write_png(path, image));
        }
        std::fprintf(stderr, "BENCH_MLX_PEAK %zu\n", mx::get_peak_memory());
        previous = std::move(image);
    }
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "Anima MLX: %s\n", e.what());
    return 1;
}
