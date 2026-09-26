#include "cli.hpp"
#include "pictor/flux_klein.hpp"
#include "rng_philox.hpp"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(pictor::Status status) {
    if (!status) { std::fprintf(stderr, "%s\n", status.message); std::exit(1); }
}

// Deliberately measures sequential calls in a resident session, not native batches.
// Backend verbose logs supply stage timings; BENCH markers delimit each invocation.
int main(int argc, char** argv) {
    pictor::cli::Options options;
    check(pictor::cli::parse({argv + 1, argv + argc}, options));
    if (options.help) { std::puts(pictor::cli::usage()); return 0; }
    if (options.model != pictor::cli::Model::flux_klein || options.hidden_state_compression || options.request.seed < 0) {
        std::fputs("benchmark requires flux-klein, a fixed seed and HS disabled\n", stderr); return 1;
    }
    int warmup = 1;
    if (const auto value = std::getenv("PICTOR_BENCH_WARMUP")) {
        if (std::strcmp(value, "0") == 0) warmup = 0;
        else if (std::strcmp(value, "1") != 0) { std::fputs("PICTOR_BENCH_WARMUP must be 0 or 1\n", stderr); return 1; }
    }
    for (int i = 0; i < options.count; ++i) {
        std::filesystem::path path;
        check(pictor::cli::output_path(options, i, path));
        std::error_code ec;
        if (std::filesystem::exists(path, ec) && !options.overwrite) {
            std::fprintf(stderr, "output exists: %s\n", path.c_str()); return 1;
        }
        if (ec) { std::fprintf(stderr, "%s\n", ec.message().c_str()); return 1; }
    }
    std::vector<pictor::Image> refs(options.reference_images.size());
    pictor::FluxKleinEditRequest edit;
    edit.generation = options.request;
    edit.auto_resize = options.auto_resize_reference;
    for (std::size_t i = 0; i < refs.size(); ++i) {
        check(pictor::read_image(options.reference_images[i], refs[i]));
        edit.reference_images.push_back({refs[i].width, refs[i].height, refs[i].pixels.data(), refs[i].pixels.size()});
    }
    if (!refs.empty()) check(pictor::validate_flux_klein_edit_request(edit));
    // Export the backend's initial noise and actual schedule for cross-runtime
    // experiments. GGML storage [W,H,C,N] is contiguous NCHW in this file.
    const auto dir = options.output.parent_path();
    if (refs.empty()) {
        PhiloxRNG rng(static_cast<uint64_t>(options.request.seed));
        const auto noise = rng.randn(128 * (options.request.width / 16) * (options.request.height / 16));
        auto file = std::fopen((dir / "noise.f32").c_str(), "wb");
        if (!file) { std::perror("noise.f32"); return 1; }
        const auto written = std::fwrite(noise.data(), sizeof(float), noise.size(), file);
        const auto closed = std::fclose(file);
        if (written != noise.size() || closed) { std::fputs("noise write failed\n", stderr); return 1; }
        // Mirror pinned Flux2FlowDenoiser + DiscreteScheduler, with float32
        // arithmetic. Keep experimental schedule export out of the public API.
        const int seq = (options.request.width / 16) * (options.request.height / 16);
        const int steps = options.request.steps;
        const float m200 = 0.00016927f * seq + 0.45666666f;
        const float m10 = 8.73809524e-05f * seq + 1.89833333f;
        const float a = (m200 - m10) / 190.0f;
        const float mu = seq > 4300 ? m200 : a * steps + (m200 - 200.0f * a);
        std::vector<float> sigmas;
        for (int i = 0; i < steps; ++i) {
            const float t = (999.0f - (steps == 1 ? 0.0f : 999.0f / (steps - 1) * i) + 1.0f) / 1000.0f;
            sigmas.push_back(std::exp(mu) / (std::exp(mu) + (1.0f / t - 1.0f)));
        }
        sigmas.push_back(0);
        file = std::fopen((dir / "sigmas.json").c_str(), "w");
        if (!file) { std::perror("sigmas.json"); return 1; }
        std::fputs("[", file);
        for (std::size_t i = 0; i < sigmas.size(); ++i) std::fprintf(file, "%s%.9g", i ? "," : "", sigmas[i]);
        std::fputs("]\n", file);
        if (std::fclose(file)) { std::perror("sigmas.json"); return 1; }
    }
    std::unique_ptr<pictor::FluxKleinSession> session;
    check(pictor::FluxKleinSession::create({options.session.model_path, options.text_encoder, options.vae,
                                         options.session.threads, true}, options.backend, session));
    std::fprintf(stderr, "BENCH_LOAD %.6f\n", session->load_seconds());
    pictor::Image previous;
    for (int i = -warmup; i < options.count; ++i) {
        const auto unix_time = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::fprintf(stderr, "BENCH_BEGIN %d %.6f\n", i, unix_time); std::fflush(stderr);
        pictor::Image image;
        if (refs.empty()) check(session->generate(options.request, image));
        else check(session->edit(edit, image));
        std::fprintf(stderr, "BENCH_END %d %.6f\n", i, image.generation_seconds); std::fflush(stderr);
        if (!previous.pixels.empty() && image.pixels != previous.pixels) {
            std::fputs("identical resident calls produced different pixels\n", stderr); return 1;
        }
        if (i >= 0) {
            std::filesystem::path path;
            check(pictor::cli::output_path(options, i, path));
            check(pictor::write_png(path, image));
        }
        previous = std::move(image);
    }
}
