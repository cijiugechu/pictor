#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"
#include "pictor/pictor.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void check(pictor::Status status) {
    if (!status) { std::fprintf(stderr, "%s\n", status.message); std::exit(1); }
}
void check_c(pictor_status status, const pictor_error& error) {
    if (status != PICTOR_OK) { std::fprintf(stderr, "%s\n", error.message); std::exit(1); }
}
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
struct Progress { int calls = 0; int last = -1; bool valid = true; };
void on_progress(int32_t step, int32_t steps, float, void* userdata) noexcept {
    auto& state = *static_cast<Progress*>(userdata);
    state.calls++; state.last = step;
    state.valid = state.valid && steps == 4 && step >= 0 && step <= steps;
    std::fprintf(stderr, "Klein step %d/%d\n", step, steps);
}
}

int main(int argc, char** argv) {
    const std::filesystem::path directory = argc > 1 ? argv[1] : "models/flux2-klein-4b";
    const std::filesystem::path output = argc > 2 ? argv[2] : "outputs/klein-smoke";
    const auto diffusion = (directory / "flux-2-klein-4b-Q4_0.gguf").string();
    const auto encoder = (directory / "Qwen3-4B-Q4_K_M.gguf").string();
    const auto vae = (directory / "flux2-vae.safetensors").string();
    std::unique_ptr<pictor::FluxKleinSession> session;
    check(pictor::FluxKleinSession::create({diffusion, encoder, vae}, session));
    const auto load = session->load_seconds();
    auto request = pictor::flux_klein_request();
    request.prompt = "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light";
    request.seed = 666;
    pictor::Image first, repeat;
    Progress state;
    check(session->generate(request, first, [](const pictor::Progress& p, void* data) noexcept {
        on_progress(p.step, p.steps, p.seconds, data);
    }, &state));
    require(state.valid && state.calls > 0 && state.last == 4, "C++ callback failed");
    check(session->generate(request, repeat));
    require(first.pixels == repeat.pixels && first.seed == repeat.seed, "resident repeat differs");
    const auto range = std::minmax_element(first.pixels.begin(), first.pixels.end());
    require(first.width == 512 && first.height == 512 && first.channels == 3 && *range.second - *range.first > 32, "unexpected image");
    session.reset();
    check(pictor::write_png(output / "cpp.png", first));

    pictor_error error{};
    pictor_flux_klein_options options;
    check_c(pictor_flux_klein_options_init(&options, sizeof(options), &error), error);
    options.diffusion_model_path = diffusion.c_str();
    options.text_encoder_path = encoder.c_str();
    options.vae_path = vae.c_str();
    pictor_session* c_session = nullptr;
    check_c(pictor_flux_klein_session_create(&options, &c_session, &error), error);
    auto owner = std::unique_ptr<pictor_session, decltype(&pictor_session_destroy)>(c_session, pictor_session_destroy);
    pictor_request c_request;
    check_c(pictor_flux_klein_request_init(&c_request, sizeof(c_request), &error), error);
    c_request.prompt = request.prompt.c_str(); c_request.seed = 666;
    pictor_image* c_image = nullptr;
    c_request.cache = PICTOR_CACHE_SPECTRUM;
    require(pictor_session_generate(c_session, &c_request, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT && !c_image,
            "Klein accepted unsupported cache");
    c_request.cache = PICTOR_CACHE_NONE;
    state = {};
    check_c(pictor_session_generate(c_session, &c_request, on_progress, &state, &c_image, &error), error);
    auto image_owner = std::unique_ptr<pictor_image, decltype(&pictor_image_destroy)>(c_image, pictor_image_destroy);
    require(state.valid && state.calls > 0 && state.last == 4, "C callback failed");
    owner.reset();
    pictor_image_info info;
    check_c(pictor_image_get_info(c_image, &info, sizeof(info), &error), error);
    require(info.pixels_len == first.pixels.size() && info.seed == first.seed &&
            std::memcmp(info.pixels, first.pixels.data(), info.pixels_len) == 0, "C ABI / C++ outputs differ");
    check_c(pictor_image_write_png(c_image, (output / "c.png").string().c_str(), &error), error);
    std::printf("PASS: Klein C++/C ABI parity, resident repeat, callbacks, cache rejection, image ownership; load=%.3fs first=%.3fs repeat=%.3fs c=%.3fs\n",
                load, first.generation_seconds, repeat.generation_seconds, info.generation_seconds);
}
