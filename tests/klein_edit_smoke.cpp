#include "pictor/flux_klein.hpp"
#include "pictor/pictor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
void check(pictor::Status status) { require(status.ok(), status.message); }
void check_c(pictor_status status, const pictor_error& error) { require(status == PICTOR_OK, error.message); }
void progress(int32_t step, int32_t steps, float, void* data) noexcept {
    *static_cast<int*>(data) = step;
    std::fprintf(stderr, "Edit step %d/%d\n", step, steps);
}
}

int main(int argc, char** argv) {
    const std::filesystem::path directory = argc > 1 ? argv[1] : "models/flux2-klein-4b";
    const char* reference_path = argc > 2 ? argv[2] : "outputs/klein-reference.png";
    const std::filesystem::path output = argc > 3 ? argv[3] : "outputs/klein-edit-smoke";
    const auto diffusion = (directory / "flux-2-klein-4b-Q4_0.gguf").string();
    const auto encoder = (directory / "Qwen3-4B-Q4_K_M.gguf").string();
    const auto vae = (directory / "flux2-vae.safetensors").string();
    pictor::Image reference;
    check(pictor::read_image(reference_path, reference));
    const auto original = reference.pixels;
    pictor::FluxKleinEditRequest request;
    request.generation.prompt = "Change the fox's fur to white and the forest to a snowy winter scene. Keep the fox's pose and composition.";
    request.generation.seed = 666;
    request.reference_images = {{reference.width, reference.height, reference.pixels.data(), reference.pixels.size()}};
    std::unique_ptr<pictor::FluxKleinSession> session;
    check(pictor::FluxKleinSession::create({diffusion, encoder, vae}, session));
    pictor::Image edited, text;
    int last = -1;
    check(session->edit(request, edited, [](const pictor::Progress& p, void* data) noexcept {
        progress(p.step, p.steps, p.seconds, data);
    }, &last));
    require(last == 4 && edited.pixels != original && reference.pixels == original, "reference edit/callback/immutability failed");
    auto generation = pictor::flux_klein_request();
    generation.prompt = "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light";
    generation.seed = 666;
    check(session->generate(generation, text));
    session.reset();
    check(pictor::write_png(output / "cpp.png", edited));
    check(pictor::write_png(output / "text.png", text));

    pictor_error error{};
    pictor_flux_klein_options options;
    check_c(pictor_flux_klein_options_init(&options, sizeof(options), &error), error);
    options.diffusion_model_path = diffusion.c_str(); options.text_encoder_path = encoder.c_str(); options.vae_path = vae.c_str();
    pictor_session* handle = nullptr;
    check_c(pictor_flux_klein_session_create(&options, &handle, &error), error);
    auto owner = std::unique_ptr<pictor_session, decltype(&pictor_session_destroy)>(handle, pictor_session_destroy);
    pictor_image* source = nullptr;
    check_c(pictor_image_load(reference_path, &source, &error), error);
    auto source_owner = std::unique_ptr<pictor_image, decltype(&pictor_image_destroy)>(source, pictor_image_destroy);
    pictor_image_info info;
    check_c(pictor_image_get_info(source, &info, sizeof(info), &error), error);
    pictor_image_view view{sizeof(view), info.width, info.height, info.pixels, info.pixels_len};
    pictor_request c_request;
    check_c(pictor_flux_klein_request_init(&c_request, sizeof(c_request), &error), error);
    c_request.prompt = request.generation.prompt.c_str(); c_request.seed = 666;
    pictor_flux_klein_edit_options edit;
    check_c(pictor_flux_klein_edit_options_init(&edit, sizeof(edit), &error), error);
    edit.reference_images = &view; edit.reference_images_count = 1;
    pictor_image* image = nullptr;
    view.pixels_len--;
    require(pictor_flux_klein_session_edit(handle, &c_request, &edit, nullptr, nullptr, &image, &error) == PICTOR_INVALID_ARGUMENT && !image,
            "short reference buffer accepted");
    view.pixels_len++; last = -1;
    check_c(pictor_flux_klein_session_edit(handle, &c_request, &edit, progress, &last, &image, &error), error);
    auto image_owner = std::unique_ptr<pictor_image, decltype(&pictor_image_destroy)>(image, pictor_image_destroy);
    require(last == 4, "C edit progress failed");
    owner.reset(); source_owner.reset();
    check_c(pictor_image_get_info(image, &info, sizeof(info), &error), error);
    require(info.pixels_len == edited.pixels.size() && !std::memcmp(info.pixels, edited.pixels.data(), info.pixels_len), "C++/C edit parity failed");
    check_c(pictor_image_write_png(image, (output / "c.png").string().c_str(), &error), error);
    std::printf("PASS: Klein editing C++/C parity, edit-to-text reuse, input immutability, callbacks, recovery and ownership; cpp=%.3fs text=%.3fs c=%.3fs\n",
                edited.generation_seconds, text.generation_seconds, info.generation_seconds);
}
