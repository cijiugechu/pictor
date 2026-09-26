#include "pictor/flux_klein.hpp"
#include "pictor/pictor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
using namespace pictor;
static void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static void check(Status s) {
    require(s.ok(), s.message);
}
static void check_c(pictor_status s, const pictor_error &e) {
    require(s == PICTOR_OK, e.message);
}
int main() {
    const FluxKleinOptions options{"models/mlx-flux2-klein-4b-4bit/transformer",
                                   "models/mlx-flux2-klein-4b-4bit/text_encoder",
                                   "models/flux2-klein-4b/full_encoder_small_decoder.safetensors", 0, true};
    std::unique_ptr<FluxKleinSession> session;
    check(FluxKleinSession::create(options, session));
    require(session->backend() == KleinBackend::mlx, "automatic directory selects MLX");
    require(!session->set_hidden_state_compression(true), "unsupported HS returns error");
    check(session->set_hidden_state_compression(false));
    auto request = flux_klein_request();
    request.prompt = "一只红狐狸，在雪地中。 café ✨";
    request.width = 80;
    request.height = 64;
    request.steps = 1;
    request.seed = 666;
    BatchResult batch;
    struct Events {
        int total = 0;
        int last = -1;
    } events;
    const auto progress = [](const BatchProgress &p, void *ptr) noexcept {
        auto &e = *static_cast<Events *>(ptr);
        require(p.image_count == 2 && p.image_index >= e.last && p.image_index < 2, "batch progress order");
        require(p.sampling.step >= 0 && p.sampling.step <= p.sampling.steps, "progress bounds");
        ++e.total;
        e.last = p.image_index;
    };
    check(session->generate_batch(request, 2, batch, progress, &events));
    require(events.total == 4 && batch.images.size() == 2 && batch.images[1].seed == 667, "batch result and callbacks");
    Image single;
    check(session->generate(request, single));
    require(single.pixels == batch.images[0].pixels, "first seed batch/single exact");
    request.seed++;
    check(session->generate(request, single));
    require(single.pixels == batch.images[1].pixels, "second seed batch/single exact");
    check(write_png("outputs/mlx-port/native-smoke.png", single));
    FluxKleinEditRequest edit;
    edit.generation = request;
    edit.generation.prompt = "Make the fox blue";
    edit.auto_resize = false;
    edit.reference_images = {{single.width, single.height, single.pixels.data(), single.pixels.size()}};
    BatchResult edits;
    check(session->edit_batch(edit, 2, edits));
    Image edited;
    check(session->edit(edit, edited));
    require(edited.pixels == edits.images[0].pixels, "edit batch/single exact");
    check(write_png("outputs/mlx-port/native-edit-smoke.png", edited));
    // The reference borrows the output that edit replaces.
    check(session->edit(edit, single));
    require(single.pixels == edited.pixels, "reference may borrow output");
    edit.reference_images = {{edited.width, edited.height, edited.pixels.data(), edited.pixels.size()},
                             {80, 64, batch.images[0].pixels.data(), batch.images[0].pixels.size()}};
    edit.auto_resize = true;
    edit.generation.width = 64;
    edit.generation.height = 80;
    check(session->edit(edit, single));
    require(single.width == 64 && single.height == 80, "two references and auto resize");
    auto tiled = request;
    tiled.vae_tiling = true;
    check(session->generate(tiled, single));
    require(single.pixels == batch.images[1].pixels, "small tiled decode preserves output");
    tiled.width = 272;
    check(session->generate(tiled, single));
    require(single.width == 272 && single.pixels.size() == 272 * 64 * 3, "overlapping edge tiles");
    auto guided = request;
    guided.negative_prompt = "blurry";
    guided.cfg_scale = 2;
    check(session->generate(guided, single));
    require(single.pixels != batch.images[1].pixels, "negative CFG pass used");
    auto invalid = request;
    invalid.seed = std::numeric_limits<std::int64_t>::max();
    require(!session->generate_batch(invalid, 2, batch) && batch.images.empty(), "failed batch clears output");
    invalid = request;
    invalid.prompt = std::string(1, char(0xff));
    require(!session->generate(invalid, single) && single.pixels.empty(), "invalid UTF-8 catches exception");
    check(session->generate(request, single));
    session.reset();
    require(!single.pixels.empty(), "pixels outlive session");
    pictor_error error{};
    pictor_flux_klein_options c_options{};
    check_c(pictor_flux_klein_options_init(&c_options, sizeof(c_options), &error), error);
    const auto dit = options.diffusion_model_path.string(), text = options.text_encoder_path.string(),
               vae = options.vae_path.string();
    c_options.diffusion_model_path = dit.c_str();
    c_options.text_encoder_path = text.c_str();
    c_options.vae_path = vae.c_str();
    pictor_session *c_session = nullptr;
    check_c(pictor_flux_klein_session_create_with_backend(&c_options, PICTOR_KLEIN_BACKEND_MLX, &c_session, &error),
            error);
    pictor_klein_backend backend;
    check_c(pictor_flux_klein_session_backend(c_session, &backend, &error), error);
    require(backend == PICTOR_KLEIN_BACKEND_MLX, "C selected backend");
    pictor_request c_request{};
    check_c(pictor_flux_klein_request_init(&c_request, sizeof(c_request), &error), error);
    c_request.prompt = request.prompt.c_str();
    c_request.width = request.width;
    c_request.height = request.height;
    c_request.steps = request.steps;
    c_request.seed = request.seed;
    pictor_image *image = nullptr;
    check_c(pictor_session_generate(c_session, &c_request, nullptr, nullptr, &image, &error), error);
    pictor_image_info info{};
    check_c(pictor_image_get_info(image, &info, sizeof(info), &error), error);
    require(info.pixels_len == single.pixels.size() &&
                std::memcmp(info.pixels, single.pixels.data(), info.pixels_len) == 0,
            "C/C++ RGB exact");
    pictor_session_destroy(c_session);
    check_c(pictor_image_write_png(image, "outputs/mlx-port/native-c-smoke.png", &error), error);
    pictor_image_destroy(image);
    std::puts("PASS: native MLX C++/C, batch/reference parity, Unicode, ownership, errors and callbacks");
}
