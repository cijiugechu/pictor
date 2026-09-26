#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"
#include "pictor/pictor.h"
#include <atomic>
#include <cstdio>
#include <thread>

extern "C" int pictor_test_backend_overlaps();
#define CHECK(expression) do { if (!(expression)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); return 1; } } while (0)
struct State { int expected; int calls = 0; bool failed = false; };
void progress(const pictor::Progress& value, void* userdata) noexcept {
    auto& state = *static_cast<State*>(userdata);
    state.calls++;
    if (value.steps != state.expected || value.step != value.steps) state.failed = true;
}
int main() {
    std::unique_ptr<pictor::AnimaSession> anima;
    std::unique_ptr<pictor::FluxKleinSession> klein;
    CHECK(pictor::AnimaSession::create({__FILE__}, anima));
    CHECK(pictor::FluxKleinSession::create({__FILE__, __FILE__, __FILE__}, klein));
    pictor::Image first, second;
    State a{3}, b{4};
    std::atomic<bool> start{false};
    auto run = [&](auto& session, auto request, auto& image, auto& state, int pixel) {
        while (!start.load()) std::this_thread::yield();
        request.prompt = "test";
        request.seed = 666;
        for (int i = 0; i < 5; ++i) {
            const auto status = session->generate(request, image, progress, &state);
            if (!status || image.pixels.empty() || image.pixels[0] != pixel) state.failed = true;
        }
    };
    std::thread left([&] { run(anima, pictor::preset_request(pictor::Preset::fast), first, a, 11); });
    std::thread right([&] { run(klein, pictor::flux_klein_request(), second, b, 22); });
    start = true;
    left.join(); right.join();
    CHECK(a.calls == 5 && b.calls == 5 && !a.failed && !b.failed && pictor_test_backend_overlaps() == 0);
    auto invalid = pictor::flux_klein_request();
    invalid.prompt = "test";
    invalid.cache = pictor::CacheMode::spectrum;
    CHECK(!klein->generate(invalid, second) && second.pixels.empty());
    auto valid = pictor::flux_klein_request(); valid.prompt = "test";
    CHECK(klein->generate(valid, second));
    std::vector<std::uint8_t> rgb(16 * 16 * 3, 40), other(16 * 16 * 3, 3);
    pictor::FluxKleinEditRequest edit;
    edit.generation.prompt = "make it winter";
    edit.reference_images = {{16, 16, rgb.data(), rgb.size()}, {16, 16, other.data(), other.size()}};
    CHECK(klein->edit(edit, second) && second.pixels[0] == 68);
    CHECK(rgb[0] == 40 && other[0] == 3);
    // An output may also own the memory viewed by a reference.
    edit.reference_images = {{second.width, second.height, second.pixels.data(), second.pixels.size()}};
    CHECK(klein->edit(edit, second) && second.pixels[0] == 90);
    CHECK(klein->generate(valid, second) && second.pixels[0] == 22);
    pictor_error error{};
    pictor_flux_klein_options c_options;
    CHECK(pictor_flux_klein_options_init(&c_options, sizeof(c_options), &error) == PICTOR_OK);
    c_options.diffusion_model_path = c_options.text_encoder_path = c_options.vae_path = __FILE__;
    pictor_session* c_session = nullptr;
    CHECK(pictor_flux_klein_session_create(&c_options, &c_session, &error) == PICTOR_OK);
    pictor_flux_klein_edit_options c_edit;
    CHECK(pictor_flux_klein_edit_options_init(&c_edit, sizeof(c_edit), &error) == PICTOR_OK);
    pictor_request c_request;
    CHECK(pictor_flux_klein_request_init(&c_request, sizeof(c_request), &error) == PICTOR_OK);
    c_request.prompt = "winter";
    pictor_image_view refs[] = {{sizeof(pictor_image_view), 16, 16, rgb.data(), rgb.size()},
                               {sizeof(pictor_image_view), 16, 16, other.data(), other.size()}};
    c_edit.reference_images = refs; c_edit.reference_images_count = 2;
    pictor_image* c_image = nullptr;
    c_edit.auto_resize = 2;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT && !c_image);
    c_edit.auto_resize = 1; refs[0].pixels_len--;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT && !c_image);
    refs[0].pixels_len++; refs[0].struct_size--;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT);
    refs[0].struct_size++;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_OK);
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT);
    pictor_session_destroy(c_session);
    pictor_image_info info;
    CHECK(pictor_image_get_info(c_image, &info, sizeof(info), &error) == PICTOR_OK && info.pixels[0] == 68);
    pictor_image_destroy(c_image);
    anima.reset(); klein.reset();
    CHECK(first.pixels[0] == 11 && second.pixels[0] == 22);
    std::puts("PASS: shared lock, callback isolation, reference routing/ownership, C edit errors, session reuse");
}
