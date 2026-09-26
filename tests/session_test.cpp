#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"
#include "pictor/pictor.h"
#include <atomic>
#include <cstdio>
#include <thread>

extern "C" int pictor_test_backend_overlaps();
extern "C" int pictor_test_generation_calls();
#define CHECK(expression) do { if (!(expression)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); return 1; } } while (0)
struct State { int expected; int calls = 0; bool failed = false; };
void progress(const pictor::Progress& value, void* userdata) noexcept {
    auto& state = *static_cast<State*>(userdata);
    state.calls++;
    if (value.steps != state.expected || (value.step != 0 && value.step != value.steps)) state.failed = true;
}
struct BatchState { int calls = 0; bool failed = false; };
void batch_progress(const pictor::BatchProgress& value, void* data) noexcept {
    auto& state = *static_cast<BatchState*>(data);
    if (value.image_index != state.calls / 2 || value.image_count != 2 ||
        value.sampling.step != (state.calls % 2 ? value.sampling.steps : 0)) state.failed = true;
    ++state.calls;
}
void c_batch_progress(int32_t index, int32_t count, int32_t step, int32_t steps, float seconds, void* data) {
    batch_progress({index, count, {step, steps, seconds}}, data);
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
    CHECK(a.calls == 10 && b.calls == 10 && !a.failed && !b.failed && pictor_test_backend_overlaps() == 0);
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
    edit.reference_images = {{16, 16, rgb.data(), rgb.size()}, {16, 16, other.data(), other.size()}};
    CHECK(klein->set_hidden_state_compression(true));
    CHECK(klein->generate(valid, second) && second.pixels[0] == 122);
    pictor::BatchResult hs_batch;
    CHECK(klein->edit_batch(edit, 2, hs_batch) && hs_batch.images[1].pixels[0] == 168);
    CHECK(klein->set_hidden_state_compression(false));
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
    CHECK(pictor_flux_klein_session_set_hidden_state_compression(nullptr, 1, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_session_set_hidden_state_compression(c_session, 2, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_session_set_hidden_state_compression(c_session, 1, &error) == PICTOR_OK);
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_OK);
    pictor_image_info hs_info;
    CHECK(pictor_image_get_info(c_image, &hs_info, sizeof(hs_info), &error) == PICTOR_OK && hs_info.pixels[0] == 168);
    pictor_image_destroy(c_image); c_image = nullptr;
    CHECK(pictor_flux_klein_session_set_hidden_state_compression(c_session, 0, &error) == PICTOR_OK);
    c_edit.auto_resize = 2;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT && !c_image);
    c_edit.auto_resize = 1; refs[0].pixels_len--;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT && !c_image);
    refs[0].pixels_len++; refs[0].struct_size--;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT);
    refs[0].struct_size++;
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_OK);
    CHECK(pictor_flux_klein_session_edit(c_session, &c_request, &c_edit, nullptr, nullptr, &c_image, &error) == PICTOR_INVALID_ARGUMENT);
    // Batch calls must route once, preserve seed order, and filter VAE progress.
    for (int model = 0; model < 2; ++model) {
        auto request = model ? pictor::flux_klein_request() : pictor::preset_request(pictor::Preset::fast);
        request.prompt = "seed-colors"; request.seed = 42; request.width = request.height = 64;
        pictor::BatchResult batch;
        BatchState state;
        const int calls = pictor_test_generation_calls();
        const auto generated = model ? klein->generate_batch(request, 2, batch, batch_progress, &state)
                                     : anima->generate_batch(request, 2, batch, batch_progress, &state);
        CHECK(generated && pictor_test_generation_calls() == calls + 1);
        CHECK(batch.images.size() == 2 && state.calls == 4 && !state.failed);
        CHECK(batch.images[0].seed == 42 && batch.images[1].seed == 43);
        CHECK(batch.images[0].pixels[0] == (model ? 64 : 53) && batch.images[1].pixels[0] == (model ? 65 : 54));
        CHECK(batch.generation_seconds > 0 && batch.images[0].generation_seconds == batch.generation_seconds / 2);
        const auto call = [&](int count) {
            return model ? klein->generate_batch(request, count, batch) : anima->generate_batch(request, count, batch);
        };
        for (int count : {0, 9, -1}) CHECK(!call(count) && batch.images.empty() && batch.generation_seconds == 0);
        CHECK(call(pictor::max_batch_count) && batch.images.size() == pictor::max_batch_count);
        request.seed = INT64_MAX; CHECK(!call(2)); CHECK(call(1));
        request.seed = INT64_MAX - 1; CHECK(call(2) && batch.images[1].seed == INT64_MAX);
        request.seed = -1; CHECK(call(2) && batch.images[0].seed >= 0 && batch.images[1].seed == batch.images[0].seed + 1);
        request.width = request.height = 4096; CHECK(!call(2));
        request.width = request.height = 64;
        for (const char* prompt : {"bad-last", "null-last", "fail"}) {
            request.prompt = prompt;
            CHECK(call(2).code == pictor::ErrorCode::backend_error && batch.images.empty());
        }
    }
    pictor::BatchResult edited;
    edit.reference_images = {{16, 16, rgb.data(), rgb.size()}, {16, 16, other.data(), other.size()}};
    CHECK(klein->edit_batch(edit, 2, edited) && edited.images[1].pixels[0] == 68);
    edit.reference_images = {{edited.images[0].width, edited.images[0].height,
                              edited.images[0].pixels.data(), edited.images[0].pixels.size()}};
    CHECK(klein->edit_batch(edit, 2, edited) && edited.images[0].pixels[0] == 90);
    pictor_image* c_images[2] = {};
    double seconds = 0;
    BatchState c_state;
    c_request.seed = 42;
    CHECK(pictor_flux_klein_session_edit_batch(c_session, &c_request, &c_edit, 2, c_batch_progress, &c_state,
                                              c_images, &seconds, &error) == PICTOR_OK);
    CHECK(c_state.calls == 4 && !c_state.failed && seconds > 0);
    pictor_image_info batch_info;
    CHECK(pictor_image_get_info(c_images[1], &batch_info, sizeof(batch_info), &error) == PICTOR_OK);
    CHECK(batch_info.seed == 43 && batch_info.pixels[0] == 68 && batch_info.generation_seconds == seconds / 2);
    CHECK(pictor_session_generate_batch(c_session, &c_request, 2, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_INVALID_ARGUMENT && seconds == 0);
    for (auto*& image : c_images) { pictor_image_destroy(image); image = nullptr; }
    CHECK(pictor_flux_klein_session_edit_batch(c_session, &c_request, nullptr, 2, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_generate_batch(c_session, &c_request, 9, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_INVALID_ARGUMENT);
    c_request.prompt = "null-last";
    CHECK(pictor_session_generate_batch(c_session, &c_request, 2, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_BACKEND_ERROR);
    CHECK(!c_images[0] && !c_images[1] && seconds == 0);
    c_request.prompt = "seed-colors";
    CHECK(pictor_session_generate_batch(c_session, &c_request, 2, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_OK);
    pictor_session_destroy(c_session);
    pictor_image_info info;
    CHECK(pictor_image_get_info(c_image, &info, sizeof(info), &error) == PICTOR_OK && info.pixels[0] == 68);
    pictor_image_destroy(c_image);
    CHECK(pictor_image_get_info(c_images[1], &batch_info, sizeof(batch_info), &error) == PICTOR_OK && batch_info.pixels[0] == 65);
    for (auto* image : c_images) pictor_image_destroy(image);
    pictor_session_options a_options;
    CHECK(pictor_session_options_init(&a_options, sizeof(a_options), &error) == PICTOR_OK);
    a_options.model_path = __FILE__; c_session = nullptr;
    CHECK(pictor_session_create(&a_options, &c_session, &error) == PICTOR_OK);
    CHECK(pictor_request_init(&c_request, sizeof(c_request), PICTOR_PRESET_FAST, &error) == PICTOR_OK);
    CHECK(pictor_flux_klein_session_set_hidden_state_compression(c_session, 1, &error) == PICTOR_INVALID_ARGUMENT);
    c_request.prompt = "seed-colors"; c_request.seed = 42;
    c_images[0] = c_images[1] = nullptr;
    CHECK(pictor_flux_klein_session_edit_batch(c_session, &c_request, &c_edit, 2, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_generate_batch(c_session, &c_request, 2, nullptr, nullptr, c_images, &seconds, &error) == PICTOR_OK);
    pictor_session_destroy(c_session);
    CHECK(pictor_image_get_info(c_images[1], &batch_info, sizeof(batch_info), &error) == PICTOR_OK && batch_info.pixels[0] == 54);
    for (auto* image : c_images) pictor_image_destroy(image);
    anima.reset(); klein.reset();
    CHECK(first.pixels[0] == 11 && second.pixels[0] == 22);
    std::puts("PASS: shared lock, callback isolation, reference routing/ownership, C edit errors, session reuse");
}
