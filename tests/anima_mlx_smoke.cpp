#include "pictor/anima.hpp"
#include "pictor/pictor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <unistd.h>
using namespace pictor;
static void require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
static void check(Status s) { require(s.ok(), s.message); }
static void check_c(pictor_status s, const pictor_error &e) { require(s == PICTOR_OK, e.message); }
int main() {
    std::unique_ptr<AnimaSession> session;
    const auto broken = std::filesystem::temp_directory_path() / ("pictor-anima-broken-" + std::to_string(getpid()));
    require(std::filesystem::create_directory(broken), "create isolated corrupt weights fixture");
    for (const auto *name : {"transformer-bf16.safetensors", "text_encoder-bf16.safetensors",
                             "llm_adapter-bf16.safetensors", "vae-bf16.safetensors"})
        std::ofstream(broken / name) << "invalid safetensors";
    require(!AnimaSession::create({broken}, AnimaBackend::mlx, session) && !session,
            "load exception stays inside boundary");
    std::filesystem::remove_all(broken);
    const SessionOptions options{"models/anima-p3-mlx-bf16", 0, true};
    check(AnimaSession::create(options, session));
    require(session->backend() == AnimaBackend::mlx, "automatic directory selects MLX");
    require(!AnimaSession::create(options, session), "occupied session output rejected");
    auto request = anima_request(Preset::fast);
    request.prompt = "一只 (red fox:1.2), 雪地，蓝色的天空";
    request.width = 80;
    request.height = 64;
    request.seed = 42;
    struct Events {
        int calls = 0;
    } events;
    const auto progress = [](const BatchProgress &p, void *data) noexcept {
        auto &e = *static_cast<Events *>(data);
        require(p.image_count == 2 && p.image_index == e.calls / 4, "batch progress order");
        require(p.sampling.steps == 3 && p.sampling.step == e.calls % 4, "sampling-only progress 0..steps");
        ++e.calls;
    };
    BatchResult batch;
    check(session->generate_batch(request, 2, batch, progress, &events));
    require(events.calls == 8 && batch.images.size() == 2 && batch.images[1].seed == 43,
            "batch shape and consecutive seeds");
    require(batch.images[0].generation_seconds == batch.generation_seconds / 2, "batch amortized timing");
    Image first, second;
    Status left_status, right_status;
    auto next = request;
    ++next.seed;
    int single_events = 0;
    const auto single_progress = [](const Progress &p, void *data) noexcept {
        auto &count = *static_cast<int *>(data);
        require(p.steps == 3 && p.step == count++, "single progress adapter preserves 0..steps");
    };
    std::thread left([&] { left_status = session->generate(request, first, single_progress, &single_events); });
    std::thread right([&] { right_status = session->generate(next, second); });
    left.join();
    right.join();
    check(left_status);
    check(right_status);
    require(single_events == 4, "single callback count");
    require(first.pixels == batch.images[0].pixels && second.pixels == batch.images[1].pixels,
            "concurrent calls serialize and match batch exactly");
    auto guided = request;
    guided.cfg_scale = 2;
    guided.negative_prompt = "blurry";
    Image image;
    check(session->generate(guided, image));
    require(image.pixels != first.pixels, "CFG/negative pass changes output");
    guided.cfg_scale = 0;
    guided.negative_prompt.clear();
    check(session->generate(guided, image));
    const auto unconditional = image.pixels;
    guided.prompt = "a completely different positive prompt";
    check(session->generate(guided, image));
    require(image.pixels == unconditional, "CFG zero ignores positive conditioning");
    auto tiled = request;
    tiled.vae_tiling = true;
    check(session->generate(tiled, image));
    require(image.pixels == first.pixels, "single-tile path exact");
    tiled.width = 272;
    tiled.steps = 1;
    check(session->generate(tiled, image));
    require(image.pixels.size() == 272 * 64 * 3, "overlapping edge tiles have correct dimensions");
    check(write_png("outputs/anima-api/tiled-smoke.png", image));
    auto invalid = request;
    invalid.cache = CacheMode::spectrum;
    require(!session->generate(invalid, image) && image.pixels.empty(),
            "explicit unsupported cache fails and clears output");
    invalid = request;
    invalid.seed = std::numeric_limits<std::int64_t>::max();
    require(!session->generate_batch(invalid, 2, batch) && batch.images.empty(), "overflow clears batch");
    invalid = request;
    for (const auto &bad :
         {std::string("\xff"), std::string("\xe4\xb8"), std::string("\xc0\xaf"), std::string("\xed\xa0\x80")}) {
        invalid.prompt = bad;
        require(session->generate(invalid, image).code == ErrorCode::invalid_argument && image.pixels.empty(),
                "invalid UTF-8 returns status");
    }
    check(session->generate(request, image));
    require(image.pixels == first.pixels, "session recovers after errors");
    auto random = request;
    random.seed = -1;
    random.steps = 1;
    check(session->generate_batch(random, 2, batch));
    require(batch.images[0].seed >= 0 && batch.images[1].seed == batch.images[0].seed + 1,
            "random base seed resolved once");
    session.reset();
    require(!first.pixels.empty(), "C++ pixels outlive session");

    pictor_error error{};
    pictor_session_options c_options{};
    check_c(pictor_session_options_init(&c_options, sizeof(c_options), &error), error);
    c_options.model_path = "models/anima-p3-mlx-bf16";
    pictor_session *c_session = nullptr;
    check_c(pictor_anima_session_create_with_backend(&c_options, PICTOR_ANIMA_BACKEND_MLX, &c_session, &error), error);
    pictor_anima_backend backend;
    check_c(pictor_anima_session_backend(c_session, &backend, &error), error);
    require(backend == PICTOR_ANIMA_BACKEND_MLX, "C backend query");
    pictor_klein_backend wrong_backend;
    require(pictor_flux_klein_session_backend(c_session, &wrong_backend, &error) == PICTOR_INVALID_ARGUMENT,
            "wrong model query rejected");
    pictor_request c_request{};
    check_c(pictor_anima_request_init(&c_request, sizeof(c_request), PICTOR_PRESET_FAST, &error), error);
    c_request.prompt = request.prompt.c_str();
    c_request.width = 80;
    c_request.height = 64;
    c_request.seed = 42;
    pictor_image *images[2]{};
    double seconds = 0;
    int c_events = 0;
    const auto c_progress = [](int32_t index, int32_t count, int32_t step, int32_t steps, float, void *data) {
        auto &calls = *static_cast<int *>(data);
        require(count == 2 && index == calls / 4 && steps == 3 && step == calls % 4, "C batch progress bridge");
        ++calls;
    };
    check_c(pictor_session_generate_batch(c_session, &c_request, 2, c_progress, &c_events, images, &seconds, &error),
            error);
    require(c_events == 8, "C callback count");
    for (int i = 0; i < 2; ++i) {
        pictor_image_info info{};
        check_c(pictor_image_get_info(images[i], &info, sizeof(info), &error), error);
        const auto &expected = i ? second : first;
        require(info.seed == 42 + i && info.generation_seconds == seconds / 2 &&
                    info.pixels_len == expected.pixels.size() &&
                    std::memcmp(info.pixels, expected.pixels.data(), info.pixels_len) == 0,
                "C/C++ batch exact pixels and timing");
    }
    pictor_session_destroy(c_session);
    check_c(pictor_image_write_png(images[0], "outputs/anima-api/c-smoke.png", &error), error);
    for (auto *value : images)
        pictor_image_destroy(value);
    check(AnimaSession::create({"models/xocialize-anima-mlx"}, session));
    request.steps = 1;
    check(session->generate(request, image));
    require(image.width == 80 && image.height == 64, "public INT4 checkpoint through public API");
    std::puts("PASS: Anima MLX public C/C++, batching, concurrency, CFG, tiling, errors, ownership and public weights");
}
