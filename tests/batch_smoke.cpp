// Model-dependent parity and resident timing check: batch vs consecutive calls.
#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"
#include <cstdio>
#include <cstdlib>

static void check(pictor::Status status) {
    if (!status) { std::fprintf(stderr, "%s\n", status.message); std::exit(1); }
}
static void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
struct State { int current = 0; int completed = 0; bool failed = false; };
static void progress(const pictor::BatchProgress& event, void* data) noexcept {
    auto& state = *static_cast<State*>(data);
    if (event.image_index != state.current || event.image_count != 2) state.failed = true;
    if (event.sampling.step == event.sampling.steps) { ++state.current; ++state.completed; }
}
int main(int argc, char** argv) {
    const bool klein = argc > 1 && std::string(argv[1]) == "--klein";
    std::unique_ptr<pictor::AnimaSession> anima;
    std::unique_ptr<pictor::FluxKleinSession> flux;
    if (klein) {
        const std::filesystem::path dir = argc > 2 ? argv[2] : "models/flux2-klein-4b";
        check(pictor::FluxKleinSession::create({dir / "flux-2-klein-4b-Q4_0.gguf", dir / "Qwen3-4B-Q4_K_M.gguf",
                                              dir / "flux2-vae.safetensors", 0, true}, flux));
    } else {
        check(pictor::AnimaSession::create({argc > 2 ? argv[2] : "models/Anima-P3-Turbo-AIO-Q4_K.gguf", 0, true}, anima));
    }
    auto request = klein ? pictor::flux_klein_request() : pictor::preset_request(pictor::Preset::fast);
    request.prompt = "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light";
    request.width = request.height = 256;
    request.seed = 42;
    pictor::Image references[2];
    pictor::FluxKleinEditRequest edit;
    for (int mode = 0; mode < 2; ++mode) {
        if (mode == 1) {
            if (klein) {
                request.vae_tiling = true; // Exercise non-sampling progress during reference encoding/decoding.
                request.prompt = "Put the fox from image 1 into the forest from image 2.";
                for (const auto& image : references)
                    edit.reference_images.push_back({image.width, image.height, image.pixels.data(), image.pixels.size()});
            }
            else { request.negative_prompt = "blurry"; request.cfg_scale = 2; request.cache = pictor::CacheMode::none; }
        }
        auto single = [&](int index, pictor::Image& output) {
            auto item = request; item.seed += index;
            edit.generation = item;
            return anima ? anima->generate(item, output) : mode ? flux->edit(edit, output) : flux->generate(item, output);
        };
        pictor::Image sequential[2];
        double serial_seconds = 0;
        for (int i = 0; i < 2; ++i) { check(single(i, sequential[i])); serial_seconds += sequential[i].generation_seconds; }
        edit.generation = request;
        pictor::BatchResult batch;
        State state;
        check(anima ? anima->generate_batch(request, 2, batch, progress, &state)
                    : mode ? flux->edit_batch(edit, 2, batch, progress, &state)
                           : flux->generate_batch(request, 2, batch, progress, &state));
        require(!state.failed && state.completed == 2, "incorrect batch progress");
        require(batch.images.size() == 2, "incorrect image count");
        for (int i = 0; i < 2; ++i)
            require(batch.images[i].pixels == sequential[i].pixels && batch.images[i].seed == request.seed + i,
                    "batch differs from individual generation");
        double repeat_seconds = 0;
        for (int i = 0; i < 2; ++i) {
            pictor::Image repeated; check(single(i, repeated)); repeat_seconds += repeated.generation_seconds;
            require(repeated.pixels == sequential[i].pixels, "batch contaminated subsequent generation");
        }
        std::printf("PASS model=%s mode=%d serial=%.3f batch=%.3f serial_repeat=%.3f seconds; exact RGB parity\n",
                    klein ? "klein" : "anima", mode, serial_seconds, batch.generation_seconds, repeat_seconds);
        std::fflush(stdout);
        if (klein && mode == 0) { references[0] = std::move(sequential[0]); references[1] = std::move(sequential[1]); }
    }
}
