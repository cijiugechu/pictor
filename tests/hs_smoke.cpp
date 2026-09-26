#include "pictor/flux_klein.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(pictor::Status status) {
    if (!status) { std::fprintf(stderr, "%s\n", status.message); std::exit(1); }
}
static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
int main(int argc, char** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : "models/flux2-klein-4b";
    const std::filesystem::path out = argc > 2 ? argv[2] : "outputs/hs-smoke";
    std::unique_ptr<pictor::FluxKleinSession> session;
    check(pictor::FluxKleinSession::create({dir / "flux-2-klein-4b-Q4_0.gguf", dir / "Qwen3-4B-Q4_K_M.gguf",
                                          dir / "flux2-vae.safetensors", 0, true}, session));
    auto request = pictor::flux_klein_request();
    request.prompt = "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light";
    request.width = 80; request.height = 112; request.seed = 666; request.steps = 1;
    pictor::Image exact, accelerated, restored;
    check(session->generate(request, exact)); // Default off.
    check(session->set_hidden_state_compression(true));
    check(session->generate(request, accelerated));
    require(exact.pixels == accelerated.pixels, "one-step request must remain exact");
    request.steps = 2;
    check(session->set_hidden_state_compression(false));
    check(session->generate(request, exact));
    check(session->set_hidden_state_compression(true));
    check(session->generate(request, accelerated));
    require(exact.pixels != accelerated.pixels, "HS did not affect the multi-step forward");
    check(session->set_hidden_state_compression(false));
    check(session->generate(request, restored));
    require(exact.pixels == restored.pixels, "disabling HS failed to restore the original output");
    std::puts("PASS: one-step exact, odd rectangular output, disable restores exact path"); std::fflush(stdout);

    // Distinct rectangular reference grids including odd rows/columns, no resize.
    std::vector<std::uint8_t> ref1(80 * 64 * 3), ref2(112 * 80 * 3);
    for (std::size_t i = 0; i < ref1.size(); ++i) ref1[i] = static_cast<uint8_t>((i * 7) % 256);
    for (std::size_t i = 0; i < ref2.size(); ++i) ref2[i] = static_cast<uint8_t>((i * 3) % 256);
    pictor::FluxKleinEditRequest edit;
    edit.generation = request;
    edit.auto_resize = false;
    edit.generation.vae_tiling = true;
    edit.reference_images = {{80, 64, ref1.data(), ref1.size()}, {112, 80, ref2.data(), ref2.size()}};
    check(session->set_hidden_state_compression(true));
    pictor::BatchResult batch;
    check(session->edit_batch(edit, 2, batch));
    for (int i = 0; i < 2; ++i) {
        edit.generation.seed = request.seed + i;
        check(session->edit(edit, accelerated));
        require(accelerated.pixels == batch.images[i].pixels, "HS batch step schedule did not reset per image");
    }
    for (std::size_t i = 0; i < ref1.size(); ++i) require(ref1[i] == (i * 7) % 256, "reference pixels modified");
    for (std::size_t i = 0; i < ref2.size(); ++i) require(ref2[i] == (i * 3) % 256, "reference pixels modified");
    std::puts("PASS: HS multi-reference odd grids, tiled VAE and batch/single parity"); std::fflush(stdout);

    request.width = request.height = 512; request.steps = 4;
    check(session->set_hidden_state_compression(false));
    check(session->generate(request, exact));
    check(session->set_hidden_state_compression(true));
    check(session->generate(request, accelerated));
    require(exact.pixels.size() == accelerated.pixels.size(), "HS changed output dimensions");
    const auto range = std::minmax_element(accelerated.pixels.begin(), accelerated.pixels.end());
    require(*range.second - *range.first > 32, "HS output is nearly blank");
    double mae = 0, mse = 0;
    for (std::size_t i = 0; i < exact.pixels.size(); ++i) {
        const double diff = static_cast<int>(exact.pixels[i]) - accelerated.pixels[i];
        mae += std::abs(diff); mse += diff * diff;
    }
    mae /= exact.pixels.size(); mse /= exact.pixels.size();
    require(mse > 0, "HS did not change output");
    check(pictor::write_png(out / "exact.png", exact));
    check(pictor::write_png(out / "hs.png", accelerated));
    std::printf("MEASURE: 512x512 four-step output; exact=%.3fs HS=%.3fs RGB_MAE=%.4f PSNR=%.3fdB (difference metrics, not a visual quality pass)\n",
                exact.generation_seconds, accelerated.generation_seconds, mae, 10 * std::log10(255.0 * 255 / mse));
}
