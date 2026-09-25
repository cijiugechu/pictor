#include "pictor/anima.hpp"
#include "logging.hpp"

#include <algorithm>
#include <iostream>
#include <cstdlib>

namespace {
void check(pictor::Status status) {
    if (!status) { pictor::logging::app().error("{}", status.message); std::exit(1); }
}
void require(bool condition, const char* message) {
    if (!condition) { pictor::logging::app().error("{}", message); std::exit(1); }
}
}

int main(int argc, char** argv) {
    const std::filesystem::path model = argc > 1 ? argv[1] : "models/Anima-P3-Turbo-AIO-Q4_K.gguf";
    const std::filesystem::path output = argc > 2 ? argv[2] : "outputs/smoke";
    std::unique_ptr<pictor::AnimaSession> session;
    check(pictor::AnimaSession::create({model}, session));
    auto request = pictor::preset_request(pictor::Preset::fast);
    request.prompt = "masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people";
    request.seed = 666;
    request.cache = pictor::CacheMode::none;
    pictor::logging::app().info("Smoke: first generation (cache disabled)");
    pictor::Image first;
    check(session->generate(request, first));
    pictor::logging::app().info("Smoke: repeating the identical request");
    pictor::Image second;
    check(session->generate(request, second));
    require(first.pixels == second.pixels && first.seed == second.seed, "same seed/request changed output when reusing the session");
    const auto range = std::minmax_element(first.pixels.begin(), first.pixels.end());
    require(first.width == 512 && first.height == 768 && *range.second - *range.first >= 32, "unexpected dimensions or nearly blank image");
    check(pictor::write_png(output / "repeat-1.png", first));
    check(pictor::write_png(output / "repeat-2.png", second));
    request.seed = 667;
    request.prompt = "masterpiece, best quality, anime landscape, an ancient castle in snow, night sky, no people";
    request.cache = pictor::CacheMode::spectrum;
    pictor::logging::app().info("Smoke: changing prompt, seed, and cache");
    pictor::Image changed;
    check(session->generate(request, changed));
    require(changed.pixels != first.pixels, "new request reused stale output");
    check(pictor::write_png(output / "changed.png", changed));
    std::cout << "PASS: resident session, deterministic repeat, changed prompt/seed/cache, PNG output\n"
              << "load=" << session->load_seconds() << "s; first=" << first.generation_seconds
              << "s; repeat=" << second.generation_seconds << "s; changed=" << changed.generation_seconds << "s\n";
    return 0;
}
