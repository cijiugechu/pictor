#include "pictor/anima.hpp"
#include "cli.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void check(pictor::Status s) {
    if (!s) {
        std::fprintf(stderr, "%s\n", s.message);
        std::exit(1);
    }
}
int main(int argc, char **argv) {
    pictor::cli::Options o;
    check(pictor::cli::parse({argv + 1, argv + argc}, o));
    if (o.help) {
        std::puts(pictor::cli::usage());
        return 0;
    }
    if (o.model != pictor::cli::Model::anima || o.request.seed < 0)
        return 2;
    for (int i = 0; i < o.count; ++i) {
        std::filesystem::path path;
        check(pictor::cli::output_path(o, i, path));
        std::error_code ec;
        const bool exists = std::filesystem::exists(path, ec);
        if (ec || (exists && !o.overwrite)) {
            std::fputs("cannot use output path\n", stderr);
            return 2;
        }
    }
    o.session.verbose = true;
    std::unique_ptr<pictor::AnimaSession> s;
    check(pictor::AnimaSession::create(o.session, s));
    std::fprintf(stderr, "BENCH_LOAD %.6f\n", s->load_seconds());
    int warm = 1;
    if (auto env = std::getenv("PICTOR_BENCH_WARMUP")) {
        if (std::strcmp(env, "0") == 0)
            warm = 0;
        else if (std::strcmp(env, "1") != 0)
            return 2;
    }
    pictor::Image previous;
    for (int i = -warm; i < o.count; ++i) {
        std::fprintf(stderr, "BENCH_BEGIN %d\n", i);
        std::fflush(stderr);
        pictor::Image image;
        check(s->generate(o.request, image));
        std::fprintf(stderr, "BENCH_END %d %.6f\n", i, image.generation_seconds);
        std::fflush(stderr);
        if (!previous.pixels.empty() && previous.pixels != image.pixels) {
            std::fputs("resident RGB mismatch\n", stderr);
            return 1;
        }
        if (i >= 0) {
            std::filesystem::path path;
            check(pictor::cli::output_path(o, i, path));
            check(pictor::write_png(path, image));
        }
        previous = std::move(image);
    }
}
