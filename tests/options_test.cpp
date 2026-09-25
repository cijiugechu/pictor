#include "cli.hpp"

#include <iostream>
#include <limits>
#include <cstdlib>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

pictor::cli::Options parse(const std::vector<std::string>& args) {
    pictor::cli::Options output;
    const auto status = pictor::cli::parse(args, output);
    require(status.ok(), status.message);
    return output;
}

void rejects(const std::vector<std::string>& args) {
    pictor::cli::Options output;
    require(pictor::cli::parse(args, output).code == pictor::ErrorCode::invalid_argument, "accepted invalid arguments");
}
} // namespace

int main() {
    using namespace pictor;
    auto options = parse({"anima", "-p", "a cat"});
    require(options.request.width == 512 && options.request.height == 768, "default resolution");
    require(options.request.steps == 8 && options.request.cache == CacheMode::spectrum, "balanced preset");
    options = parse({"anima", "-p", "a cat", "--preset", "quality"});
    require(options.request.steps == 16 && options.request.cache == CacheMode::none, "quality preset");
    options = parse({"anima", "-p", "a cat", "--preset", "fast"});
    require(options.request.steps == 3 && options.request.cache == CacheMode::spectrum, "fast preset");
    options = parse({"anima", "-p", "a cat", "--steps", "5", "--cache", "none", "--preset", "fast"});
    require(options.request.steps == 5 && options.request.cache == CacheMode::none, "override before preset");
    options = parse({"anima", "-p", "a cat", "--preset", "quality", "--steps", "9", "--cache", "spectrum"});
    require(options.request.steps == 9 && options.request.cache == CacheMode::spectrum, "override after preset");

    for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
        {"--width", "513"}, {"--height", "0"}, {"--width", "8192"}, {"--steps", "0"},
        {"--steps", "2x"}, {"--steps", "9999999999999"}, {"--cfg-scale", "nan"},
        {"--cfg-scale", "inf"}, {"--cfg-scale", "-1"}, {"--seed", "-2"},
        {"--count", "0"}, {"--count", "65"}, {"--threads", "-1"},
        {"--preset", "wrong"}, {"--cache", "wrong"}, {"--output", "image.jpg"},
        {"--model", ""}, {"--seed", "9223372036854775808"}})
        rejects({"anima", "-p", "cat", pair.first, pair.second});
    rejects({"anima", "-p", "cat", "--seed", "9223372036854775807", "--count", "2"});
    rejects({"anima", "--prompt", " "});
    rejects({"anima", "--prompt"});
    rejects({"anima", "-p", "cat", "--bogus"});
    rejects({"other"});

    options = parse({"anima", "-p", "cat", "-o", "some dir/a.b.png", "--count", "2"});
    std::filesystem::path path;
    require(cli::output_path(options, 0, path).ok() && path == "some dir/a.b-001.png", "first batch output");
    require(cli::output_path(options, 1, path).ok() && path == "some dir/a.b-002.png", "second batch output");
    require(!cli::output_path(options, 2, path) && path.empty(), "invalid batch output");
    std::int64_t seed;
    require(resolve_seed(123, seed).ok() && seed == 123, "fixed seed");
    require(resolve_seed(-1, seed).ok() && seed >= 0, "random seed");
    require(!resolve_seed(-2, seed) && seed == 0, "invalid seed");
    require(cli::json_string("quote\"\n\\") == "\"quote\\\"\\u000a\\\\\"", "JSON escaping");
    require(parse({}).help && parse({"--help"}).help && parse({"anima", "--help"}).help, "help");
    require(parse({"--version"}).version, "version");
    std::cout << "All option tests passed\n";
    return 0;
}
