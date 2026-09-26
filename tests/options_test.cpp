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
    require(!parse({"flux-klein", "-p", "cat"}).hidden_state_compression, "HS must default off");
    require(parse({"flux-klein", "-p", "cat", "--backend", "ggml", "--hs-compression"}).hidden_state_compression, "HS flag");
    rejects({"anima", "-p", "cat", "--hs-compression"});
    auto options = parse({"anima", "-p", "a cat"});
#ifdef PICTOR_DEFAULT_MLX
    const auto default_cache = CacheMode::none;
    require(options.anima_backend == AnimaBackend::mlx && options.session.model_path == "models/anima-p3-mlx-bf16", "Anima MLX default");
#else
    const auto default_cache = CacheMode::spectrum;
    require(options.anima_backend == AnimaBackend::ggml && options.session.model_path.extension() == ".gguf", "Anima ggml default");
#endif
    require(options.request.width == 512 && options.request.height == 768, "default resolution");
    require(options.request.steps == 8 && options.request.cache == default_cache, "balanced preset");
    options = parse({"anima", "-p", "a cat", "--preset", "quality"});
    require(options.request.steps == 16 && options.request.cache == CacheMode::none, "quality preset");
    options = parse({"anima", "-p", "a cat", "--preset", "fast"});
    require(options.request.steps == 3 && options.request.cache == default_cache, "fast preset");
    options = parse({"anima", "-p", "a cat", "--steps", "5", "--cache", "none", "--preset", "fast"});
    require(options.request.steps == 5 && options.request.cache == CacheMode::none, "override before preset");
    options = parse({"anima", "-p", "a cat", "--backend", "ggml", "--preset", "quality", "--steps", "9", "--cache", "spectrum"});
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
    options.output = "some dir/猫{seed}.png";
    require(cli::output_path(options, 1, path).ok() && path == "some dir/猫{seed}-002.png", "literal filename characters");
    std::int64_t seed;
    require(resolve_seed(123, seed).ok() && seed == 123, "fixed seed");
    require(resolve_seed(-1, seed).ok() && seed >= 0, "random seed");
    require(!resolve_seed(-2, seed) && seed == 0, "invalid seed");
    require(cli::json_string("quote\"\n\\") == "\"quote\\\"\\u000a\\\\\"", "JSON escaping");
    require(cli::json_string(std::string_view("\0\x01\t\r\x1f", 5)) == "\"\\u0000\\u0001\\u0009\\u000d\\u001f\"", "JSON control bytes");
    require(cli::json_string("猫 café {prompt}") == "\"猫 café {prompt}\"", "JSON preserves Unicode and braces");
    require(cli::json_string("") == "\"\"", "JSON empty string");
    require(parse({}).help && parse({"--help"}).help && parse({"anima", "--help"}).help, "help");
    require(parse({"--version"}).version, "version");
    options = parse({"flux-klein", "-p", "a fox"});
    require(options.model == cli::Model::flux_klein, "Klein command");
    require(options.request.width == 512 && options.request.height == 512 && options.request.steps == 4 &&
            options.request.cache == CacheMode::none && options.request.cfg_scale == 1, "Klein defaults");
    require(options.vae.filename()=="full_encoder_small_decoder.safetensors", "Small Decoder default");
#ifdef PICTOR_DEFAULT_MLX
    require(options.session.model_path=="models/mlx-flux2-klein-4b-4bit/transformer", "Apple Silicon MLX default");
#endif
    require(parse({"flux-klein","-p","fox","--backend","ggml"}).session.model_path.extension()==".gguf", "ggml fallback defaults");
    require(parse({"flux-klein","-p","fox","--backend","mlx"}).text_encoder.filename()=="text_encoder", "explicit MLX defaults");
    rejects({"flux-klein","-p","fox","--backend","other"});
    require(parse({"anima","-p","fox","--backend","mlx"}).request.cache == CacheMode::none, "Anima explicit MLX");
    require(parse({"anima","-p","fox","--model","legacy.gguf"}).anima_backend == AnimaBackend::ggml, "explicit GGUF preserves ggml");
    require(parse({"anima","-p","fox","--model","tests"}).anima_backend == AnimaBackend::mlx, "directory selects MLX");
    rejects({"anima","-p","fox","--backend","mlx","--cache","spectrum"});
    require(parse({"anima","-p","fox","--backend","ggml"}).request.cache == CacheMode::spectrum, "explicit ggml preserves Spectrum");
    rejects({"flux-klein","-p","fox","--backend","mlx","--hs-compression"});
    options = parse({"flux-klein", "-p", "a fox", "--diffusion-model", "dit.gguf", "--text-encoder", "te.gguf", "--vae", "vae.safetensors", "--steps", "6", "--threads", "2"});
    require(options.session.model_path == "dit.gguf" && options.text_encoder == "te.gguf" && options.vae == "vae.safetensors" && options.request.steps == 6 && options.session.threads == 2, "Klein split model flags");
    require(parse({"flux-klein", "-p", "fox", "--llm", "alias.gguf"}).text_encoder == "alias.gguf", "LLM alias");
    rejects({"flux-klein", "-p", "fox", "--model", "aio.gguf"});
    rejects({"flux-klein", "-p", "fox", "--preset", "fast"});
    rejects({"flux-klein", "-p", "fox", "--cache", "spectrum"});
    rejects({"flux-klein", "-p", "fox", "--text-encoder", ""});
    rejects({"flux-klein", "-p", "fox", "--vae", ""});
    rejects({"anima", "-p", "cat", "--diffusion-model", "dit.gguf"});
    rejects({"anima", "-p", "cat", "--vae", "vae.safetensors"});
    require(parse({"flux-klein", "--help"}).help, "Klein help");
    options = parse({"flux-klein", "-p", "winter", "-r", "one.png", "--ref-image", "two.jpg", "--disable-auto-resize-ref-image"});
    require(options.reference_images.size() == 2 && options.reference_images[0] == "one.png" &&
            options.reference_images[1] == "two.jpg" && !options.auto_resize_reference, "ordered reference flags");
    rejects({"anima", "-p", "winter", "-r", "one.png"});
    rejects({"flux-klein", "-p", "winter", "--ref-image", ""});
    rejects({"flux-klein", "-p", "winter", "--disable-auto-resize-ref-image"});
    rejects({"flux-klein", "-p", "winter", "-r", "1", "-r", "2", "-r", "3", "-r", "4", "-r", "5"});
    std::cout << "All option tests passed\n";
    return 0;
}
