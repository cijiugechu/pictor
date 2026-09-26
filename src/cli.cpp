#include "cli.hpp"

#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>

namespace pictor::cli {
namespace {
template<typename T>
Status integer(const std::string& text, const std::string& option, T& value) noexcept {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        return failure(ErrorCode::invalid_argument, option + " requires an integer in range");
    return {};
}

Status real(const std::string& text, const std::string& option, float& value) noexcept {
    char* end = nullptr;
    errno = 0;
    value = std::strtof(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || end != text.c_str() + text.size() || !std::isfinite(value))
        return failure(ErrorCode::invalid_argument, option + " requires a finite number");
    return {};
}
} // namespace

Status parse(const std::vector<std::string>& args, Options& output) noexcept {
    output = {};
    Options options;
    options.session.model_path = "models/Anima-P3-Turbo-AIO-Q4_K.gguf";
    if (args.empty() || (args.size() == 1 && (args[0] == "--help" || args[0] == "-h"))) {
        options.help = true; output = std::move(options); return {};
    }
    if (args.size() == 1 && args[0] == "--version") {
        options.version = true; output = std::move(options); return {};
    }
    if (args[0] == "flux-klein") {
        options.model = Model::flux_klein;
        options.session.model_path = "models/flux2-klein-4b/flux-2-klein-4b-Q4_0.gguf";
        options.text_encoder = "models/flux2-klein-4b/Qwen3-4B-Q4_K_M.gguf";
        options.vae = "models/flux2-klein-4b/full_encoder_small_decoder.safetensors";
        options.request = flux_klein_request();
    } else if (args[0] != "anima") {
        return failure(ErrorCode::invalid_argument, "expected 'anima' or 'flux-klein' (see --help)");
    }

    bool diffusion_given = false, text_given = false, anima_model_given = false;
    Preset preset = Preset::balanced;
    std::optional<int> steps;
    std::optional<CacheMode> cache;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg == "--help" || arg == "-h") { options.help = true; output = std::move(options); return {}; }
        if (arg == "--verbose") { options.session.verbose = true; continue; }
        if (arg == "--overwrite") { options.overwrite = true; continue; }
        if (arg == "--hs-compression") {
            if (options.model != Model::flux_klein)
                return failure(ErrorCode::invalid_argument, "--hs-compression requires flux-klein");
            options.hidden_state_compression = true; continue;
        }
        if (arg == "--vae-tiling") { options.request.vae_tiling = true; continue; }
        if (arg == "--disable-auto-resize-ref-image") {
            if (options.model != Model::flux_klein) return failure(ErrorCode::invalid_argument, "reference flags require flux-klein");
            options.auto_resize_reference = false; continue;
        }
        if (i + 1 >= args.size()) return failure(ErrorCode::invalid_argument, "missing value for " + arg);
        const auto& value = args[++i];
        Status status;
        if (arg == "--backend") {
            if (value == "auto") { options.backend = KleinBackend::automatic; options.anima_backend = AnimaBackend::automatic; }
            else if (value == "ggml") { options.backend = KleinBackend::ggml; options.anima_backend = AnimaBackend::ggml; }
            else if (value == "mlx") { options.backend = KleinBackend::mlx; options.anima_backend = AnimaBackend::mlx; }
            else return failure(ErrorCode::invalid_argument, "backend must be auto, ggml or mlx");
        } else if (arg == "--model") {
            if (options.model != Model::anima) return failure(ErrorCode::invalid_argument, "Klein uses --diffusion-model, --text-encoder and --vae");
            options.session.model_path = value;
            anima_model_given = true;
        } else if (arg == "--diffusion-model" || arg == "--text-encoder" || arg == "--llm" || arg == "--vae") {
            if (options.model != Model::flux_klein) return failure(ErrorCode::invalid_argument, arg + " is only supported by flux-klein");
            if (arg == "--diffusion-model") { options.session.model_path = value; diffusion_given = true; }
            else if (arg == "--vae") options.vae = value;
            else { options.text_encoder = value; text_given = true; }
        }
        else if (arg == "--ref-image" || arg == "-r") {
            if (options.model != Model::flux_klein || value.empty())
                return failure(ErrorCode::invalid_argument, "--ref-image requires flux-klein and a nonempty path");
            if (options.reference_images.size() == 4)
                return failure(ErrorCode::invalid_argument, "at most 4 reference images are supported");
            options.reference_images.emplace_back(value);
        }
        else if (arg == "--prompt" || arg == "-p") options.request.prompt = value;
        else if (arg == "--negative-prompt") options.request.negative_prompt = value;
        else if (arg == "--output" || arg == "-o") options.output = value;
        else if (arg == "--width" || arg == "-W") status = integer(value, arg, options.request.width);
        else if (arg == "--height" || arg == "-H") status = integer(value, arg, options.request.height);
        else if (arg == "--steps") { int parsed = 0; status = integer(value, arg, parsed); steps = parsed; }
        else if (arg == "--cfg-scale") status = real(value, arg, options.request.cfg_scale);
        else if (arg == "--seed") status = integer(value, arg, options.request.seed);
        else if (arg == "--count") status = integer(value, arg, options.count);
        else if (arg == "--threads") status = integer(value, arg, options.session.threads);
        else if (arg == "--preset") {
            if (options.model != Model::anima) return failure(ErrorCode::invalid_argument, "--preset is Anima-only; Klein defaults to 4 steps");
            if (value == "fast") preset = Preset::fast;
            else if (value == "balanced") preset = Preset::balanced;
            else if (value == "quality") preset = Preset::quality;
            else return failure(ErrorCode::invalid_argument, "preset must be fast, balanced, or quality");
        } else if (arg == "--cache") {
            if (value == "none") cache = CacheMode::none;
            else if (value == "spectrum") cache = CacheMode::spectrum;
            else return failure(ErrorCode::invalid_argument, "cache must be none or spectrum");
        } else return failure(ErrorCode::invalid_argument, "unknown argument: " + arg);
        if (!status) return status;
    }

    if (options.model == Model::flux_klein) {
        auto selected = options.backend;
        if (selected == KleinBackend::automatic) {
            std::error_code ec;
            if (diffusion_given) selected = std::filesystem::is_directory(options.session.model_path,ec) ? KleinBackend::mlx : KleinBackend::ggml;
            else if (text_given) selected = std::filesystem::is_directory(options.text_encoder,ec) ? KleinBackend::mlx : KleinBackend::ggml;
#ifdef PICTOR_DEFAULT_MLX
            else selected = KleinBackend::mlx;
#else
            else selected = KleinBackend::ggml;
#endif
        }
        if (selected == KleinBackend::mlx) {
            if (!diffusion_given) options.session.model_path = "models/mlx-flux2-klein-4b-4bit/transformer";
            if (!text_given) options.text_encoder = "models/mlx-flux2-klein-4b-4bit/text_encoder";
        }
        options.backend = selected;
        if (selected == KleinBackend::mlx && options.hidden_state_compression)
            return failure(ErrorCode::invalid_argument, "--hs-compression requires --backend ggml");
    }
    if (options.model == Model::anima) {
        auto selected = options.anima_backend;
        if (selected == AnimaBackend::automatic) {
            std::error_code ec;
            if (anima_model_given) selected = std::filesystem::is_directory(options.session.model_path, ec) ? AnimaBackend::mlx : AnimaBackend::ggml;
#ifdef PICTOR_DEFAULT_MLX
            else selected = AnimaBackend::mlx;
#else
            else selected = AnimaBackend::ggml;
#endif
        }
        if (!anima_model_given && selected == AnimaBackend::mlx) options.session.model_path = "models/anima-p3-mlx-bf16";
        options.anima_backend = selected;
    }
    const auto defaults = options.model == Model::anima
        ? (options.anima_backend == AnimaBackend::mlx ? anima_request(preset) : preset_request(preset)) : flux_klein_request();
    options.request.steps = steps.value_or(defaults.steps);
    options.request.cache = cache.value_or(defaults.cache);
    if (options.model == Model::anima && options.anima_backend == AnimaBackend::mlx && options.request.cache != CacheMode::none)
        return failure(ErrorCode::invalid_argument, "Anima MLX requires --cache none; Spectrum requires --backend ggml");
    const auto validation = options.model == Model::anima ? validate_request(options.request) : validate_flux_klein_request(options.request);
    if (!validation) return validation;
    if (!options.auto_resize_reference && options.reference_images.empty())
        return failure(ErrorCode::invalid_argument, "--disable-auto-resize-ref-image requires --ref-image");
    if (options.model == Model::flux_klein && (options.text_encoder.empty() || options.vae.empty()))
        return failure(ErrorCode::invalid_argument, "Klein text encoder and VAE paths must not be empty");
    if (options.session.model_path.empty()) return failure(ErrorCode::invalid_argument, "model path must not be empty");
    if (options.session.threads < 0) return failure(ErrorCode::invalid_argument, "threads must be nonnegative");
    if (options.count < 1 || options.count > 64) return failure(ErrorCode::invalid_argument, "count must be between 1 and 64");
    if (options.request.seed > std::numeric_limits<std::int64_t>::max() - (options.count - 1))
        return failure(ErrorCode::invalid_argument, "seed sequence would overflow");
    if (options.output.extension() != ".png") return failure(ErrorCode::invalid_argument, "output must have a .png extension");
    output = std::move(options);
    return {};
}

Status output_path(const Options& options, int index, std::filesystem::path& output) noexcept {
    output.clear();
    if (index < 0 || index >= options.count) return failure(ErrorCode::invalid_argument, "image index out of range");
    if (options.count == 1) { output = options.output; return {}; }
    std::ostringstream name;
    name << options.output.stem().string() << '-' << std::setfill('0') << std::setw(3) << index + 1 << ".png";
    output = options.output.parent_path() / name.str();
    return {};
}

std::string json_string(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

const char* usage() {
    return R"(pictor 0.1.0 — Anima and FLUX.2-klein-4B inference

Usage:
  pictor anima --prompt "anime landscape" [options]
  pictor flux-klein --prompt "a red fox" [options]

  --model PATH          Anima GGUF or MLX directory (MLX default: models/anima-p3-mlx-bf16)
  --backend MODE        auto (default), mlx, ggml; MLX-enabled builds default both models to MLX
                        Anima ggml default: models/Anima-P3-Turbo-AIO-Q4_K.gguf
  --diffusion-model PATH Klein diffusion GGUF or MLX safetensors directory
  --text-encoder PATH    Klein Qwen3 GGUF or MLX directory (alias: --llm)
  --vae PATH             Klein VAE (default: full_encoder_small_decoder.safetensors)
  --ref-image, -r PATH    Klein reference PNG/JPEG; repeat for up to 4 ordered images
  --disable-auto-resize-ref-image  Keep reference size (multiples of 16 required)
                        MLX weights: models/mlx-flux2-klein-4b-4bit/; VAE: models/flux2-klein-4b/
  --prompt, -p TEXT      Required positive prompt
  --negative-prompt TEXT Negative prompt (CFG 1 normally skips unconditional guidance)
  --preset NAME         Anima: fast 3, balanced 8, quality 16 steps; MLX uses no cache
  --steps N             Steps (1..1000); Klein default: 4
  --cache MODE          none or spectrum (Spectrum: Anima ggml only; fast/balanced default)
  --cfg-scale N          Guidance scale (default: 1)
  --width, -W N          Width in pixels (default: 512)
  --height, -H N         Height (Anima default: 768; Klein default: 512)
  --hs-compression       Klein ggml experimental hidden-state compression (default off; may cause artifacts)
  --seed N               Nonnegative seed or -1 for random (default)
  --count N              Generate 1..64 images in bounded batches; seeds increment
  --output, -o PATH      PNG path (default: output.png); count >1 adds -001, -002, ...
  --vae-tiling           Decode VAE in tiles to reduce peak memory
  --threads N            ggml CPU threads (MLX manages its own scheduling)
  --overwrite           Allow replacing existing PNG/JSON output files
  --verbose             Print backend logs
  --help, -h            Show help
  --version              Show version

Anima: 512x768, er_sde/smoothstep, flash attention.
Klein 4B distilled: 512x512, 4 steps, CFG 1, Euler/discrete, diffusion flash attention.
Klein supports text-to-image and reference-image editing; output size stays explicit.
Dimensions must be multiples of 16, from 64 to 4096; large images may exhaust memory.
Batches reuse text/reference encodings; at most 8 images / 16 megapixels per batch.
Images are saved after each batch completes. Timings include the batch average.
Each PNG has a JSON sidecar with effective settings, seed, and timings.
Builds and --help do not need weights. Fetch with:
  zig build download-model        (Anima)
  zig build download-klein-model  (Klein, Qwen3, VAE)
)";
}

} // namespace pictor::cli
