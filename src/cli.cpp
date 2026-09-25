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
    if (args[0] != "anima") return failure(ErrorCode::invalid_argument, "expected the 'anima' command (see --help)");

    Preset preset = Preset::balanced;
    std::optional<int> steps;
    std::optional<CacheMode> cache;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg == "--help" || arg == "-h") { options.help = true; output = std::move(options); return {}; }
        if (arg == "--verbose") { options.session.verbose = true; continue; }
        if (arg == "--overwrite") { options.overwrite = true; continue; }
        if (arg == "--vae-tiling") { options.request.vae_tiling = true; continue; }
        if (i + 1 >= args.size()) return failure(ErrorCode::invalid_argument, "missing value for " + arg);
        const auto& value = args[++i];
        Status status;
        if (arg == "--model") options.session.model_path = value;
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

    const auto defaults = preset_request(preset);
    options.request.steps = steps.value_or(defaults.steps);
    options.request.cache = cache.value_or(defaults.cache);
    if (const auto status = validate_request(options.request); !status) return status;
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
    return R"(pictor 0.1.0 — Anima P3 Turbo AIO inference

Usage:
  pictor anima --prompt "anime landscape" [options]

  --model PATH           AIO GGUF (default: models/Anima-P3-Turbo-AIO-Q4_K.gguf)
  --prompt, -p TEXT      Required positive prompt
  --negative-prompt TEXT Negative prompt (CFG 1 normally skips unconditional guidance)
  --preset NAME          fast: 3/Spectrum; balanced: 8/Spectrum; quality: 16/no cache
  --steps N              Override preset steps (1..1000)
  --cache MODE           Override preset cache: none or spectrum
  --cfg-scale N          Guidance scale (default: 1)
  --width, -W N          Width in pixels (default: 512)
  --height, -H N         Height in pixels (default: 768)
  --seed N               Nonnegative seed or -1 for random (default)
  --count N              Generate 1..64 images using one loaded model; seeds increment
  --output, -o PATH      PNG path (default: output.png); count >1 adds -001, -002, ...
  --vae-tiling           Decode VAE in tiles to reduce peak memory
  --threads N            CPU threads (default: physical core count)
  --overwrite           Allow replacing existing PNG/JSON output files
  --verbose             Print backend logs
  --help, -h            Show help
  --version              Show version

Sampler: er_sde. Scheduler: smoothstep. Flash attention enabled.
Dimensions must be multiples of 16, from 64 to 4096; large images may exhaust memory.
Each PNG has a JSON sidecar with effective settings, seed, and timings.
Builds and --help do not need model weights. Fetch them with: zig build download-model
)";
}

} // namespace pictor::cli
