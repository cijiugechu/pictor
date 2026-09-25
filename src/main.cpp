#include "cli.hpp"
#include "logging.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
pictor::Status write_metadata(const std::filesystem::path& path, const pictor::cli::Options& options,
                    const pictor::Image& image, double load_seconds) noexcept {
    std::error_code ec;
    const auto model = std::filesystem::absolute(options.session.model_path, ec);
    if (ec) return pictor::failure(pictor::ErrorCode::io_error, "cannot resolve model path: " + ec.message());
    std::ofstream out(path);
    if (!out) return pictor::failure(pictor::ErrorCode::io_error, "cannot write metadata: " + path.string());
    const auto& request = options.request;
    const auto quote = pictor::cli::json_string;
    out << std::setprecision(9)
        << "{\n  \"pictor_version\": \"0.1.0\",\n"
        << "  \"sd_cpp_revision\": \"90e87bc846f17059771efb8aaa31e9ef0cab6f78\",\n"
        << "  \"model\": " << quote(model.string()) << ",\n"
        << "  \"prompt\": " << quote(request.prompt) << ",\n"
        << "  \"negative_prompt\": " << quote(request.negative_prompt) << ",\n"
        << "  \"width\": " << image.width << ",\n  \"height\": " << image.height << ",\n"
        << "  \"steps\": " << request.steps << ",\n  \"cfg_scale\": " << request.cfg_scale << ",\n"
        << "  \"seed\": " << image.seed << ",\n"
        << "  \"sampler\": \"er_sde\",\n  \"scheduler\": \"smoothstep\",\n"
        << "  \"cache\": " << quote(request.cache == pictor::CacheMode::spectrum ? "spectrum" : "none") << ",\n"
        << "  \"flash_attention\": true,\n"
        << "  \"vae_tiling\": " << (request.vae_tiling ? "true" : "false") << ",\n"
        << "  \"load_seconds\": " << load_seconds << ",\n"
        << "  \"generation_seconds\": " << image.generation_seconds << "\n}\n";
    out.close();
    if (!out) return pictor::failure(pictor::ErrorCode::io_error, "failed writing metadata: " + path.string());
    return {};
}

int report(pictor::Status status) {
    pictor::logging::app().error("{}", status.message);
    return status.code == pictor::ErrorCode::invalid_argument ? 2 : 1;
}
} // namespace

int main(int argc, char** argv) {
    pictor::cli::Options options;
    if (const auto status = pictor::cli::parse({argv + 1, argv + argc}, options); !status) return report(status);
    if (options.help) { std::cout << pictor::cli::usage(); return 0; }
    if (options.version) { std::cout << "pictor 0.1.0 (sd.cpp 90e87bc)\n"; return 0; }

    // Check every destination before loading weights or starting an expensive generation.
    for (int i = 0; i < options.count; ++i) {
        std::filesystem::path path;
        if (const auto status = pictor::cli::output_path(options, i, path); !status) return report(status);
        auto metadata = path;
        metadata.replace_extension(".json");
        for (const auto& destination : {path, metadata}) {
            std::error_code ec;
            const auto state = std::filesystem::status(destination, ec);
            if (ec && ec != std::errc::no_such_file_or_directory)
                return report(pictor::failure(pictor::ErrorCode::io_error, "cannot inspect output: " + ec.message()));
            if (!options.overwrite && std::filesystem::exists(state))
                return report(pictor::failure(pictor::ErrorCode::invalid_argument, "output exists: " + destination.string() + " (use --overwrite)"));
            if (std::filesystem::is_directory(state))
                return report(pictor::failure(pictor::ErrorCode::invalid_argument, "output is a directory: " + destination.string()));
        }
    }

    std::int64_t base_seed;
    if (const auto status = pictor::resolve_seed(options.request.seed, base_seed); !status) return report(status);
    pictor::logging::app().info("Loading {}", options.session.model_path.string());
    std::unique_ptr<pictor::AnimaSession> session;
    if (const auto status = pictor::AnimaSession::create(options.session, session); !status) return report(status);
    pictor::logging::app().info("Model loaded in {:.2f}s", session->load_seconds());
    for (int i = 0; i < options.count; ++i) {
        options.request.seed = base_seed + i;
        pictor::logging::app().info("Generating {}/{} (seed {})", i + 1, options.count, options.request.seed);
        pictor::Image image;
        if (const auto status = session->generate(options.request, image, [](const pictor::Progress& progress, void*) noexcept {
                pictor::logging::app().info("Step {}/{}", progress.step, progress.steps);
            }); !status) return report(status);
        std::filesystem::path path;
        if (const auto status = pictor::cli::output_path(options, i, path); !status) return report(status);
        if (const auto status = pictor::write_png(path, image); !status) return report(status);
        auto metadata = path;
        metadata.replace_extension(".json");
        if (const auto status = write_metadata(metadata, options, image, session->load_seconds()); !status) return report(status);
        pictor::logging::app().info("Generated in {:.2f}s", image.generation_seconds);
        std::cout << path.string() << '\n';
    }
    return 0;
}
