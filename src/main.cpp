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
    const bool klein = options.model == pictor::cli::Model::flux_klein;
    std::filesystem::path text_encoder, vae;
    if (klein) {
        text_encoder = std::filesystem::absolute(options.text_encoder, ec);
        if (ec) return pictor::failure(pictor::ErrorCode::io_error, "cannot resolve text encoder path: " + ec.message());
        vae = std::filesystem::absolute(options.vae, ec);
        if (ec) return pictor::failure(pictor::ErrorCode::io_error, "cannot resolve VAE path: " + ec.message());
    }
    std::ofstream out(path);
    if (!out) return pictor::failure(pictor::ErrorCode::io_error, "cannot write metadata: " + path.string());
    const auto& request = options.request;
    const auto quote = pictor::cli::json_string;
    out << std::setprecision(9)
        << "{\n  \"pictor_version\": \"0.1.0\",\n"
        << "  \"sd_cpp_revision\": \"90e87bc846f17059771efb8aaa31e9ef0cab6f78\",\n"
        << "  \"model_type\": " << quote(klein ? "flux2-klein-4b" : "anima") << ",\n"
        << "  \"model\": " << quote(model.string()) << ",\n";
    if (klein) out << "  \"text_encoder\": " << quote(text_encoder.string()) << ",\n"
                   << "  \"vae\": " << quote(vae.string()) << ",\n";
    out << "  \"mode\": " << quote(options.reference_images.empty() ? "text-to-image" : "reference-edit") << ",\n";
    if (!options.reference_images.empty()) {
        out << "  \"reference_images\": [";
        for (std::size_t i = 0; i < options.reference_images.size(); ++i) {
            const auto reference = std::filesystem::absolute(options.reference_images[i], ec);
            if (ec) return pictor::failure(pictor::ErrorCode::io_error, "cannot resolve reference path: " + ec.message());
            out << (i ? ", " : "") << quote(reference.string());
        }
        out << "],\n  \"auto_resize_reference\": " << (options.auto_resize_reference ? "true" : "false") << ",\n";
    }
    out
        << "  \"prompt\": " << quote(request.prompt) << ",\n"
        << "  \"negative_prompt\": " << quote(request.negative_prompt) << ",\n"
        << "  \"width\": " << image.width << ",\n  \"height\": " << image.height << ",\n"
        << "  \"steps\": " << request.steps << ",\n  \"cfg_scale\": " << request.cfg_scale << ",\n"
        << "  \"seed\": " << image.seed << ",\n"
        << "  \"sampler\": " << quote(klein ? "euler" : "er_sde") << ",\n"
        << "  \"scheduler\": " << quote(klein ? "discrete" : "smoothstep") << ",\n"
        << "  \"cache\": " << quote(request.cache == pictor::CacheMode::spectrum ? "spectrum" : "none") << ",\n"
        << "  \"flash_attention\": " << (klein ? "false" : "true") << ",\n"
        << "  \"diffusion_flash_attention\": true,\n"
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

    // Decode once before loading weights; owners survive the entire output batch.
    std::vector<pictor::Image> reference_images(options.reference_images.size());
    pictor::FluxKleinEditRequest edit;
    edit.generation = options.request;
    edit.auto_resize = options.auto_resize_reference;
    for (std::size_t i = 0; i < reference_images.size(); ++i) {
        auto& ref = reference_images[i];
        if (const auto status = pictor::read_image(options.reference_images[i], ref); !status) return report(status);
        edit.reference_images.push_back({ref.width, ref.height, ref.pixels.data(), ref.pixels.size()});
    }
    if (!reference_images.empty())
        if (const auto status = pictor::validate_flux_klein_edit_request(edit); !status) return report(status);

    std::int64_t base_seed;
    if (const auto status = pictor::resolve_seed(options.request.seed, base_seed); !status) return report(status);
    pictor::logging::app().info("Loading {}", options.session.model_path.string());
    std::unique_ptr<pictor::AnimaSession> anima;
    std::unique_ptr<pictor::FluxKleinSession> klein;
    const auto created = options.model == pictor::cli::Model::anima
        ? pictor::AnimaSession::create(options.session, anima)
        : pictor::FluxKleinSession::create({options.session.model_path, options.text_encoder, options.vae,
                                          options.session.threads, options.session.verbose}, klein);
    if (!created) return report(created);
    const auto load_seconds = anima ? anima->load_seconds() : klein->load_seconds();
    pictor::logging::app().info("Model loaded in {:.2f}s", load_seconds);
    for (int i = 0; i < options.count; ++i) {
        options.request.seed = base_seed + i;
        pictor::logging::app().info("Generating {}/{} (seed {})", i + 1, options.count, options.request.seed);
        pictor::Image image;
        const auto progress = [](const pictor::Progress& value, void*) noexcept {
            pictor::logging::app().info("Step {}/{}", value.step, value.steps);
        };
        edit.generation = options.request;
        const auto generated = anima ? anima->generate(options.request, image, progress)
            : reference_images.empty() ? klein->generate(options.request, image, progress)
                                       : klein->edit(edit, image, progress);
        if (!generated) return report(generated);
        std::filesystem::path path;
        if (const auto status = pictor::cli::output_path(options, i, path); !status) return report(status);
        if (const auto status = pictor::write_png(path, image); !status) return report(status);
        auto metadata = path;
        metadata.replace_extension(".json");
        if (const auto status = write_metadata(metadata, options, image, load_seconds); !status) return report(status);
        pictor::logging::app().info("Generated in {:.2f}s", image.generation_seconds);
        std::cout << path.string() << '\n';
    }
    return 0;
}
