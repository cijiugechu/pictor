#include "cli.hpp"
#include "logging.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <spdlog/fmt/fmt.h>

namespace {
pictor::Status write_metadata(const std::filesystem::path& path, const pictor::cli::Options& options,
                    const pictor::Image& image, double load_seconds, int batch_count, double batch_seconds) noexcept {
    std::error_code ec;
    const auto model = std::filesystem::absolute(options.session.model_path, ec);
    if (ec) return pictor::failure(pictor::ErrorCode::io_error, fmt::format("cannot resolve model path: {}", ec.message()));
    const bool klein = options.model == pictor::cli::Model::flux_klein;
    std::filesystem::path text_encoder, vae;
    if (klein) {
        text_encoder = std::filesystem::absolute(options.text_encoder, ec);
        if (ec) return pictor::failure(pictor::ErrorCode::io_error, fmt::format("cannot resolve text encoder path: {}", ec.message()));
        vae = std::filesystem::absolute(options.vae, ec);
        if (ec) return pictor::failure(pictor::ErrorCode::io_error, fmt::format("cannot resolve VAE path: {}", ec.message()));
    }
    std::ofstream out(path);
    if (!out) return pictor::failure(pictor::ErrorCode::io_error, fmt::format("cannot write metadata: {}", path.string()));
    const auto& request = options.request;
    const auto quote = pictor::cli::json_string;
    out << fmt::format("{{\n  \"pictor_version\": \"0.1.0\",\n"
                       "  \"sd_cpp_revision\": \"90e87bc846f17059771efb8aaa31e9ef0cab6f78\",\n"
                       "  \"model_type\": {},\n  \"model\": {},\n",
                       quote(klein ? "flux2-klein-4b" : "anima"), quote(model.string()));
    if (klein) out << fmt::format("  \"text_encoder\": {},\n  \"vae\": {},\n",
                                  quote(text_encoder.string()), quote(vae.string()));
    const bool mlx = klein ? options.backend == pictor::KleinBackend::mlx : options.anima_backend == pictor::AnimaBackend::mlx;
    out << fmt::format("  \"backend\": {},\n  \"mode\": {},\n",
                       quote(mlx ? "mlx" : "ggml"), quote(options.reference_images.empty() ? "text-to-image" : "reference-edit"));
    if (!options.reference_images.empty()) {
        out << "  \"reference_images\": [";
        for (std::size_t i = 0; i < options.reference_images.size(); ++i) {
            const auto reference = std::filesystem::absolute(options.reference_images[i], ec);
            if (ec) return pictor::failure(pictor::ErrorCode::io_error, fmt::format("cannot resolve reference path: {}", ec.message()));
            out << fmt::format("{}{}", i ? ", " : "", quote(reference.string()));
        }
        out << fmt::format("],\n  \"auto_resize_reference\": {},\n", options.auto_resize_reference);
    }
    out << fmt::format("  \"prompt\": {},\n  \"negative_prompt\": {},\n"
                       "  \"width\": {},\n  \"height\": {},\n"
                       "  \"steps\": {},\n  \"cfg_scale\": {:.9g},\n  \"seed\": {},\n",
                       quote(request.prompt), quote(request.negative_prompt), image.width, image.height,
                       request.steps, request.cfg_scale, image.seed);
    out << fmt::format("  \"sampler\": {},\n  \"scheduler\": {},\n  \"cache\": {},\n"
                       "  \"flash_attention\": {},\n  \"diffusion_flash_attention\": true,\n"
                       "  \"vae_tiling\": {},\n",
                       quote(klein ? "euler" : "er_sde"), quote(klein ? "discrete" : "smoothstep"),
                       quote(request.cache == pictor::CacheMode::spectrum ? "spectrum" : "none"),
                       !klein || mlx, request.vae_tiling);
    out << fmt::format("  \"load_seconds\": {:.9g},\n  \"hidden_state_compression\": {},\n"
                       "  \"batch_count\": {},\n  \"batch_generation_seconds\": {:.9g},\n"
                       "  \"generation_seconds_kind\": \"batch_average\",\n"
                       "  \"generation_seconds\": {:.9g}\n}}\n",
                       load_seconds, options.hidden_state_compression, batch_count, batch_seconds, image.generation_seconds);
    out.close();
    if (!out) return pictor::failure(pictor::ErrorCode::io_error, fmt::format("failed writing metadata: {}", path.string()));
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
                return report(pictor::failure(pictor::ErrorCode::io_error, fmt::format("cannot inspect output: {}", ec.message())));
            if (!options.overwrite && std::filesystem::exists(state))
                return report(pictor::failure(pictor::ErrorCode::invalid_argument, fmt::format("output exists: {} (use --overwrite)", destination.string())));
            if (std::filesystem::is_directory(state))
                return report(pictor::failure(pictor::ErrorCode::invalid_argument, fmt::format("output is a directory: {}", destination.string())));
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
        ? pictor::AnimaSession::create(options.session, options.anima_backend, anima)
        : pictor::FluxKleinSession::create({options.session.model_path, options.text_encoder, options.vae,
                                          options.session.threads, options.session.verbose}, options.backend, klein);
    if (!created) return report(created);
    if (klein) options.backend = klein->backend();
    if (anima) options.anima_backend = anima->backend();
    if (klein)
        if (const auto status = klein->set_hidden_state_compression(options.hidden_state_compression); !status) return report(status);
    const auto load_seconds = anima ? anima->load_seconds() : klein->load_seconds();
    pictor::logging::app().info("Model loaded in {:.2f}s", load_seconds);
    const int batch_limit = std::min(pictor::max_batch_count, static_cast<int>(pictor::max_batch_pixels /
        (static_cast<std::size_t>(options.request.width) * options.request.height)));
    for (int offset = 0; offset < options.count;) {
        const int count = std::min(batch_limit, options.count - offset);
        options.request.seed = base_seed + offset;
        pictor::logging::app().info("Generating batch: images {}..{} of {}", offset + 1, offset + count, options.count);
        struct ProgressContext { int offset; int total; } context{offset, options.count};
        const auto progress = [](const pictor::BatchProgress& value, void* data) noexcept {
            const auto& context = *static_cast<ProgressContext*>(data);
            pictor::logging::app().info("Image {}/{}: step {}/{}", context.offset + value.image_index + 1,
                                       context.total, value.sampling.step, value.sampling.steps);
        };
        pictor::BatchResult batch;
        edit.generation = options.request;
        const auto generated = anima ? anima->generate_batch(options.request, count, batch, progress, &context)
            : reference_images.empty() ? klein->generate_batch(options.request, count, batch, progress, &context)
                                       : klein->edit_batch(edit, count, batch, progress, &context);
        if (!generated) return report(generated);
        for (int i = 0; i < count; ++i) {
            const auto& image = batch.images[i];
            std::filesystem::path path;
            if (const auto status = pictor::cli::output_path(options, offset + i, path); !status) return report(status);
            if (const auto status = pictor::write_png(path, image); !status) return report(status);
            auto metadata = path;
            metadata.replace_extension(".json");
            if (const auto status = write_metadata(metadata, options, image, load_seconds, count, batch.generation_seconds); !status)
                return report(status);
            std::cout << path.string() << '\n';
        }
        pictor::logging::app().info("Batch generated in {:.2f}s ({:.2f}s/image amortized)",
                                   batch.generation_seconds, batch.generation_seconds / count);
        offset += count;
    }
    return 0;
}
