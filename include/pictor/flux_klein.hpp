#pragma once

#include "pictor/types.hpp"
#include <memory>

namespace pictor {

struct FluxKleinOptions {
    std::filesystem::path diffusion_model_path;
    std::filesystem::path text_encoder_path;
    std::filesystem::path vae_path;
    int threads = 0;
    bool verbose = false;
};

// Klein 4B distilled: 512x512, 4 steps, CFG 1, no cache.
GenerationRequest flux_klein_request();
[[nodiscard]] Status validate_flux_klein_request(const GenerationRequest& request) noexcept;

struct FluxKleinEditRequest {
    GenerationRequest generation = flux_klein_request();
    std::vector<ImageView> reference_images; // 1..4 ordered references
    bool auto_resize = true;
};
[[nodiscard]] Status validate_flux_klein_edit_request(const FluxKleinEditRequest& request) noexcept;

// Text-to-image and reference editing with resident split weights and full VAE.
// Shares the backend lock with Anima.
// The same output ownership and callback rules as AnimaSession apply.
class FluxKleinSession {
public:
    [[nodiscard]] static Status create(const FluxKleinOptions& options, std::unique_ptr<FluxKleinSession>& output) noexcept;
    ~FluxKleinSession();
    FluxKleinSession(const FluxKleinSession&) = delete;
    FluxKleinSession& operator=(const FluxKleinSession&) = delete;
    [[nodiscard]] Status generate(const GenerationRequest& request, Image& output,
                                  ProgressCallback progress = nullptr, void* userdata = nullptr) noexcept;
    double load_seconds() const noexcept;
    [[nodiscard]] Status edit(const FluxKleinEditRequest& request, Image& output,
                              ProgressCallback progress = nullptr, void* userdata = nullptr) noexcept;
private:
    struct Impl;
    explicit FluxKleinSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace pictor
