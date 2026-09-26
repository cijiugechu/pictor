#pragma once

#include "pictor/flux_klein.hpp"
#include "stable-diffusion.h"
#include <memory>

namespace pictor::detail {

enum class Model { anima, flux_klein };
struct ModelFiles {
    std::filesystem::path model;
    std::filesystem::path text_encoder;
    std::filesystem::path vae;
};

// One implementation owns every backend context and the process-wide callback lock.
class Session {
public:
    static Status create(Model model, const ModelFiles& files, int threads, bool verbose,
                         std::unique_ptr<Session>& output) noexcept;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Status generate(const GenerationRequest& request, Image& image, ProgressCallback progress, void* userdata,
                    const FluxKleinEditRequest* edit = nullptr) noexcept;
    Status generate_batch(const GenerationRequest& request, int count, BatchResult& output,
                          BatchProgressCallback progress, void* userdata,
                          const FluxKleinEditRequest* edit = nullptr) noexcept;
    Status set_hidden_state_compression(bool enabled) noexcept;
    double load_seconds() const noexcept { return load_seconds_; }
private:
    explicit Session(Model model, bool verbose) noexcept : model_(model), verbose_(verbose) {}
    sd_ctx_t* context_ = nullptr;
    Model model_;
    bool verbose_;
    bool hidden_state_compression_ = false; // Protected by backend_mutex.
    double load_seconds_ = 0;
    std::string model_path_, text_encoder_path_, vae_path_;
};
} // namespace pictor::detail
