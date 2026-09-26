#pragma once

#include "pictor/types.hpp"
#include <memory>

namespace pictor {

enum class Preset { fast, balanced, quality };
GenerationRequest preset_request(Preset preset);

struct SessionOptions {
    std::filesystem::path model_path;
    int threads = 0; // 0 selects sd.cpp's physical core count.
    bool verbose = false;
};

// Owns resident model weights. Backend calls across sessions are serialized because
// this sd.cpp revision uses global callbacks. Callbacks must not re-enter the API.
class AnimaSession {
public:
    // output must be empty. On failure it remains empty.
    [[nodiscard]] static Status create(const SessionOptions& options, std::unique_ptr<AnimaSession>& output) noexcept;
    ~AnimaSession();
    AnimaSession(const AnimaSession&) = delete;
    AnimaSession& operator=(const AnimaSession&) = delete;

    // output is cleared before each attempt, including failed requests.
    [[nodiscard]] Status generate(const GenerationRequest& request, Image& output,
                                  ProgressCallback progress = nullptr, void* userdata = nullptr) noexcept;
    // Same settings, consecutive seeds; -1 resolves one random starting seed.
    [[nodiscard]] Status generate_batch(const GenerationRequest& request, int count, BatchResult& output,
                                       BatchProgressCallback progress = nullptr, void* userdata = nullptr) noexcept;
    double load_seconds() const noexcept;

private:
    struct Impl;
    explicit AnimaSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace pictor
