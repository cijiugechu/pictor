#pragma once

#include "pictor/status.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace pictor {

enum class Preset { fast, balanced, quality };
enum class CacheMode { none, spectrum };

struct GenerationRequest {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 768;
    int steps = 8;
    float cfg_scale = 1.0f;
    std::int64_t seed = -1;
    CacheMode cache = CacheMode::spectrum;
    bool vae_tiling = false;
};

GenerationRequest preset_request(Preset preset);
[[nodiscard]] Status validate_request(const GenerationRequest& request) noexcept;
[[nodiscard]] Status resolve_seed(std::int64_t seed, std::int64_t& output) noexcept;

struct SessionOptions {
    std::filesystem::path model_path;
    int threads = 0; // 0 selects sd.cpp's physical core count.
    bool verbose = false;
};

struct Progress {
    int step; // 0 is the initial event; final step equals steps.
    int steps;
    float seconds;
};
// Synchronous callback on the generating thread; must not re-enter pictor.
using ProgressCallback = void (*)(const Progress&, void* userdata) noexcept;

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<std::uint8_t> pixels;
    std::int64_t seed = 0;
    double generation_seconds = 0;
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
    double load_seconds() const noexcept;

private:
    struct Impl;
    explicit AnimaSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Status write_png(const std::filesystem::path& path, const Image& image) noexcept;

} // namespace pictor
