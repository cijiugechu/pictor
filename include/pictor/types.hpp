#pragma once

#include "pictor/status.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace pictor {

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

[[nodiscard]] Status validate_request(const GenerationRequest& request) noexcept;
[[nodiscard]] Status resolve_seed(std::int64_t seed, std::int64_t& output) noexcept;

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

// Borrowed, tightly packed RGB8; bytes remain valid for the complete edit call.
struct ImageView {
    int width = 0;
    int height = 0;
    const std::uint8_t* pixels = nullptr;
    std::size_t pixels_len = 0;
};
[[nodiscard]] Status validate_image_view(const ImageView& image) noexcept;
// Decode PNG/JPEG to RGB8; alpha is discarded, EXIF orientation is not applied.
[[nodiscard]] Status read_image(const std::filesystem::path& path, Image& output) noexcept;

[[nodiscard]] Status write_png(const std::filesystem::path& path, const Image& image) noexcept;

} // namespace pictor
