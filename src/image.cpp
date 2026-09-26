#include "pictor/types.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_MAX_DIMENSIONS 4096
#include "stb_image.h"

#include <cstdio>
#include <memory>
#include <spdlog/fmt/fmt.h>

namespace pictor {
Status read_image(const std::filesystem::path& path, Image& output) noexcept {
    output = {};
    if (path.empty()) return failure(ErrorCode::invalid_argument, "image path must not be empty");
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec))
        return failure(ErrorCode::io_error, fmt::format("image file not found or inaccessible: {}", path.string()));
    auto file = std::unique_ptr<FILE, decltype(&std::fclose)>(std::fopen(path.string().c_str(), "rb"), std::fclose);
    if (!file) return failure(ErrorCode::io_error, fmt::format("cannot open image: {}", path.string()));
    int width = 0, height = 0, channels = 0;
    if (!stbi_info_from_file(file.get(), &width, &height, &channels) ||
        width < 1 || height < 1 || width > 4096 || height > 4096)
        return failure(ErrorCode::invalid_argument, "expected a PNG/JPEG with dimensions 1..4096");
    auto pixels = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>(
        stbi_load_from_file(file.get(), &width, &height, &channels, 3), stbi_image_free);
    if (!pixels) return failure(ErrorCode::invalid_argument, fmt::format("cannot decode PNG/JPEG: {}", path.string()));
    const auto size = static_cast<std::size_t>(width) * height * 3;
    output = {width, height, 3, {pixels.get(), pixels.get() + size}, 0, 0};
    return {};
}
} // namespace pictor
