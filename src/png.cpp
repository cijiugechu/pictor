#include "pictor/anima.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <limits>
#include <spdlog/fmt/fmt.h>

namespace pictor {

Status write_png(const std::filesystem::path& path, const Image& image) noexcept {
    if (image.width <= 0 || image.height <= 0 || image.channels != 3 ||
        image.width > std::numeric_limits<int>::max() / image.channels ||
        image.pixels.size() != static_cast<std::size_t>(image.width) * image.height * image.channels)
        return failure(ErrorCode::invalid_argument, "invalid RGB image buffer");
    if (path.empty()) return failure(ErrorCode::invalid_argument, "PNG path must not be empty");
    std::error_code ec;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return failure(ErrorCode::io_error, fmt::format("cannot create PNG directory: {}", ec.message()));
    if (!stbi_write_png(path.string().c_str(), image.width, image.height, image.channels,
                        image.pixels.data(), image.width * image.channels))
        return failure(ErrorCode::io_error, fmt::format("failed to write PNG: {}", path.string()));
    return {};
}

} // namespace pictor
