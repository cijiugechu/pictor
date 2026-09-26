#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"

#include <cmath>
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace pictor {

GenerationRequest preset_request(Preset preset) {
    GenerationRequest request;
    switch (preset) {
    case Preset::fast: request.steps = 3; break;
    case Preset::balanced: break;
    case Preset::quality:
        request.steps = 16;
        request.cache = CacheMode::none;
        break;
    }
    return request;
}

GenerationRequest anima_request(Preset preset) {
    auto request = preset_request(preset);
    request.cache = CacheMode::none;
    return request;
}

GenerationRequest flux_klein_request() {
    GenerationRequest request;
    request.height = 512;
    request.steps = 4;
    request.cache = CacheMode::none;
    return request;
}

Status validate_flux_klein_request(const GenerationRequest& request) noexcept {
    if (const auto status = validate_request(request); !status) return status;
    if (request.cache != CacheMode::none)
        return failure(ErrorCode::invalid_argument, "Klein text-to-image currently requires cache=none");
    return {};
}

Status validate_image_view(const ImageView& image) noexcept {
    if (image.width < 1 || image.height < 1 || image.width > 4096 || image.height > 4096 ||
        !image.pixels || image.pixels_len != static_cast<std::size_t>(image.width) * image.height * 3)
        return failure(ErrorCode::invalid_argument, "reference must be tightly packed RGB8, with dimensions 1..4096 and matching buffer length");
    return {};
}

Status validate_flux_klein_edit_request(const FluxKleinEditRequest& request) noexcept {
    if (const auto status = validate_flux_klein_request(request.generation); !status) return status;
    if (request.reference_images.empty() || request.reference_images.size() > 4)
        return failure(ErrorCode::invalid_argument, "Klein editing requires 1..4 reference images");
    for (const auto& image : request.reference_images) {
        if (const auto status = validate_image_view(image); !status) return status;
        if (!request.auto_resize) {
            if (image.width % 16 || image.height % 16)
                return failure(ErrorCode::invalid_argument, "reference dimensions must be multiples of 16 when auto-resize is disabled");
        } else {
            const double area = std::min(1024 * 1024, request.generation.width * request.generation.height);
            const double width = std::sqrt(area * image.width / image.height);
            const double height = width * image.height / image.width;
            const double resized_width = std::round(width / 16) * 16;
            const double resized_height = std::round(height / 16) * 16;
            if (resized_width < 16 || resized_height < 16 || resized_width > 4096 || resized_height > 4096)
                return failure(ErrorCode::invalid_argument, "reference aspect ratio produces unsupported resized dimensions");
        }
    }
    return {};
}

Status validate_request(const GenerationRequest& request) noexcept {
    if (request.prompt.find_first_not_of(" \t\r\n") == std::string::npos)
        return failure(ErrorCode::invalid_argument, "prompt must not be empty");
    if (request.width < 64 || request.height < 64 || request.width > 4096 || request.height > 4096 ||
        request.width % 16 != 0 || request.height % 16 != 0)
        return failure(ErrorCode::invalid_argument, "width and height must be multiples of 16 between 64 and 4096");
    if (request.steps < 1 || request.steps > 1000)
        return failure(ErrorCode::invalid_argument, "steps must be between 1 and 1000");
    if (!std::isfinite(request.cfg_scale) || request.cfg_scale < 0 || request.cfg_scale > 100)
        return failure(ErrorCode::invalid_argument, "cfg-scale must be finite and between 0 and 100");
    if (request.seed < -1)
        return failure(ErrorCode::invalid_argument, "seed must be -1 (random) or a nonnegative integer");
    if (request.cache != CacheMode::none && request.cache != CacheMode::spectrum)
        return failure(ErrorCode::invalid_argument, "unsupported cache mode");
    return {};
}

Status validate_batch_request(const GenerationRequest& request, int count) noexcept {
    if (const auto status = validate_request(request); !status) return status;
    if (count < 1 || count > max_batch_count)
        return failure(ErrorCode::invalid_argument, "batch count must be between 1 and 8");
    if (static_cast<std::size_t>(request.width) * request.height * count > max_batch_pixels)
        return failure(ErrorCode::invalid_argument, "batch output exceeds 16 megapixels");
    if (request.seed > INT64_MAX - (count - 1))
        return failure(ErrorCode::invalid_argument, "seed sequence would overflow");
    return {};
}

Status resolve_seed(std::int64_t seed, std::int64_t& output) noexcept {
    output = 0;
    if (seed < -1) return failure(ErrorCode::invalid_argument, "invalid seed");
    if (seed >= 0) { output = seed; return {}; }
    // macOS/Linux: use the OS RNG without std::random_device's throwing API.
    const int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return failure(ErrorCode::io_error, "cannot open system random source");
    std::uint32_t value = 0;
    auto* cursor = reinterpret_cast<unsigned char*>(&value);
    std::size_t remaining = sizeof(value);
    while (remaining) {
        const auto count = read(fd, cursor, remaining);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            close(fd);
            return failure(ErrorCode::io_error, "cannot read system random source");
        }
        remaining -= static_cast<std::size_t>(count);
        cursor += count;
    }
    close(fd);
    output = value;
    return {};
}

} // namespace pictor
