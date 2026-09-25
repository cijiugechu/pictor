#include "pictor/anima.hpp"

#include <cmath>
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
