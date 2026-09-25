#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

namespace pictor {

enum class ErrorCode : std::int32_t {
    ok = 0,
    invalid_argument = 1,
    io_error = 2,
    backend_error = 3,
    out_of_memory = 4,
};

// No allocation is needed to return an error. Long messages are truncated.
struct Status {
    ErrorCode code = ErrorCode::ok;
    char message[512]{};
    bool ok() const noexcept { return code == ErrorCode::ok; }
    explicit operator bool() const noexcept { return ok(); }
};

inline Status failure(ErrorCode code, std::string_view message) noexcept {
    Status status;
    status.code = code;
    const auto length = message.size() < sizeof(status.message) - 1 ? message.size() : sizeof(status.message) - 1;
    if (length) std::memcpy(status.message, message.data(), length);
    return status;
}

} // namespace pictor
