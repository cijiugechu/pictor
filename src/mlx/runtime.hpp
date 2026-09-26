#pragma once
#include "logging.hpp"
#include "pictor/status.hpp"
#include "stable-diffusion.h"
#include <filesystem>
#include <mlx/mlx.h>
#include <stdexcept>
namespace pictor::mlx_backend {
namespace mx = mlx::core;
struct RuntimeScope {
    mx::Device previous = mx::default_device();
    RuntimeScope() { mx::set_default_device(mx::Device(mx::Device::gpu)); }
    ~RuntimeScope() { mx::set_default_device(previous); }
};
struct TokenizerLogScope {
    TokenizerLogScope() {
        sd_set_log_callback(
            [](sd_log_level_t level, const char *text, void *) {
                if (level >= SD_LOG_WARN && text)
                    logging::mlx().warn("{}", text);
            },
            nullptr);
    }
    ~TokenizerLogScope() { sd_set_log_callback(nullptr, nullptr); }
};
inline Status exception_status() noexcept {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return failure(ErrorCode::out_of_memory, "MLX allocation failed");
    } catch (const std::filesystem::filesystem_error &e) {
        return failure(ErrorCode::io_error, e.what());
    } catch (const std::invalid_argument &e) {
        return failure(ErrorCode::invalid_argument, e.what());
    } catch (const std::exception &e) {
        return failure(ErrorCode::backend_error, e.what());
    } catch (...) {
        return failure(ErrorCode::backend_error, "unknown MLX runtime failure");
    }
}
} // namespace pictor::mlx_backend
