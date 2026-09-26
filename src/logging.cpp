#include "logging.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace pictor::logging {
namespace {
std::shared_ptr<spdlog::logger> make_logger(const char* name, spdlog::level::level_enum level) {
    auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>(name, std::move(sink));
    logger->set_pattern("[%H:%M:%S] [%n] [%^%l%$] %v");
    logger->set_level(level);
    return logger;
}
} // namespace

spdlog::logger& app() {
    static auto logger = make_logger("pictor", spdlog::level::info);
    return *logger;
}

spdlog::logger& backend() {
    static auto logger = make_logger("sd.cpp", spdlog::level::debug);
    return *logger;
}
spdlog::logger& mlx() {
    static auto logger = make_logger("mlx", spdlog::level::info);
    return *logger;
}
} // namespace pictor::logging
