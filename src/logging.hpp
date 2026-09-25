#pragma once

#include <spdlog/spdlog.h>

namespace pictor::logging {
// Private named loggers: do not replace an embedding application's default logger.
spdlog::logger& app();
spdlog::logger& backend();
} // namespace pictor::logging
