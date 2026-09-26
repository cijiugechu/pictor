#pragma once
#include <mutex>
namespace pictor::detail {
// Serialize both runtimes, including sd.cpp's process-wide tokenizer callbacks.
std::mutex &inference_mutex() noexcept;
} // namespace pictor::detail
