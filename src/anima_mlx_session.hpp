#pragma once
#include "pictor/anima.hpp"
namespace pictor::detail {
class AnimaMlxSession {
  public:
    static Status create(const SessionOptions &, std::unique_ptr<AnimaMlxSession> &) noexcept;
    ~AnimaMlxSession();
    Status generate_batch(const GenerationRequest &, int, BatchResult &, BatchProgressCallback, void *) noexcept;
    double load_seconds() const noexcept;

  private:
    struct Impl;
    explicit AnimaMlxSession(std::unique_ptr<Impl>) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace pictor::detail
