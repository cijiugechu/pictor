#pragma once
#include "pictor/flux_klein.hpp"
namespace pictor::detail {
class MlxSession {
  public:
    static Status create(const FluxKleinOptions &, std::unique_ptr<MlxSession> &) noexcept;
    ~MlxSession();
    Status generate_batch(const GenerationRequest &, int, BatchResult &, BatchProgressCallback, void *,
                          const FluxKleinEditRequest * = nullptr) noexcept;
    double load_seconds() const noexcept;

  private:
    struct Impl;
    explicit MlxSession(std::unique_ptr<Impl>) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace pictor::detail
