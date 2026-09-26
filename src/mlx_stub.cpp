#include "mlx_session.hpp"
#include "anima_mlx_session.hpp"
namespace pictor::detail {
struct AnimaMlxSession::Impl {};
AnimaMlxSession::~AnimaMlxSession() = default;
Status AnimaMlxSession::create(const SessionOptions &, std::unique_ptr<AnimaMlxSession> &) noexcept {
    return failure(ErrorCode::invalid_argument, "MLX is not built in; use ggml weights or build with -Dmlx=true on Apple Silicon");
}
Status AnimaMlxSession::generate_batch(const GenerationRequest &, int, BatchResult &output, BatchProgressCallback, void *) noexcept {
    output = {};
    return failure(ErrorCode::backend_error, "MLX is not built in");
}
double AnimaMlxSession::load_seconds() const noexcept { return 0; }
struct MlxSession::Impl {};
MlxSession::~MlxSession() = default;
Status MlxSession::create(const FluxKleinOptions &, std::unique_ptr<MlxSession> &) noexcept {
    return failure(ErrorCode::invalid_argument,
                   "MLX is not built in; use ggml weights or build with -Dmlx=true on Apple Silicon");
}
Status MlxSession::generate_batch(const GenerationRequest &, int, BatchResult &output, BatchProgressCallback, void *,
                                  const FluxKleinEditRequest *) noexcept {
    output = {};
    return failure(ErrorCode::backend_error, "MLX is not built in");
}
double MlxSession::load_seconds() const noexcept {
    return 0;
}
} // namespace pictor::detail
