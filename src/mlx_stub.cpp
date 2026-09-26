#include "mlx_session.hpp"
namespace pictor::detail {
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
