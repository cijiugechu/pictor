#include "pictor/anima.hpp"
#include "session.hpp"
#include "anima_mlx_session.hpp"
#include <new>

namespace pictor {
struct AnimaSession::Impl {
    std::unique_ptr<detail::Session> session;
    std::unique_ptr<detail::AnimaMlxSession> mlx;
};

AnimaSession::AnimaSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
AnimaSession::~AnimaSession() = default;

Status AnimaSession::create(const SessionOptions& options, std::unique_ptr<AnimaSession>& output) noexcept {
    return create(options, AnimaBackend::automatic, output);
}
Status AnimaSession::create(const SessionOptions& options, AnimaBackend selected, std::unique_ptr<AnimaSession>& output) noexcept {
    if (output) return failure(ErrorCode::invalid_argument, "session output must be empty");
    if (selected == AnimaBackend::automatic) {
        std::error_code ec;
        selected = std::filesystem::is_directory(options.model_path, ec) ? AnimaBackend::mlx : AnimaBackend::ggml;
    }
    if (selected != AnimaBackend::mlx && selected != AnimaBackend::ggml)
        return failure(ErrorCode::invalid_argument, "invalid Anima backend");
    auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!impl) return failure(ErrorCode::out_of_memory, "cannot allocate session");
    const auto status = selected == AnimaBackend::mlx ? detail::AnimaMlxSession::create(options, impl->mlx)
        : detail::Session::create(detail::Model::anima, {options.model_path, {}, {}}, options.threads, options.verbose, impl->session);
    if (!status) return status;
    auto* session = new (std::nothrow) AnimaSession(std::move(impl));
    if (!session) return failure(ErrorCode::out_of_memory, "cannot allocate session handle");
    output.reset(session);
    return {};
}

Status AnimaSession::generate(const GenerationRequest& request, Image& output, ProgressCallback progress, void* userdata) noexcept {
    if (!impl_->mlx) return impl_->session->generate(request, output, progress, userdata);
    output = {};
    struct Adapter { ProgressCallback callback; void* userdata; } adapter{progress, userdata};
    const auto callback = [](const BatchProgress& value, void* data) noexcept {
        const auto& a = *static_cast<Adapter*>(data);
        a.callback(value.sampling, a.userdata);
    };
    BatchResult result;
    const auto status = impl_->mlx->generate_batch(request, 1, result,
        progress ? callback : static_cast<BatchProgressCallback>(nullptr), &adapter);
    if (status) output = std::move(result.images.front());
    return status;
}
Status AnimaSession::generate_batch(const GenerationRequest& request, int count, BatchResult& output,
                                      BatchProgressCallback progress, void* userdata) noexcept {
    return impl_->mlx ? impl_->mlx->generate_batch(request, count, output, progress, userdata)
                      : impl_->session->generate_batch(request, count, output, progress, userdata);
}
double AnimaSession::load_seconds() const noexcept { return impl_->mlx ? impl_->mlx->load_seconds() : impl_->session->load_seconds(); }
AnimaBackend AnimaSession::backend() const noexcept { return impl_->mlx ? AnimaBackend::mlx : AnimaBackend::ggml; }
} // namespace pictor
