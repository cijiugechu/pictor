#include "pictor/anima.hpp"
#include "session.hpp"
#include <new>

namespace pictor {
struct AnimaSession::Impl { std::unique_ptr<detail::Session> session; };

AnimaSession::AnimaSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
AnimaSession::~AnimaSession() = default;

Status AnimaSession::create(const SessionOptions& options, std::unique_ptr<AnimaSession>& output) noexcept {
    if (output) return failure(ErrorCode::invalid_argument, "session output must be empty");
    auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!impl) return failure(ErrorCode::out_of_memory, "cannot allocate session");
    const auto status = detail::Session::create(detail::Model::anima, {options.model_path, {}, {}}, options.threads, options.verbose, impl->session);
    if (!status) return status;
    auto* session = new (std::nothrow) AnimaSession(std::move(impl));
    if (!session) return failure(ErrorCode::out_of_memory, "cannot allocate session handle");
    output.reset(session);
    return {};
}

Status AnimaSession::generate(const GenerationRequest& request, Image& output, ProgressCallback progress, void* userdata) noexcept {
    return impl_->session->generate(request, output, progress, userdata);
}
double AnimaSession::load_seconds() const noexcept { return impl_->session->load_seconds(); }
} // namespace pictor
