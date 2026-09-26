#include "pictor/flux_klein.hpp"
#include "session.hpp"
#include <new>

namespace pictor {
struct FluxKleinSession::Impl { std::unique_ptr<detail::Session> session; };

FluxKleinSession::FluxKleinSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
FluxKleinSession::~FluxKleinSession() = default;

Status FluxKleinSession::create(const FluxKleinOptions& options, std::unique_ptr<FluxKleinSession>& output) noexcept {
    if (output) return failure(ErrorCode::invalid_argument, "session output must be empty");
    auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!impl) return failure(ErrorCode::out_of_memory, "cannot allocate session");
    const auto status = detail::Session::create(detail::Model::flux_klein, {options.diffusion_model_path, options.text_encoder_path, options.vae_path}, options.threads, options.verbose, impl->session);
    if (!status) return status;
    auto* session = new (std::nothrow) FluxKleinSession(std::move(impl));
    if (!session) return failure(ErrorCode::out_of_memory, "cannot allocate session handle");
    output.reset(session);
    return {};
}

Status FluxKleinSession::generate(const GenerationRequest& request, Image& output, ProgressCallback progress, void* userdata) noexcept {
    return impl_->session->generate(request, output, progress, userdata);
}
double FluxKleinSession::load_seconds() const noexcept { return impl_->session->load_seconds(); }
Status FluxKleinSession::edit(const FluxKleinEditRequest& request, Image& output, ProgressCallback progress, void* userdata) noexcept {
    // Keep the caller's existing pixels alive if a reference views output itself.
    Image result;
    const auto status = impl_->session->generate(request.generation, result, progress, userdata, &request);
    output = std::move(result);
    return status;
}
} // namespace pictor
