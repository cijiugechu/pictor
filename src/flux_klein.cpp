#include "pictor/flux_klein.hpp"
#include "session.hpp"
#include "mlx_session.hpp"
#include <new>

namespace pictor {
struct FluxKleinSession::Impl {
    std::unique_ptr<detail::Session> session;
    std::unique_ptr<detail::MlxSession> mlx;
    Status batch(const GenerationRequest& r, int count, BatchResult& output, BatchProgressCallback p, void* u,
                 const FluxKleinEditRequest* edit = nullptr) noexcept {
        return mlx ? mlx->generate_batch(r,count,output,p,u,edit) : session->generate_batch(r,count,output,p,u,edit);
    }
    Status generate(const GenerationRequest& r, Image& output, ProgressCallback p, void* u,
                    const FluxKleinEditRequest* edit = nullptr) noexcept {
        output = {};
        struct Adapter { ProgressCallback callback; void* userdata; } adapter{p,u};
        const auto callback = [](const BatchProgress& value, void* data) noexcept {
            const auto& a = *static_cast<Adapter*>(data);
            a.callback(value.sampling,a.userdata);
        };
        BatchResult result;
        const auto status = batch(r,1,result,p ? callback : static_cast<BatchProgressCallback>(nullptr),&adapter,edit);
        if (status) output = std::move(result.images.front());
        return status;
    }
};

FluxKleinSession::FluxKleinSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
FluxKleinSession::~FluxKleinSession() = default;

Status FluxKleinSession::create(const FluxKleinOptions& options, std::unique_ptr<FluxKleinSession>& output) noexcept {
    return create(options,KleinBackend::automatic,output);
}
Status FluxKleinSession::create(const FluxKleinOptions& options, KleinBackend selected,
                               std::unique_ptr<FluxKleinSession>& output) noexcept {
    if (output) return failure(ErrorCode::invalid_argument, "session output must be empty");
    auto impl = std::unique_ptr<Impl>(new (std::nothrow) Impl);
    if (!impl) return failure(ErrorCode::out_of_memory, "cannot allocate session");
    if (selected == KleinBackend::automatic) {
        std::error_code ec;
        selected = std::filesystem::is_directory(options.diffusion_model_path,ec) ? KleinBackend::mlx : KleinBackend::ggml;
    }
    if (selected != KleinBackend::ggml && selected != KleinBackend::mlx)
        return failure(ErrorCode::invalid_argument, "invalid Klein backend");
    const auto status = selected == KleinBackend::mlx ? detail::MlxSession::create(options,impl->mlx)
        : detail::Session::create(detail::Model::flux_klein, {options.diffusion_model_path, options.text_encoder_path, options.vae_path}, options.threads, options.verbose, impl->session);
    if (!status) return status;
    auto* session = new (std::nothrow) FluxKleinSession(std::move(impl));
    if (!session) return failure(ErrorCode::out_of_memory, "cannot allocate session handle");
    output.reset(session);
    return {};
}

Status FluxKleinSession::generate(const GenerationRequest& request, Image& output, ProgressCallback progress, void* userdata) noexcept {
    return impl_->generate(request, output, progress, userdata);
}
Status FluxKleinSession::generate_batch(const GenerationRequest& request, int count, BatchResult& output,
                                      BatchProgressCallback progress, void* userdata) noexcept {
    return impl_->batch(request, count, output, progress, userdata);
}
Status FluxKleinSession::set_hidden_state_compression(bool enabled) noexcept {
    if (impl_->mlx) return enabled
        ? failure(ErrorCode::invalid_argument, "hidden-state compression requires the ggml backend") : Status{};
    return impl_->session->set_hidden_state_compression(enabled);
}
double FluxKleinSession::load_seconds() const noexcept { return impl_->mlx ? impl_->mlx->load_seconds() : impl_->session->load_seconds(); }
KleinBackend FluxKleinSession::backend() const noexcept { return impl_->mlx ? KleinBackend::mlx : KleinBackend::ggml; }
Status FluxKleinSession::edit(const FluxKleinEditRequest& request, Image& output, ProgressCallback progress, void* userdata) noexcept {
    // Keep the caller's existing pixels alive if a reference views output itself.
    Image result;
    const auto status = impl_->generate(request.generation, result, progress, userdata, &request);
    output = std::move(result);
    return status;
}
Status FluxKleinSession::edit_batch(const FluxKleinEditRequest& request, int count, BatchResult& output,
                                  BatchProgressCallback progress, void* userdata) noexcept {
    BatchResult result; // References may borrow pixels from the previous output.
    const auto status = impl_->batch(request.generation, count, result, progress, userdata, &request);
    output = std::move(result);
    return status;
}
} // namespace pictor
