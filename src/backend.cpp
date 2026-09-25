#include "backend.hpp"

#include <exception>
#include <new>

namespace pictor::backend {
namespace {
template<class Function>
Status guard(Function function) noexcept {
    try {
        return function();
    } catch (const std::bad_alloc&) {
        return failure(ErrorCode::out_of_memory, "backend allocation failed");
    } catch (const std::exception& error) {
        return failure(ErrorCode::backend_error, error.what());
    } catch (...) {
        return failure(ErrorCode::backend_error, "unknown backend exception");
    }
}
} // namespace

Status create(const sd_ctx_params_t& params, sd_ctx_t*& output) noexcept {
    output = nullptr;
    return guard([&] {
        output = new_sd_ctx(&params);
        return output ? Status{} : failure(ErrorCode::backend_error, "failed to load Anima model (enable verbose for backend logs)");
    });
}

Status generate(sd_ctx_t* context, const sd_img_gen_params_t& params, sd_image_t*& output) noexcept {
    output = nullptr;
    return guard([&] {
        output = generate_image(context, &params);
        return output && output->data ? Status{} : failure(ErrorCode::backend_error, "Anima generation failed (enable verbose for backend logs)");
    });
}

Status destroy(sd_ctx_t* context) noexcept {
    return guard([&] { free_sd_ctx(context); return Status{}; });
}
} // namespace pictor::backend
