#include "backend.hpp"
#include <cstdio>
#include <cstring>

#define CHECK(expression) do { if (!(expression)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); return 1; } } while (0)

int main() {
    using namespace pictor;
    sd_ctx_t* context = nullptr;
    sd_ctx_params_t params{};
    CHECK(backend::create(params, context).code == ErrorCode::backend_error && !context);
    params.n_threads = 1;
    CHECK(backend::create(params, context).code == ErrorCode::out_of_memory && !context);
    params.n_threads = 2;
    CHECK(backend::create(params, context).code == ErrorCode::backend_error && !context);
    params.n_threads = 3;
    const auto status = backend::create(params, context);
    CHECK(status.code == ErrorCode::backend_error && std::strlen(status.message) == 511);
    sd_image_t* image = nullptr;
    sd_img_gen_params_t request{};
    CHECK(backend::generate(context, request, image).code == ErrorCode::backend_error && !image);
    CHECK(backend::destroy(context).code == ErrorCode::backend_error);
    std::puts("PASS: upstream exceptions converted before entering no-exception code");
}
