#include "flux_hs.hpp"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstring>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(int argc, char** argv) {
    CHECK(!FluxHS::active(false, 1, 4));
    CHECK(!FluxHS::active(true, 1, 1));
    CHECK(!FluxHS::active(true, 0, 4));
    CHECK(FluxHS::active(true, 1, 4) && FluxHS::active(true, 3, 4));
    CHECK(!FluxHS::active(true, 4, 4));
    // A rectangular output, two differently shaped references, and odd border cells.
    const std::vector<FluxHS::Grid> grids{{4, 2}, {3, 5}, {1, 3}};
    const auto plan = FluxHS::make_plan(3, grids);
    CHECK(plan.restore.size() == 26 && plan.pool[0].size() == 10);
    CHECK(plan.positions.size() == 13 && plan.positions[0] == 0 && plan.positions[2] == 2);
    CHECK(plan.positions[3] == 8); // Text offset 3 + output cell center y=1,x=1.
    CHECK(plan.pool[0][2] == 8 && plan.pool[3][2] == 12); // First reference.
    CHECK(plan.restore[8] == 2 && plan.restore[23] == 8); // No cross-image pooling.
    ggml_init_params init{16 * 1024 * 1024, nullptr, false};
    auto ctx = ggml_init(init);
    CHECK(ctx);
    constexpr int channels = 256, text = 3, images = 26, low_images = 10;
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, text + images);
    auto pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 2, text + images);
    auto* input = static_cast<float*>(x->data);
    for (int token = 0; token < text + images; ++token)
        for (int c = 0; c < channels; ++c) input[token * channels + c] = token * 10 + c;
    for (int i = 0; i < ggml_nelements(pe); ++i) static_cast<float*>(pe->data)[i] = i;
    const auto graph = FluxHS::make_graph(ctx, plan, pe, [](ggml_tensor* dst, const void* src) {
        std::memcpy(dst->data, src, ggml_nbytes(dst));
    });
    auto low = graph.reduce(ctx, x);
    // Simulate a learned update varying by low-resolution token and channel.
    auto update = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, text + low_images);
    for (int token = 0; token < text + low_images; ++token)
        for (int c = 0; c < channels; ++c) static_cast<float*>(update->data)[token * channels + c] = (token + c + 1) * 0.25f;
    auto updated = ggml_add(ctx, low, update);
    auto expanded = graph.expand_residual(ctx, x, low, updated);
    auto identity = graph.expand_residual(ctx, x, low, low);
    auto forward = ggml_new_graph(ctx);
    for (auto result : {low, graph.pe, expanded, identity}) ggml_build_forward_expand(forward, result);
    CHECK(ggml_graph_compute_with_ctx(ctx, forward, 2) == GGML_STATUS_SUCCESS);
    // Independent scalar reference uses per-image geometry rather than Plan indices.
    int src_offset = 0, dst_offset = 0;
    for (int i = 0; i < text * channels; ++i) CHECK(static_cast<float*>(low->data)[i] == input[i]);
    for (const auto& grid : grids) {
        const int lw = (grid.width + 1) / 2;
        for (int y = 0; y < grid.height; y += 2) {
            for (int x0 = 0; x0 < grid.width; x0 += 2) {
                const int dst = dst_offset + y / 2 * lw + x0 / 2;
                for (int c = 0; c < channels; ++c) {
                    float sum = 0; int count = 0;
                    for (int yy = y; yy < std::min(y + 2, grid.height); ++yy)
                        for (int xx = x0; xx < std::min(x0 + 2, grid.width); ++xx) {
                            sum += input[(text + src_offset + yy * grid.width + xx) * channels + c]; ++count;
                        }
                    CHECK(static_cast<float*>(low->data)[(text + dst) * channels + c] == sum / count);
                }
                const int center = text + src_offset + std::min(y + 1, grid.height - 1) * grid.width + std::min(x0 + 1, grid.width - 1);
                for (int c = 0; c < 8; ++c)
                    CHECK(static_cast<float*>(graph.pe->data)[(text + dst) * 8 + c] == center * 8 + c);
            }
        }
        for (int y = 0; y < grid.height; ++y)
            for (int x0 = 0; x0 < grid.width; ++x0)
                for (int c = 0; c < channels; ++c) {
                    const int target = (text + src_offset + y * grid.width + x0) * channels + c;
                    const float delta = (text + dst_offset + y / 2 * lw + x0 / 2 + c + 1) * 0.25f;
                    CHECK(static_cast<float*>(expanded->data)[target] == input[target] + delta);
                }
        src_offset += grid.width * grid.height;
        dst_offset += lw * ((grid.height + 1) / 2);
    }
    for (int i = 0; i < text + images; ++i)
        for (int c = 0; c < channels; ++c)
            CHECK(static_cast<float*>(identity->data)[i * channels + c] == input[i * channels + c]);
    for (int i = 0; i < text; ++i)
        for (int c = 0; c < channels; ++c)
            CHECK(static_cast<float*>(expanded->data)[i * channels + c] == input[i * channels + c] + (i + c + 1) * 0.25f);
    if (argc > 1 && std::strcmp(argv[1], "--gpu") == 0) {
        // Copy the exact same graph to the selected GPU, including non-contiguous
        // views and multi-image gather indices; compare every computed F32 node.
        auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        CHECK(backend);
        auto buffer = ggml_backend_cpu_buffer_from_ptr(ggml_get_mem_buffer(ctx), ggml_get_mem_size(ctx));
        CHECK(buffer);
        for (auto tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor))
            tensor->buffer = buffer;
        auto copy = ggml_backend_graph_copy(backend, forward);
        CHECK(copy.graph);
        CHECK(ggml_backend_graph_compute(backend, copy.graph) == GGML_STATUS_SUCCESS);
        for (int i = 0; i < ggml_graph_n_nodes(forward); ++i) {
            auto expected = ggml_graph_node(forward, i);
            auto actual = ggml_graph_node(copy.graph, i);
            if (expected->type != GGML_TYPE_F32 || !ggml_is_contiguous(expected)) continue;
            std::vector<float> values(ggml_nelements(actual));
            ggml_backend_tensor_get(actual, values.data(), 0, ggml_nbytes(actual));
            CHECK(std::memcmp(values.data(), expected->data, ggml_nbytes(expected)) == 0);
        }
        std::printf("PASS: HS graph CPU/%s exact numerical parity\n", ggml_backend_name(backend));
        ggml_backend_graph_copy_free(copy);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    }
    ggml_free(ctx);
    std::puts("PASS: HS rectangular/multi-reference/odd-edge pooling, RoPE selection, residual preservation, step gating");
}
