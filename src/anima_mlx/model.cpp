// Native port of xocialize/anima-mlx, pinned in benchmarks/anima-mlx-model.json.
// Copyright (c) 2026 xocialize. MIT: licenses/anima-mlx.txt.
#include "model.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <spdlog/fmt/fmt.h>
namespace pictor::anima_mlx {
static Weights load(const std::filesystem::path &p) {
    auto w = mx::load_safetensors(p.string()).first;
    std::vector<array> values;
    for (const auto &[k, v] : w)
        values.push_back(v);
    mx::eval(values);
    return w;
}
Model::Model(const std::filesystem::path &p, bool int4, bool bf16)
    : te_(load(p / "text_encoder-bf16.safetensors")), ad_(load(p / "llm_adapter-bf16.safetensors")),
      dit_(load(p / (int4 ? "transformer-int4.safetensors" : "transformer-bf16.safetensors"))),
      vae_(load(p / "vae-bf16.safetensors")), bf16_(bf16) {
    if (te_.at("embed_tokens.weight").shape() != mx::Shape{151936, 1024} ||
        dit_.at("patch_embed.proj.weight").shape() != mx::Shape{2048, 68})
        throw std::invalid_argument("unsupported Anima architecture");
    if (!dit_.count("transformer_blocks.27.ff.net.2.weight") || dit_.count("transformer_blocks.28.ff.net.2.weight"))
        throw std::invalid_argument("expected the validated 28-layer Anima model");
}
static array norm(const Weights &w, const std::string &n, const array &x) {
    return mx::fast::rms_norm(x, w.at(fmt::format("{}.weight", n)), 1e-6f);
}
static std::pair<array, array> rope(int seq, int dim, float theta, mx::Dtype dtype) {
    auto inv = 1.0f / mx::power(array(theta), mx::arange(0, dim, 2, mx::float32) / float(dim));
    auto f = mx::reshape(mx::arange(seq, mx::float32), {seq, 1}) * mx::reshape(inv, {1, dim / 2});
    auto e = mx::reshape(mx::concatenate({f, f}, -1), {1, 1, seq, dim});
    return {mx::astype(mx::cos(e), dtype), mx::astype(mx::sin(e), dtype)};
}
static array rotate(const array &x, const std::pair<array, array> &r) {
    int d = x.shape(-1) / 2;
    return x * r.first + mx::concatenate({-part(x, -1, d, d * 2), part(x, -1, 0, d)}, -1) * r.second;
}
static array attend(const array &q, const array &k, const array &v, bool causal = false) {
    // The reference FP32 RoPE promotes q/k; MLX promotes values to the same dtype.
    auto dtype = mx::promote_types(q.dtype(), v.dtype());
    auto y = mx::fast::scaled_dot_product_attention(mx::astype(q, dtype), mx::astype(k, dtype), mx::astype(v, dtype),
                                                    1.0f / std::sqrt(float(q.shape(-1))), causal ? "causal" : "");
    return mx::reshape(mx::transpose(y, {0, 2, 1, 3}), {q.shape(0), q.shape(2), q.shape(1) * q.shape(3)});
}
static array gelu(const array &x) {
    static auto fn = mx::compile([](const std::vector<array> &a) {
        const auto &x = a[0];
        return std::vector<array>{x * (array(1, x.dtype()) + mx::erf(x / array(std::sqrt(2.0f), x.dtype()))) *
                                  array(0.5f, x.dtype())};
    });
    return fn({x})[0];
}
array Model::text(const std::vector<int> &ids) const {
    const auto &w = te_;
    auto x = mx::reshape(mx::take(w.at("embed_tokens.weight"), array(ids.data(), {int(ids.size())}, mx::int32), 0),
                         {1, int(ids.size()), 1024});
    auto r = rope(int(ids.size()), 128, 1000000.0f, bf16_ ? x.dtype() : mx::float32);
    for (int i = 0; i < 28; ++i) {
        auto b = fmt::format("layers.{}", i), a = fmt::format("{}.self_attn", b);
        auto n = norm(w, fmt::format("{}.input_layernorm", b), x);
        auto q = rotate(norm(w, fmt::format("{}.q_norm", a), heads(linear(w, fmt::format("{}.q_proj", a), n), 16)), r);
        auto k = rotate(norm(w, fmt::format("{}.k_norm", a), heads(linear(w, fmt::format("{}.k_proj", a), n), 8)), r);
        auto v = heads(linear(w, fmt::format("{}.v_proj", a), n), 8);
        x = x + linear(w, fmt::format("{}.o_proj", a), attend(q, k, v, true));
        n = norm(w, fmt::format("{}.post_attention_layernorm", b), x);
        x = x + linear(w, fmt::format("{}.mlp.down_proj", b),
                       silu(linear(w, fmt::format("{}.mlp.gate_proj", b), n)) * linear(w, fmt::format("{}.mlp.up_proj", b), n));
    }
    return norm(w, "norm", x);
}
array Model::adapt(const array &context, const std::vector<int> &ids, const std::vector<float> &weights) const {
    const auto &w = ad_;
    auto x = mx::astype(mx::reshape(mx::take(w.at("embed.weight"), array(ids.data(), {int(ids.size())}, mx::int32), 0),
                                    {1, int(ids.size()), 1024}),
                        context.dtype());
    auto rq = rope(x.shape(1), 64, 10000, bf16_ ? x.dtype() : mx::float32);
    auto rk = rope(context.shape(1), 64, 10000, bf16_ ? x.dtype() : mx::float32);
    auto att = [&](const std::string &a, const array &n, const array &ctx, const std::pair<array, array> &kr) {
        auto q = rotate(norm(w, fmt::format("{}.q_norm", a), heads(linear(w, fmt::format("{}.q_proj", a), n), 16)), rq);
        auto k = rotate(norm(w, fmt::format("{}.k_norm", a), heads(linear(w, fmt::format("{}.k_proj", a), ctx), 16)), kr);
        auto v = heads(linear(w, fmt::format("{}.v_proj", a), ctx), 16);
        return linear(w, fmt::format("{}.o_proj", a), attend(q, k, v));
    };
    for (int i = 0; i < 6; ++i) {
        auto b = fmt::format("blocks.{}", i);
        auto n = norm(w, fmt::format("{}.norm_self_attn", b), x);
        x = x + att(fmt::format("{}.self_attn", b), n, n, rq);
        x = x + att(fmt::format("{}.cross_attn", b), norm(w, fmt::format("{}.norm_cross_attn", b), x), context, rk);
        x = x + linear(w, fmt::format("{}.mlp.2", b), gelu(linear(w, fmt::format("{}.mlp.0", b), norm(w, fmt::format("{}.norm_mlp", b), x))));
    }
    x = norm(w, "norm", linear(w, "out_proj", x));
    if (!weights.empty())
        x = x * mx::astype(array(weights.data(), {1, int(weights.size()), 1}), x.dtype());
    if (x.shape(1) < 512)
        x = mx::pad(x, std::vector<std::pair<int, int>>{{0, 0}, {0, 512 - x.shape(1)}, {0, 0}});
    return part(x, 1, 0, 512);
}
array Model::context(const std::string &prompt) const {
    auto t = tokenize(prompt);
    return adapt(text(t.qwen), t.t5, t.weights);
}
static std::pair<array, array> image_rope(int h, int w) {
    std::vector<float> values(h * w * 128);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int start = 0;
            for (int a = 0; a < 3; ++a) {
                int dim = a == 0 ? 44 : 42;
                float pos = a == 0 ? 0 : float(a == 1 ? y : x);
                float theta = 10000 * std::pow(a == 0 ? 1.0f : 4.0f, float(dim) / float(dim - 2));
                for (int j = 0; j < dim / 2; ++j) {
                    auto v = pos / std::pow(theta, float(j * 2) / dim);
                    values[(y * w + x) * 128 + start + j] = v;
                    values[(y * w + x) * 128 + 64 + start + j] = v;
                }
                start += dim / 2;
            }
        }
    auto f = array(values.data(), {1, 1, h * w, 128});
    return {mx::cos(f), mx::sin(f)};
}
array Model::predict(const array &latent, float sigma, const array &context) const {
    const auto &w = dit_;
    int h = latent.shape(2), wi = latent.shape(3);
    auto input = bf16_ ? mx::astype(latent, mx::bfloat16) : latent;
    auto x = mx::concatenate({input, mx::zeros({1, 1, h, wi}, input.dtype())}, 1);
    x = mx::reshape(mx::transpose(mx::reshape(x, {1, 17, h / 2, 2, wi / 2, 2}), {0, 2, 4, 1, 3, 5}),
                    {1, h / 2 * wi / 2, 68});
    x = linear(w, "patch_embed.proj", x);
    auto freq = mx::exp(mx::arange(1024, mx::float32) * (-std::log(10000.0f) / 1024));
    auto proj = mx::reshape(mx::concatenate({mx::cos(freq * sigma), mx::sin(freq * sigma)}, 0), {1, 2048});
    if (bf16_)
        proj = mx::astype(proj, mx::bfloat16);
    auto temb = linear(w, "time_embed.t_embedder.linear_2", silu(linear(w, "time_embed.t_embedder.linear_1", proj)));
    auto embedded = silu(norm(w, "time_embed.norm", proj));
    auto r = image_rope(h / 2, wi / 2);
    auto ctx = bf16_ ? mx::astype(context, mx::bfloat16) : context;
    auto mod = [&](const std::string &b, const array &a, int chunks) {
        auto e = linear(w, fmt::format("{}.linear_2", b), linear(w, fmt::format("{}.linear_1", b), embedded)) + part(temb, -1, 0, 2048 * chunks);
        auto parts = mx::split(mx::expand_dims(e, 1), chunks, -1);
        // Preserve reference's explicit mean/variance computation.
        auto mean = mx::mean(a, -1, true);
        auto var = mx::mean(mx::square(a - mean), -1, true);
        auto n = (a - mean) * mx::rsqrt(var + array(1e-6f, var.dtype()));
        parts[0] = n * (array(1, parts[1].dtype()) + parts[1]) + parts[0];
        return parts;
    };
    auto att = [&](const std::string &a, const array &n, const array &kv, bool rotated) {
        auto q = norm(w, fmt::format("{}.norm_q", a), heads(linear(w, fmt::format("{}.to_q", a), n), 16));
        auto k = norm(w, fmt::format("{}.norm_k", a), heads(linear(w, fmt::format("{}.to_k", a), kv), 16));
        auto v = heads(linear(w, fmt::format("{}.to_v", a), kv), 16);
        if (rotated) {
            q = mx::astype(rotate(mx::astype(q, mx::float32), r), v.dtype());
            k = mx::astype(rotate(mx::astype(k, mx::float32), r), v.dtype());
        }
        return linear(w, fmt::format("{}.to_out.0", a), attend(q, k, v));
    };
    for (int i = 0; i < 28; ++i) {
        auto b = fmt::format("transformer_blocks.{}", i);
        auto m = mod(fmt::format("{}.norm1", b), x, 3);
        x = x + m[2] * att(fmt::format("{}.attn1", b), m[0], m[0], true);
        m = mod(fmt::format("{}.norm2", b), x, 3);
        x = x + m[2] * att(fmt::format("{}.attn2", b), m[0], ctx, false);
        m = mod(fmt::format("{}.norm3", b), x, 3);
        x = x + m[2] * linear(w, fmt::format("{}.ff.net.2", b), gelu(linear(w, fmt::format("{}.ff.net.0.proj", b), m[0])));
    }
    x = linear(w, "proj_out", mod("norm_out", x, 2)[0]);
    return mx::reshape(mx::transpose(mx::reshape(x, {1, h / 2, wi / 2, 2, 2, 16}), {0, 5, 1, 3, 2, 4}), {1, 16, h, wi});
}
} // namespace pictor::anima_mlx
