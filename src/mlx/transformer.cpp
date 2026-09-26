// Klein 4B equations follow MFLUX 0.20.0 (MIT license); see third-party notices.
#include "transformer.hpp"
#include <cmath>
#include <stdexcept>

namespace pictor::mlx_backend {
namespace {
std::pair<array, array> rotary(const array &ids) {
    std::vector<array> cs, sn;
    const auto omega = 1.0f / mx::power(array(2000.0f), mx::arange(0, 32, 2, mx::float32) / 32.0f);
    for (int axis = 0; axis < 4; ++axis) {
        const auto p = mx::astype(part(ids, 1, axis, axis + 1), mx::float32);
        cs.push_back(mx::cos(p * omega));
        sn.push_back(mx::sin(p * omega));
    }
    return {mx::concatenate(cs, -1), mx::concatenate(sn, -1)};
}
array rotate(const array &x, const std::pair<array, array> &pos) {
    const auto pairs = mx::reshape(mx::astype(x, mx::float32), {1, x.shape(1), x.shape(2), 64, 2});
    const auto real = mx::squeeze(part(pairs, 4, 0, 1), 4), imag = mx::squeeze(part(pairs, 4, 1, 2), 4);
    const auto c = mx::reshape(pos.first, {1, 1, x.shape(2), 64});
    const auto s = mx::reshape(pos.second, {1, 1, x.shape(2), 64});
    return mx::astype(mx::reshape(mx::stack({real * c + (-imag) * s, imag * c + real * s}, -1), x.shape()), x.dtype());
}
array swiglu(const array &x) {
    const auto p = mx::split(x, 2, -1);
    return silu(p[0]) * p[1];
}
array feedforward(const Weights &w, const std::string &n, const array &x) {
    return linear(w, n + ".linear_out", swiglu(linear(w, n + ".linear_in", x)));
}
std::vector<array> modulation(const Weights &w, const std::string &n, const array &t, int count) {
    return mx::split(mx::expand_dims(linear(w, n + ".linear", silu(t)), 1), count, -1);
}
array modulate(const array &x, const std::vector<array> &m, int offset) {
    return (array(1.0f, m[offset + 1].dtype()) + m[offset + 1]) * layer_norm(x) + m[offset];
}
} // namespace
Transformer::Transformer(const std::filesystem::path &path) : weights_(load_weights(path)) {
    if (weights_.at("x_embedder.weight").shape(0) != 3072 ||
        !weights_.count("single_transformer_blocks.19.attn.to_out.weight"))
        throw std::invalid_argument("MLX requires Klein 4B affine 4-bit/group-64 weights");
    compiled_ = mx::compile(
        [this](const std::vector<array> &a) { return std::vector<array>{forward(a[0], a[1], a[2], a[3])}; });
}
array Transformer::predict(const array &image, const array &text, const array &t, const array &ids) {
    return compiled_({image, text, t, ids})[0];
}
array Transformer::forward(const array &image, const array &text, const array &timestep, const array &image_ids) {
    const auto &w = weights_;
    auto t = mx::astype(mx::reshape(timestep, {1}), mx::bfloat16);
    t = t * mx::astype(mx::where(mx::max(t) <= 1.0f, array(1000.0f), array(1.0f)), mx::bfloat16);
    const auto freqs = mx::exp(-std::log(10000.0f) * mx::arange(128, mx::float32) / 128.0f);
    const auto args = mx::astype(mx::reshape(t, {1, 1}), mx::float32) * freqs;
    auto temb =
        linear(w, "time_guidance_embed.linear_2",
               silu(linear(w, "time_guidance_embed.linear_1", mx::concatenate({mx::cos(args), mx::sin(args)}, -1))));
    temb = mx::astype(temb, mx::bfloat16);
    auto x = linear(w, "x_embedder", image), txt = linear(w, "context_embedder", text);
    const int nt = txt.shape(1);
    std::vector<int> text_coords(nt * 4);
    for (int i = 0; i < nt; ++i)
        text_coords[i * 4 + 3] = i;
    const auto pos = rotary(mx::concatenate({array(text_coords.data(), {nt, 4}, mx::int32), image_ids}, 0));
    const auto im = modulation(w, "double_stream_modulation_img", temb, 6);
    const auto tm = modulation(w, "double_stream_modulation_txt", temb, 6);
    for (int i = 0; i < 5; ++i) {
        const std::string base = "transformer_blocks." + std::to_string(i), a = base + ".attn";
        const auto nx = modulate(x, im, 0), nc = modulate(txt, tm, 0);
        auto q = rms(w, a + ".norm_q", heads(linear(w, a + ".to_q", nx), 24), 1e-5f);
        auto k = rms(w, a + ".norm_k", heads(linear(w, a + ".to_k", nx), 24), 1e-5f);
        auto v = heads(linear(w, a + ".to_v", nx), 24);
        const auto tq = rms(w, a + ".norm_added_q", heads(linear(w, a + ".add_q_proj", nc), 24), 1e-5f);
        const auto tk = rms(w, a + ".norm_added_k", heads(linear(w, a + ".add_k_proj", nc), 24), 1e-5f);
        const auto tv = heads(linear(w, a + ".add_v_proj", nc), 24);
        q = rotate(mx::concatenate({tq, q}, 2), pos);
        k = rotate(mx::concatenate({tk, k}, 2), pos);
        const auto out = attention(q, k, mx::concatenate({tv, v}, 2));
        x = x + im[2] * linear(w, a + ".to_out", part(out, 1, nt, out.shape(1)));
        txt = txt + tm[2] * linear(w, a + ".to_add_out", part(out, 1, 0, nt));
        x = x + im[5] * feedforward(w, base + ".ff", modulate(x, im, 3));
        txt = txt + tm[5] * feedforward(w, base + ".ff_context", modulate(txt, tm, 3));
    }
    x = mx::concatenate({txt, x}, 1);
    const auto sm = modulation(w, "single_stream_modulation", temb, 3);
    for (int i = 0; i < 20; ++i) {
        const std::string a = "single_transformer_blocks." + std::to_string(i) + ".attn";
        const auto proj = linear(w, a + ".to_qkv_mlp_proj", modulate(x, sm, 0));
        auto q = rms(w, a + ".norm_q", heads(part(proj, -1, 0, 3072), 24), 1e-5f);
        auto k = rms(w, a + ".norm_k", heads(part(proj, -1, 3072, 6144), 24), 1e-5f);
        const auto v = heads(part(proj, -1, 6144, 9216), 24);
        const auto out = attention(rotate(q, pos), rotate(k, pos), v);
        const auto mlp = swiglu(part(proj, -1, 9216, proj.shape(-1)));
        x = x + sm[2] * linear(w, a + ".to_out", mx::concatenate({out, mlp}, -1));
    }
    x = part(x, 1, nt, x.shape(1));
    const auto mods = mx::split(linear(w, "norm_out.linear", silu(temb)), 2, -1);
    x = layer_norm(x) * mx::expand_dims(array(1.0f, mods[0].dtype()) + mods[0], 1) + mx::expand_dims(mods[1], 1);
    return linear(w, "proj_out", x);
}
} // namespace pictor::mlx_backend
