// Qwen3 conditioning equations follow MFLUX 0.20.0 (MIT license).
#include "text_encoder.hpp"
#include <limits>
#include <stdexcept>

namespace pictor::mlx_backend {
TextEncoder::TextEncoder(const std::filesystem::path &path) : weights_(load_weights(path)) {
    if (weights_.at("embed_tokens.weight").shape() != mx::Shape{151936, 320})
        throw std::invalid_argument("MLX requires Qwen3 4B affine 4-bit/group-64 weights");
}
std::vector<int> TextEncoder::tokens(const std::string &prompt) {
    auto result =
        tokenizer_.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    result.resize(512, 151643);
    return result;
}
array TextEncoder::encode(const std::string &prompt) {
    const auto &w = weights_;
    auto ids =
        tokenizer_.encode("<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    const int length = std::min(int(ids.size()), 512);
    ids.resize(512, 151643);
    const array index(ids.data(), {512}, mx::int32);
    auto x = mx::reshape(mx::dequantize(mx::take(w.at("embed_tokens.weight"), index, 0),
                                        mx::take(w.at("embed_tokens.scales"), index, 0),
                                        mx::take(w.at("embed_tokens.biases"), index, 0), 64, 4),
                         {1, 512, 2560});
    std::vector<float> mask_data(512 * 512);
    for (int y = 0; y < 512; ++y)
        for (int k = 0; k < 512; ++k)
            if (k > y || k >= length)
                mask_data[y * 512 + k] = -std::numeric_limits<float>::infinity();
    const auto mask = mx::astype(array(mask_data.data(), {1, 1, 512, 512}), mx::bfloat16);
    const auto freq = mx::reshape(mx::arange(512, mx::float32), {1, 512, 1}) *
                      (1.0f / mx::power(array(1000000.0f), mx::arange(0, 128, 2, mx::float32) / 128.0f));
    const auto emb = mx::concatenate({freq, freq}, -1);
    const auto c = mx::expand_dims(mx::astype(mx::cos(emb), mx::bfloat16), 1);
    const auto s = mx::expand_dims(mx::astype(mx::sin(emb), mx::bfloat16), 1);
    const auto rotate = [&c, &s](const array &a) {
        return a * c + mx::concatenate({-part(a, -1, 64, 128), part(a, -1, 0, 64)}, -1) * s;
    };
    std::vector<array> states;
    // Only hidden layers 9/18/27 are consumed; later layers are unused by Klein.
    for (int i = 0; i < 27; ++i) {
        const std::string b = "layers." + std::to_string(i), a = b + ".self_attn";
        const auto n = rms(w, b + ".input_layernorm", x, 1e-6f, true);
        auto q = mx::reshape(linear(w, a + ".q_proj", n), {1, 512, 32, 128});
        auto k = mx::reshape(linear(w, a + ".k_proj", n), {1, 512, 8, 128});
        q = rotate(mx::transpose(rms(w, a + ".q_norm", q, 1e-6f, true), {0, 2, 1, 3}));
        k = rotate(mx::transpose(rms(w, a + ".k_norm", k, 1e-6f, true), {0, 2, 1, 3}));
        const auto v = heads(linear(w, a + ".v_proj", n), 8);
        const auto out = mx::astype(attention(mx::astype(q, mx::float32), mx::astype(mx::repeat(k, 4, 1), mx::float32),
                                              mx::astype(mx::repeat(v, 4, 1), mx::float32), mask),
                                    mx::bfloat16);
        x = x + linear(w, a + ".o_proj", out);
        const auto post = rms(w, b + ".post_attention_layernorm", x, 1e-6f, true);
        x = x + linear(w, b + ".mlp.down_proj",
                       silu(linear(w, b + ".mlp.gate_proj", post)) * linear(w, b + ".mlp.up_proj", post));
        if (i == 8 || i == 17 || i == 26)
            states.push_back(x);
    }
    return mx::concatenate(states, -1);
}
} // namespace pictor::mlx_backend
