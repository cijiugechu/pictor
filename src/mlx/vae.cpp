// Flux2 VAE equations follow MFLUX 0.20.0 (MIT license).
#include "vae.hpp"
#include <algorithm>
#include <stdexcept>

namespace pictor::mlx_backend {
namespace {
void replace(std::string &s, const std::string &from, const std::string &to) {
    const auto p = s.find(from);
    if (p != std::string::npos)
        s.replace(p, from.size(), to);
}
Weights convert_original(Weights weights) {
    Weights out;
    for (auto &[key, value] : weights) {
        if (key == "bn.num_batches_tracked")
            continue;
        auto name = key;
        replace(name, "encoder.quant_conv", "quant_conv");
        replace(name, "decoder.post_quant_conv", "post_quant_conv");
        replace(name, ".norm_out.", ".conv_norm_out.");
        replace(name, ".nin_shortcut.", ".conv_shortcut.");
        replace(name, ".mid.block_1.", ".mid_block.resnets.0.");
        replace(name, ".mid.block_2.", ".mid_block.resnets.1.");
        const bool attn = name.find(".mid.attn_1.") != std::string::npos;
        if (attn) {
            replace(name, ".mid.attn_1.", ".mid_block.attentions.0.");
            replace(name, ".norm.", ".group_norm.");
            for (const auto &item : {std::pair{"q", "to_q"}, {"k", "to_k"}, {"v", "to_v"}, {"proj_out", "to_out"}})
                replace(name, "." + std::string(item.first) + ".", "." + std::string(item.second) + ".");
        }
        for (int i = 0; i < 4; ++i) {
            replace(name, "encoder.down." + std::to_string(i) + ".block.",
                    "encoder.down_blocks." + std::to_string(i) + ".resnets.");
            replace(name, "encoder.down." + std::to_string(i) + ".downsample.",
                    "encoder.down_blocks." + std::to_string(i) + ".downsamplers.0.");
            replace(name, "decoder.up." + std::to_string(i) + ".block.",
                    "decoder.up_blocks." + std::to_string(3 - i) + ".resnets.");
            replace(name, "decoder.up." + std::to_string(i) + ".upsample.",
                    "decoder.up_blocks." + std::to_string(3 - i) + ".upsamplers.0.");
        }
        if (value.ndim() == 4)
            value = attn ? mx::reshape(value, {value.shape(0), value.shape(1)}) : mx::transpose(value, {0, 2, 3, 1});
        out.emplace(std::move(name), std::move(value));
    }
    return out;
}
} // namespace
Vae::Vae(const std::filesystem::path &path) : weights_(load_weights(path)) {
    if (weights_.count("decoder.mid.block_1.norm1.weight"))
        weights_ = convert_original(std::move(weights_));
    if (weights_.at("bn.running_mean").size() != 128 || weights_.at("decoder.conv_in.weight").shape(3) != 32)
        throw std::invalid_argument("MLX requires a complete Flux2 VAE (original or Small Decoder)");
}
array Vae::conv(const std::string &n, const array &x, int stride, int padding) const {
    const auto &weight = weights_.at(n + ".weight");
    if (padding < 0)
        padding = weight.shape(1) / 2;
    return mx::conv2d(x, weight, {stride, stride}, {padding, padding}) + weights_.at(n + ".bias");
}
array Vae::norm(const std::string &n, const array &x) const {
    const int c = x.shape(3), hw = x.shape(1) * x.shape(2);
    auto y = mx::reshape(mx::astype(x, mx::float32), {1, hw, 32, c / 32});
    y = mx::reshape(mx::transpose(y, {0, 2, 1, 3}), {1, 32, hw * c / 32});
    y = layer_norm(y);
    y = mx::reshape(mx::transpose(mx::reshape(y, {1, 32, hw, c / 32}), {0, 2, 1, 3}), x.shape());
    return mx::astype(weights_.at(n + ".weight") * y + weights_.at(n + ".bias"), mx::bfloat16);
}
array Vae::resnet(const std::string &n, const array &x) const {
    auto y = conv(n + ".conv1", silu(norm(n + ".norm1", x)));
    y = conv(n + ".conv2", silu(norm(n + ".norm2", y)));
    return y + (weights_.count(n + ".conv_shortcut.weight") ? conv(n + ".conv_shortcut", x) : x);
}
array Vae::mid(const std::string &n, array x) const {
    x = resnet(n + ".resnets.0", x);
    const std::string a = n + ".attentions.0";
    const auto y = norm(a + ".group_norm", x);
    const auto qkv = [&](const std::string &suffix) {
        return mx::reshape(linear(weights_, a + suffix, y), {1, 1, x.shape(1) * x.shape(2), x.shape(3)});
    };
    const auto out = mx::reshape(attention(qkv(".to_q"), qkv(".to_k"), qkv(".to_v")), x.shape());
    x = x + linear(weights_, a + ".to_out", out);
    return resnet(n + ".resnets.1", x);
}
array Vae::encode(const array &pixels) {
    auto x = conv("encoder.conv_in", mx::astype(pixels, mx::bfloat16));
    for (int i = 0; i < 4; ++i) {
        const auto b = "encoder.down_blocks." + std::to_string(i);
        for (int j = 0; j < 2; ++j)
            x = resnet(b + ".resnets." + std::to_string(j), x);
        if (i < 3) {
            x = mx::pad(x, {{0, 0}, {0, 1}, {0, 1}, {0, 0}});
            x = conv(b + ".downsamplers.0.conv", x, 2, 0);
        }
    }
    x = mid("encoder.mid_block", x);
    x = conv("encoder.conv_out", silu(norm("encoder.conv_norm_out", x)));
    x = part(conv("quant_conv", x), 3, 0, 32);
    const int h = x.shape(1), w = x.shape(2);
    x = mx::transpose(x, {0, 3, 1, 2});
    x = mx::reshape(x, {1, 32, h / 2, 2, w / 2, 2});
    x = mx::reshape(mx::transpose(x, {0, 1, 3, 5, 2, 4}), {1, 128, h / 2, w / 2});
    const auto mean = mx::reshape(weights_.at("bn.running_mean"), {1, 128, 1, 1});
    const auto stddev =
        mx::sqrt(mx::reshape(weights_.at("bn.running_var"), {1, 128, 1, 1}) + array(1e-4f, mx::bfloat16));
    return (x - mean) / stddev;
}
array Vae::decode_latent(array x) const {
    x = conv("post_quant_conv", x);
    x = mid("decoder.mid_block", conv("decoder.conv_in", x));
    for (int i = 0; i < 4; ++i) {
        const auto b = "decoder.up_blocks." + std::to_string(i);
        for (int j = 0; j < 3; ++j)
            x = resnet(b + ".resnets." + std::to_string(j), x);
        if (i < 3)
            x = conv(b + ".upsamplers.0.conv", mx::repeat(mx::repeat(x, 2, 1), 2, 2));
    }
    return conv("decoder.conv_out", silu(norm("decoder.conv_norm_out", x)));
}
array Vae::decode(const array &packed, bool tiled) {
    const int h = packed.shape(2), w = packed.shape(3);
    const auto mean = mx::reshape(weights_.at("bn.running_mean"), {1, 128, 1, 1});
    const auto stddev =
        mx::sqrt(mx::reshape(weights_.at("bn.running_var"), {1, 128, 1, 1}) + array(1e-4f, mx::bfloat16));
    auto x = mx::reshape(packed * stddev + mean, {1, 32, 2, 2, h, w});
    x = mx::reshape(mx::transpose(x, {0, 1, 4, 2, 5, 3}), {1, 32, 2 * h, 2 * w});
    x = mx::transpose(x, {0, 2, 3, 1});
    if (!tiled || (2 * h <= 32 && 2 * w <= 32))
        return mx::transpose(decode_latent(x), {0, 3, 1, 2});
    // Explicit tiling is approximate because each tile has its own GroupNorm.
    // Blend overlapping 32x32 latent tiles (256px); never choose it implicitly.
    const int oh = h * 16, ow = w * 16, tile = 32, stride = 24;
    auto sum = mx::zeros({1, oh, ow, 3}, mx::float32), divisor = mx::zeros({1, oh, ow, 1}, mx::float32);
    for (int y = 0; y < 2 * h; y += stride)
        for (int z = 0; z < 2 * w; z += stride) {
            const int th = std::min(tile, 2 * h - y), tw = std::min(tile, 2 * w - z);
            auto decoded = mx::astype(decode_latent(part(part(x, 1, y, y + th), 2, z, z + tw)), mx::float32);
            std::vector<float> blend(th * 8 * tw * 8);
            for (int iy = 0; iy < th * 8; ++iy)
                for (int ix = 0; ix < tw * 8; ++ix) {
                    float a = 1, b = 1;
                    if (y > 0)
                        a = std::min(a, float(iy + 1) / 64);
                    if (y + th < 2 * h)
                        a = std::min(a, float(th * 8 - iy) / 64);
                    if (z > 0)
                        b = std::min(b, float(ix + 1) / 64);
                    if (z + tw < 2 * w)
                        b = std::min(b, float(tw * 8 - ix) / 64);
                    blend[iy * tw * 8 + ix] = a * b;
                }
            const array weight(blend.data(), {1, th * 8, tw * 8, 1});
            const std::vector<std::pair<int, int>> pads{
                {0, 0}, {y * 8, oh - (y + th) * 8}, {z * 8, ow - (z + tw) * 8}, {0, 0}};
            sum = sum + mx::pad(decoded * weight, pads);
            divisor = divisor + mx::pad(weight, pads);
            mx::eval({sum, divisor});
        }
    return mx::transpose(sum / divisor, {0, 3, 1, 2});
}
} // namespace pictor::mlx_backend
