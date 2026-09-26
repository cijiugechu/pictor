// Single-frame Wan decoder, ported from xocialize/anima-mlx (MIT).
#include "model.hpp"
#include <cmath>
namespace pictor::anima_mlx {
array Model::decode(const array &latent) const {
    const auto &w = vae_;
    const float means[] = {-0.7571, -0.7089, -0.9113, 0.1075,  -0.1745, 0.9653,  -0.1517, 1.5508,
                           0.4134,  -0.0715, 0.5517,  -0.3632, -0.1922, -0.9497, 0.2503,  -0.2921};
    const float stds[] = {2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                          3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916};
    auto x = mx::transpose(mx::astype(latent, mx::float32) * array(stds, {1, 16, 1, 1}) + array(means, {1, 16, 1, 1}),
                           {0, 2, 3, 1});
    if (bf16_)
        x = mx::astype(x, mx::bfloat16);
    auto conv = [&](const std::string &n, const array &a) {
        auto weight = w.at(n + ".weight");
        // For T=1 causal conv with zero left history, only the final temporal slice contributes.
        if (weight.ndim() == 5)
            weight = mx::squeeze(part(weight, 1, weight.shape(1) - 1, weight.shape(1)), 1);
        auto y = mx::conv2d(a, weight, {1, 1}, {weight.shape(1) / 2, weight.shape(2) / 2});
        return y + w.at(n + ".bias");
    };
    auto norm = [&](const std::string &n, const array &a) {
        auto l2 = mx::sqrt(mx::sum(a * a, -1, true));
        return a / mx::maximum(l2, array(1e-12f, l2.dtype())) * array(std::sqrt(float(a.shape(-1))), a.dtype()) *
               w.at(n + ".gamma");
    };
    auto res = [&](const std::string &n, const array &a) {
        auto skip = w.count(n + ".conv_shortcut.conv.weight") ? conv(n + ".conv_shortcut.conv", a) : a;
        auto y = conv(n + ".conv1.conv", silu(norm(n + ".norm1", a)));
        return conv(n + ".conv2.conv", silu(norm(n + ".norm2", y))) + skip;
    };
    x = conv("post_quant_conv.conv", x);
    x = conv("decoder.conv_in.conv", x);
    x = res("decoder.mid_block.resnets.0", x);
    {
        std::string n = "decoder.mid_block.attentions.0";
        int h = x.shape(1), wi = x.shape(2), c = x.shape(3);
        auto qkv = mx::reshape(conv(n + ".to_qkv", norm(n + ".norm", x)), {1, h * wi, 3, c});
        auto q = mx::squeeze(part(qkv, 2, 0, 1), 2), k = mx::squeeze(part(qkv, 2, 1, 2), 2),
             v = mx::squeeze(part(qkv, 2, 2, 3), 2);
        auto scores =
            mx::softmax(mx::matmul(q, mx::transpose(k, {0, 2, 1})) * array(1 / std::sqrt(float(c)), q.dtype()), -1);
        x = x + conv(n + ".proj", mx::reshape(mx::matmul(scores, v), {1, h, wi, c}));
    }
    x = res("decoder.mid_block.resnets.1", x);
    mx::eval(x);
    for (int i = 0; i < 4; ++i) {
        auto b = "decoder.up_blocks." + std::to_string(i);
        for (int j = 0; j < 3; ++j) {
            x = res(b + ".resnets." + std::to_string(j), x);
            mx::eval(x);
        }
        if (i < 3)
            x = conv(b + ".upsamplers.0.resample.0", mx::repeat(mx::repeat(x, 2, 1), 2, 2));
        mx::eval(x);
    }
    x = conv("decoder.conv_out.conv", silu(norm("decoder.norm_out", x)));
    return mx::clip((mx::astype(x, mx::float32) + 1.0f) * 0.5f, array(0.0f), array(1.0f));
}
array Model::decode_tiled(const array &latent) const {
    const int h = latent.shape(2), w = latent.shape(3);
    if (h <= 32 && w <= 32)
        return decode(latent);
    // Explicit approximate decode: spatial attention and convolution context differ per tile.
    // Blend overlapping 32x32 latent tiles (256px), never enable implicitly.
    const int oh = h * 8, ow = w * 8, tile = 32, stride = 24;
    auto sum = mx::zeros({1, oh, ow, 3}, mx::float32), divisor = mx::zeros({1, oh, ow, 1}, mx::float32);
    for (int y = 0; y < h; y += stride)
        for (int z = 0; z < w; z += stride) {
            const int th = std::min(tile, h - y), tw = std::min(tile, w - z);
            auto decoded = decode(part(part(latent, 2, y, y + th), 3, z, z + tw));
            std::vector<float> blend(th * 8 * tw * 8);
            for (int iy = 0; iy < th * 8; ++iy)
                for (int ix = 0; ix < tw * 8; ++ix) {
                    float a = 1, b = 1;
                    if (y > 0)
                        a = std::min(a, float(iy + 1) / 64);
                    if (y + th < h)
                        a = std::min(a, float(th * 8 - iy) / 64);
                    if (z > 0)
                        b = std::min(b, float(ix + 1) / 64);
                    if (z + tw < w)
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
    return sum / divisor;
}
} // namespace pictor::anima_mlx
