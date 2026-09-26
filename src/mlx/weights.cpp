#include "weights.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <spdlog/fmt/fmt.h>

namespace pictor::mlx_backend {
Weights load_weights(const std::filesystem::path &path) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_directory(path)) {
        for (const auto &entry : std::filesystem::directory_iterator(path))
            if (entry.path().extension() == ".safetensors")
                files.push_back(entry.path());
    } else
        files.push_back(path);
    if (files.empty())
        throw std::invalid_argument(fmt::format("no safetensors in {}", path.string()));
    std::sort(files.begin(), files.end());
    Weights out;
    for (const auto &file : files) {
        auto values = mx::load_safetensors(file.string()).first;
        for (auto &[name, value] : values) {
            if (value.dtype() == mx::float16 || value.dtype() == mx::float32)
                value = mx::astype(value, mx::bfloat16);
            if (!out.emplace(name, std::move(value)).second)
                throw std::invalid_argument(fmt::format("duplicate tensor: {}", name));
        }
    }
    std::vector<array> values;
    for (const auto &[name, value] : out)
        values.push_back(value);
    mx::eval(values);
    return out;
}
array linear(const Weights &w, const std::string &n, const array &x) {
    if (!w.count(fmt::format("{}.scales", n)) && w.count(fmt::format("{}.bias", n)))
        return mx::addmm(w.at(fmt::format("{}.bias", n)), x, mx::transpose(w.at(fmt::format("{}.weight", n))));
    auto y = w.count(fmt::format("{}.scales", n))
                 ? mx::quantized_matmul(x, w.at(fmt::format("{}.weight", n)), w.at(fmt::format("{}.scales", n)), w.at(fmt::format("{}.biases", n)), true, 64, 4)
                 : mx::matmul(x, mx::transpose(w.at(fmt::format("{}.weight", n))));
    if (w.count(fmt::format("{}.bias", n)))
        y = y + w.at(fmt::format("{}.bias", n));
    return y;
}
array silu(const array &x) {
    // MFLUX's nn.silu is compiled: one fused rounding in BF16.
    static auto fn =
        mx::compile([](const std::vector<array> &a) { return std::vector<array>{a[0] * mx::sigmoid(a[0])}; });
    return fn({x})[0];
}
array part(const array &x, int axis, int begin, int end) {
    if (axis < 0)
        axis += x.ndim();
    mx::Shape starts(x.ndim(), 0), ends = x.shape();
    starts[axis] = begin;
    ends[axis] = end;
    return mx::slice(x, starts, ends);
}
array layer_norm(const array &x) {
    return mx::fast::layer_norm(x, {}, {}, 1e-6f);
}
array rms(const Weights &w, const std::string &n, const array &x, float eps, bool qwen) {
    const auto f = mx::astype(x, mx::float32);
    if (qwen) {
        const auto normed = f * mx::rsqrt(mx::mean(mx::square(f), -1, true) + eps);
        return mx::astype(mx::astype(w.at(fmt::format("{}.weight", n)), mx::float32) * normed, x.dtype());
    }
    return mx::astype(mx::fast::rms_norm(f, w.at(fmt::format("{}.weight", n)), eps), x.dtype());
}
array heads(const array &x, int count) {
    return mx::transpose(mx::reshape(x, {1, x.shape(1), count, x.shape(2) / count}), {0, 2, 1, 3});
}
array attention(const array &q, const array &k, const array &v, const std::optional<array> &mask) {
    auto y = mx::fast::scaled_dot_product_attention(q, k, v, 1.0f / std::sqrt(float(q.shape(-1))), "", mask);
    return mx::reshape(mx::transpose(y, {0, 2, 1, 3}), {1, q.shape(2), q.shape(1) * q.shape(3)});
}
array grid_ids(int h, int w, int t) {
    std::vector<int> ids(h * w * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int i = (y * w + x) * 4;
            ids[i] = t;
            ids[i + 1] = y;
            ids[i + 2] = x;
        }
    return array(ids.data(), {h * w, 4}, mx::int32);
}
std::vector<float> discrete_sigmas(int width, int height, int steps) {
    const int seq = width / 16 * (height / 16);
    const float m200 = 0.00016927f * seq + 0.45666666f;
    const float m10 = 8.73809524e-05f * seq + 1.89833333f;
    const float a = (m200 - m10) / 190.0f;
    const float mu = seq > 4300 ? m200 : a * steps + (m200 - 200.0f * a);
    std::vector<float> result;
    for (int i = 0; i < steps; ++i) {
        const float t = (999.0f - (steps == 1 ? 0.0f : 999.0f / (steps - 1) * i) + 1.0f) / 1000.0f;
        result.push_back(std::exp(mu) / (std::exp(mu) + (1.0f / t - 1.0f)));
    }
    result.push_back(0);
    return result;
}
} // namespace pictor::mlx_backend
