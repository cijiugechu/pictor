#pragma once
#include <filesystem>
#include <mlx/mlx.h>
#include <string>
#include <unordered_map>

namespace pictor::mlx_backend {
namespace mx = mlx::core;
using mx::array;
using Weights = std::unordered_map<std::string, array>;
Weights load_weights(const std::filesystem::path &path);
array linear(const Weights &weights, const std::string &name, const array &x);
array silu(const array &x);
array part(const array &x, int axis, int begin, int end);
array layer_norm(const array &x);
array rms(const Weights &weights, const std::string &name, const array &x, float eps, bool qwen = false);
array heads(const array &x, int count);
array attention(const array &q, const array &k, const array &v, const std::optional<array> &mask = {});
array grid_ids(int height, int width, int index);
std::vector<float> discrete_sigmas(int width, int height, int steps);
} // namespace pictor::mlx_backend
