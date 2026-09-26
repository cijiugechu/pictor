#pragma once
#include "weights.hpp"
namespace pictor::mlx_backend {
class Transformer {
  public:
    explicit Transformer(const std::filesystem::path &path);
    array predict(const array &image, const array &text, const array &timestep, const array &image_ids);

  private:
    Weights weights_;
    std::function<std::vector<array>(const std::vector<array> &)> compiled_;
    array forward(const array &image, const array &text, const array &timestep, const array &image_ids);
};
} // namespace pictor::mlx_backend
