#pragma once
#include "mlx/weights.hpp"
#include <vector>
namespace pictor::anima_mlx {
using namespace mlx_backend;
struct Tokens {
    std::vector<int> qwen, t5;
    std::vector<float> weights;
};
Tokens tokenize(const std::string &prompt);
class Model {
  public:
    Model(const std::filesystem::path &directory, bool int4, bool bf16_compute);
    array text(const std::vector<int> &ids) const;
    array adapt(const array &text, const std::vector<int> &ids, const std::vector<float> &weights) const;
    array predict(const array &latent, float sigma, const array &context) const;
    array decode(const array &latent) const;
    array decode_tiled(const array &latent) const;
    array context(const std::string &prompt) const;

  private:
    Weights te_, ad_, dit_, vae_;
    bool bf16_;
};
} // namespace pictor::anima_mlx
