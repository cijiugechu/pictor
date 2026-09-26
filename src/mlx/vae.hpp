#pragma once
#include "weights.hpp"
namespace pictor::mlx_backend {
class Vae {
  public:
    explicit Vae(const std::filesystem::path &path);
    array encode(const array &pixels);             // NHWC [-1,1] -> normalized packed NCHW
    array decode(const array &packed, bool tiled); // packed NCHW -> RGB NCHW [-1,1]
  private:
    Weights weights_;
    array conv(const std::string &name, const array &x, int stride = 1, int padding = -1) const;
    array norm(const std::string &name, const array &x) const;
    array resnet(const std::string &name, const array &x) const;
    array mid(const std::string &name, array x) const;
    array decode_latent(array x) const;
};
} // namespace pictor::mlx_backend
