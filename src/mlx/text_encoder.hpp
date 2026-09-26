#pragma once
#include "tokenizers/qwen2_tokenizer.h"
#include "weights.hpp"
namespace pictor::mlx_backend {
// Preserve Qwen vocabulary while using the checkpoint's Unicode pre-tokenizer.
class QwenTokenizer : public Qwen2Tokenizer {
    std::string normalize(const std::string &text) const override;
    std::vector<std::string> token_split(const std::string &text) const override;
};
class TextEncoder {
  public:
    explicit TextEncoder(const std::filesystem::path &path);
    array encode(const std::string &prompt);
    std::vector<int> tokens(const std::string &prompt);

  private:
    Weights weights_;
    QwenTokenizer tokenizer_;
};
} // namespace pictor::mlx_backend
