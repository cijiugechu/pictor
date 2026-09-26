#include "model.hpp"
#include "tokenizers/qwen2_tokenizer.h"
#include "tokenizers/t5_unigram_tokenizer.h"
#include "util.h"
#include <stdexcept>
namespace pictor::anima_mlx {
Tokens tokenize(const std::string &prompt) {
    // Upstream tokenizers can silently drop malformed bytes. Reject them at the
    // public boundary without changing normalization or valid token sequences.
    try {
        const auto unicode = utf8_to_utf32(prompt);
        if (utf32_to_utf8(unicode) != prompt)
            throw std::invalid_argument("prompt must be valid UTF-8");
        for (const auto codepoint : unicode)
            if (codepoint >= 0xd800 && codepoint <= 0xdfff)
                throw std::invalid_argument("prompt must be valid UTF-8");
    } catch (const std::range_error &) {
        throw std::invalid_argument("prompt must be valid UTF-8");
    }
    Qwen2Tokenizer q;
    T5UniGramTokenizer t;
    Tokens out;
    for (const auto &[text, weight] : parse_prompt_attention(prompt)) {
        auto ids = q.tokenize(text, nullptr);
        out.qwen.insert(out.qwen.end(), ids.begin(), ids.end());
        ids = t.tokenize(text, nullptr, true);
        out.t5.insert(out.t5.end(), ids.begin(), ids.end());
        out.weights.insert(out.weights.end(), ids.size(), weight);
    }
    if (out.qwen.empty())
        out.qwen.push_back(151643);
    if (out.t5.empty()) {
        out.t5.push_back(1);
        out.weights.push_back(1);
    }
    return out;
}
} // namespace pictor::anima_mlx
