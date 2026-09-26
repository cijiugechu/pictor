#include "text_encoder.hpp"
#import <Foundation/Foundation.h>
#include <stdexcept>

namespace pictor::mlx_backend {
std::string QwenTokenizer::normalize(const std::string &text) const {
    @autoreleasepool {
        NSString *value = [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
        if (!value)
            throw std::invalid_argument("prompt must be valid UTF-8");
        return [[value precomposedStringWithCanonicalMapping] UTF8String];
    }
}
std::vector<std::string> QwenTokenizer::token_split(const std::string &text) const {
    @autoreleasepool {
        NSString *value = [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
        if (!value)
            throw std::invalid_argument("prompt must be valid UTF-8");
        static NSRegularExpression *regex = [NSRegularExpression
            regularExpressionWithPattern:@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
                                         @"?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                                 options:0
                                   error:nil];
        std::vector<std::string> parts;
        for (NSTextCheckingResult *match in [regex matchesInString:value options:0 range:NSMakeRange(0, value.length)])
            parts.emplace_back([[value substringWithRange:match.range] UTF8String]);
        return parts;
    }
}
} // namespace pictor::mlx_backend
