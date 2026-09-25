#pragma once

#include "pictor/anima.hpp"
#include <string_view>

namespace pictor::cli {

struct Options {
    SessionOptions session;
    GenerationRequest request;
    std::filesystem::path output = "output.png";
    int count = 1;
    bool overwrite = false;
    bool help = false;
    bool version = false;
};

[[nodiscard]] Status parse(const std::vector<std::string>& arguments, Options& output) noexcept;
[[nodiscard]] Status output_path(const Options& options, int index, std::filesystem::path& output) noexcept;
std::string json_string(std::string_view value);
const char* usage();

} // namespace pictor::cli
