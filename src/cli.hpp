#pragma once

#include "pictor/anima.hpp"
#include "pictor/flux_klein.hpp"
#include <string_view>

namespace pictor::cli {

enum class Model { anima, flux_klein };

struct Options {
    Model model = Model::anima;
    KleinBackend backend = KleinBackend::automatic;
    AnimaBackend anima_backend = AnimaBackend::automatic;
    std::filesystem::path text_encoder;
    std::filesystem::path vae;
    std::vector<std::filesystem::path> reference_images;
    bool auto_resize_reference = true;
    bool hidden_state_compression = false;
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
