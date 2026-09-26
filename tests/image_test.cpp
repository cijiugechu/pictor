#include "pictor/types.hpp"
#include "pictor/pictor.h"
#include "stb_image_write.h"
#include <cstdio>
#include <cstdlib>
#include <spdlog/fmt/fmt.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
struct TempDirectory {
    std::filesystem::path path;
    ~TempDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
int main() {
    TempDirectory temp;
    std::error_code ec;
    const auto root = std::filesystem::temp_directory_path(ec);
    CHECK(!ec);
    for (int attempt = 0; attempt < 10 && temp.path.empty(); ++attempt) {
        std::int64_t seed;
        CHECK(pictor::resolve_seed(-1, seed));
        auto path = root / fmt::format("pictor-images-{}", seed);
        if (std::filesystem::create_directory(path, ec)) temp.path = std::move(path);
        CHECK(!ec);
    }
    CHECK(!temp.path.empty());
    const auto png = (temp.path / "rgba.png").string();
    const auto jpg = (temp.path / "rgb.jpg").string();
    const auto bad = (temp.path / "truncated.png").string();
    std::vector<uint8_t> rgba(17 * 19 * 4);
    for (size_t i = 0; i < rgba.size(); i += 4) {
        rgba[i] = 100; rgba[i + 1] = 50; rgba[i + 2] = 20; rgba[i + 3] = 0;
    }
    CHECK(stbi_write_png(png.c_str(), 17, 19, 4, rgba.data(), 17 * 4));
    pictor::Image image;
    CHECK(pictor::read_image(png, image));
    CHECK(image.width == 17 && image.height == 19 && image.channels == 3 && image.pixels.size() == 17 * 19 * 3);
    CHECK(image.pixels[0] == 100 && image.pixels[1] == 50 && image.pixels[2] == 20);
    CHECK(stbi_write_jpg(jpg.c_str(), 17, 19, 3, image.pixels.data(), 95));
    CHECK(pictor::read_image(jpg, image) && image.width == 17 && image.height == 19 && image.channels == 3);
    auto* in = std::fopen(png.c_str(), "rb"); auto* out = std::fopen(bad.c_str(), "wb");
    CHECK(in && out);
    unsigned char header[33]; CHECK(std::fread(header, 1, sizeof(header), in) == sizeof(header));
    CHECK(std::fwrite(header, 1, sizeof(header), out) == sizeof(header));
    std::fclose(in); std::fclose(out);
    CHECK(!pictor::read_image(bad, image) && image.pixels.empty());
    pictor_error error{};
    pictor_image* handle = nullptr;
    CHECK(pictor_image_load(png.c_str(), &handle, &error) == PICTOR_OK);
    auto* retained = handle;
    CHECK(pictor_image_load(jpg.c_str(), &handle, &error) == PICTOR_INVALID_ARGUMENT && handle == retained);
    pictor_image_info info;
    CHECK(pictor_image_get_info(handle, &info, sizeof(info), &error) == PICTOR_OK && info.pixels_len == 17 * 19 * 3);
    pictor_image_destroy(handle);
    std::puts("PASS: PNG alpha/RGB, JPEG decode, truncated input, C image ownership");
}
