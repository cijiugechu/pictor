#include <pictor/anima.hpp>
#include <pictor/flux_klein.hpp>
#include <cstdio>
#include <cstdlib>

#define CHECK(expression) do { if (!(expression)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); return 1; } } while (0)

int main() {
    using namespace pictor;
    std::unique_ptr<AnimaSession> session;
    CHECK(!AnimaSession::create({__FILE__}, static_cast<AnimaBackend>(99), session) && !session);
    CHECK(!AnimaSession::create({"/nonexistent-pictor-mlx"}, AnimaBackend::mlx, session) && !session);
    CHECK(anima_request().cache == CacheMode::none && anima_request(Preset::fast).steps == 3);
    auto status = AnimaSession::create({"/nonexistent-pictor-cpp-model.gguf"}, session);
    CHECK(status.code == ErrorCode::invalid_argument && !session);
    auto request = preset_request(Preset::fast);
    CHECK(!validate_request(request));
    request.prompt = "cat";
    CHECK(validate_request(request));
    Image image;
    CHECK(write_png("unused.png", image).code == ErrorCode::invalid_argument);
    image = {1, 1, 3, {1, 2, 3}, 0, 0};
    CHECK(write_png("/dev/null/pictor.png", image).code == ErrorCode::io_error);
    CHECK(write_png("", image).code == ErrorCode::invalid_argument);
    status = failure(ErrorCode::backend_error, std::string(1024, 'a'));
    CHECK(std::strlen(status.message) == 511 && status.message[511] == '\0');
    std::unique_ptr<FluxKleinSession> klein;
    CHECK(FluxKleinSession::create({__FILE__, "/nonexistent-text-encoder.gguf", __FILE__}, klein).code == ErrorCode::invalid_argument && !klein);
    CHECK(FluxKleinSession::create({__FILE__, __FILE__, "/nonexistent-vae.safetensors"}, klein).code == ErrorCode::invalid_argument && !klein);
    request = flux_klein_request();
    request.prompt = "fox";
    CHECK(validate_flux_klein_request(request));
    request.cache = CacheMode::spectrum;
    CHECK(validate_flux_klein_request(request).code == ErrorCode::invalid_argument);
    FluxKleinEditRequest edit;
    edit.generation.prompt = "winter";
    CHECK(!validate_flux_klein_edit_request(edit));
    std::vector<std::uint8_t> rgb(17 * 19 * 3, 42);
    edit.reference_images = {{17, 19, rgb.data(), rgb.size()}};
    CHECK(validate_flux_klein_edit_request(edit));
    edit.auto_resize = false;
    CHECK(!validate_flux_klein_edit_request(edit));
    edit.auto_resize = true;
    edit.reference_images[0].pixels_len--;
    CHECK(!validate_flux_klein_edit_request(edit));
    edit.reference_images = {{1, 4096, rgb.data(), 1 * 4096 * 3}};
    CHECK(!validate_flux_klein_edit_request(edit));
    CHECK(read_image("/nonexistent-pictor-ref.png", image).code == ErrorCode::io_error && image.pixels.empty());
    CHECK(read_image(__FILE__, image).code == ErrorCode::invalid_argument);
    std::puts("PASS: C++ explicit status API with exceptions disabled");
}
