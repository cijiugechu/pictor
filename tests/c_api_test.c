#include <pictor/pictor.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { if (!(expression)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); return 1; } } while (0)

int main(void) {
    pictor_error error = {{0}};
    pictor_request request;
    pictor_session_options options;
    pictor_session* session = NULL;
    pictor_image* image = NULL;
    CHECK(pictor_abi_version() == PICTOR_ABI_VERSION);
    CHECK(pictor_request_init(&request, sizeof(request), PICTOR_PRESET_FAST, &error) == PICTOR_OK);
    CHECK(request.width == 512 && request.height == 768 && request.steps == 3);
    CHECK(request.cache == PICTOR_CACHE_SPECTRUM && request.seed == -1);
    CHECK(pictor_request_validate(&request, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(strstr(error.message, "prompt") != NULL);
    request.prompt = "a cat";
    CHECK(pictor_request_validate(&request, &error) == PICTOR_OK && !error.message[0]);
    request.width = 513;
    CHECK(pictor_request_validate(&request, NULL) == PICTOR_INVALID_ARGUMENT);
    request.width = 512;
    request.cfg_scale = NAN;
    CHECK(pictor_request_validate(&request, &error) == PICTOR_INVALID_ARGUMENT);
    request.cfg_scale = 1;
    request.vae_tiling = 2;
    CHECK(pictor_request_validate(&request, &error) == PICTOR_INVALID_ARGUMENT);
    request.vae_tiling = 0;
    request.cache = -1;
    CHECK(pictor_request_validate(&request, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_request_init(&request, sizeof(request), PICTOR_PRESET_QUALITY, &error) == PICTOR_OK);
    CHECK(request.steps == 16 && request.cache == PICTOR_CACHE_NONE);
    request.struct_size--;
    CHECK(pictor_request_validate(&request, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_request_init(&request, sizeof(request), 42, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(request.struct_size == 0);
    memset(&request, 0x55, sizeof(request));
    pictor_request unchanged = request;
    CHECK(pictor_request_init(&request, sizeof(request) - 1, 0, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(memcmp(&request, &unchanged, sizeof(request)) == 0);
    CHECK(pictor_request_init(NULL, sizeof(request), 0, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_request_validate(NULL, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_options_init(NULL, sizeof(options), &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_options_init(&options, sizeof(options) - 1, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_options_init(&options, sizeof(options), &error) == PICTOR_OK);
    CHECK(options.threads == 0 && options.verbose == 0 && options.model_path == NULL);
    CHECK(pictor_session_create(&options, &session, &error) == PICTOR_INVALID_ARGUMENT && session == NULL);
    options.model_path = "/nonexistent-pictor-c-abi-model.gguf";
    CHECK(pictor_session_create(&options, &session, &error) == PICTOR_INVALID_ARGUMENT && session == NULL);
    CHECK(strstr(error.message, "model file not found") != NULL);
    options.threads = -1;
    CHECK(pictor_session_create(&options, &session, &error) == PICTOR_INVALID_ARGUMENT && session == NULL);
    CHECK(strstr(error.message, "threads") != NULL);
    options.threads = 0;
    options.verbose = 2;
    CHECK(pictor_session_create(&options, &session, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_create(NULL, &session, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_create(&options, NULL, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_session_generate(NULL, &request, NULL, NULL, &image, &error) == PICTOR_INVALID_ARGUMENT && image == NULL);
    CHECK(pictor_session_generate(NULL, NULL, NULL, NULL, NULL, &error) == PICTOR_INVALID_ARGUMENT);
    double seconds = 123;
    CHECK(pictor_session_load_seconds(NULL, &seconds, &error) == PICTOR_INVALID_ARGUMENT && seconds == 0);
    CHECK(pictor_session_load_seconds(NULL, NULL, &error) == PICTOR_INVALID_ARGUMENT);
    pictor_image_info info;
    memset(&info, 0x55, sizeof(info));
    CHECK(pictor_image_get_info(NULL, &info, sizeof(info), &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(info.pixels == NULL && info.pixels_len == 0 && info.width == 0);
    CHECK(pictor_image_get_info(NULL, NULL, sizeof(info), &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_image_get_info(NULL, &info, 0, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_image_write_png(NULL, "unused.png", &error) == PICTOR_INVALID_ARGUMENT);
    pictor_image_destroy(NULL);
    pictor_session_destroy(NULL);
    pictor_flux_klein_options klein;
    CHECK(pictor_flux_klein_options_init(NULL, sizeof(klein), &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_options_init(&klein, sizeof(klein) - 1, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_options_init(&klein, sizeof(klein), &error) == PICTOR_OK && !error.message[0]);
    CHECK(klein.struct_size == sizeof(klein) && klein.diffusion_model_path == NULL && klein.text_encoder_path == NULL && klein.vae_path == NULL);
    CHECK(pictor_flux_klein_session_create(&klein, &session, &error) == PICTOR_INVALID_ARGUMENT && session == NULL);
    CHECK(pictor_flux_klein_session_create(NULL, &session, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_session_create(&klein, NULL, &error) == PICTOR_INVALID_ARGUMENT);
    klein.diffusion_model_path = "/nonexistent-klein-test-model.gguf";
    klein.text_encoder_path = "qwen.gguf";
    klein.vae_path = "vae.safetensors";
    CHECK(pictor_flux_klein_session_create(&klein, &session, &error) == PICTOR_INVALID_ARGUMENT && session == NULL);
    CHECK(strstr(error.message, "model file not found") != NULL);
    klein.verbose = 2;
    CHECK(pictor_flux_klein_session_create(&klein, &session, &error) == PICTOR_INVALID_ARGUMENT);
    klein.verbose = 0;
    klein.struct_size--;
    CHECK(pictor_flux_klein_session_create(&klein, &session, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_request_init(NULL, sizeof(request), &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_request_init(&request, sizeof(request) - 1, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_request_init(&request, sizeof(request), &error) == PICTOR_OK);
    CHECK(request.width == 512 && request.height == 512 && request.steps == 4 && request.cfg_scale == 1 && request.cache == PICTOR_CACHE_NONE);
    request.prompt = "fox";
    CHECK(pictor_request_validate(&request, &error) == PICTOR_OK);
    // Existing initializers retain Anima semantics, even after using the extension.
    CHECK(pictor_request_init(&request, sizeof(request), PICTOR_PRESET_BALANCED, &error) == PICTOR_OK);
    CHECK(request.height == 768 && request.steps == 8 && request.cache == PICTOR_CACHE_SPECTRUM);
    puts("PASS: C ABI validation, errors, sized structs, NULL handling");
    pictor_flux_klein_edit_options edit;
    CHECK(pictor_flux_klein_edit_options_init(&edit, sizeof(edit), &error) == PICTOR_OK);
    CHECK(edit.auto_resize == 1 && edit.reference_images == NULL && edit.reference_images_count == 0);
    CHECK(pictor_flux_klein_edit_options_init(&edit, sizeof(edit) - 1, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_edit_options_init(NULL, sizeof(edit), &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_flux_klein_session_edit(NULL, &request, &edit, NULL, NULL, &image, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_image_load(NULL, &image, &error) == PICTOR_INVALID_ARGUMENT);
    CHECK(pictor_image_load("/nonexistent-ref.png", &image, &error) == PICTOR_IO_ERROR && image == NULL);
    CHECK(pictor_image_load(__FILE__, &image, &error) == PICTOR_INVALID_ARGUMENT && image == NULL);
    return 0;
}
