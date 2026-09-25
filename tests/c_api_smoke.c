#include <pictor/pictor.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression) do { if (!(expression)) { \
    fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #expression, error.message); goto cleanup; } } while (0)

struct progress_state { int calls; int last_step; int invalid; };
static void progress(int32_t step, int32_t steps, float seconds, void* userdata) {
    struct progress_state* state = userdata;
    state->calls++;
    state->last_step = step;
    // The backend emits an initial step 0 before denoising starts.
    if (step < 0 || step > steps || steps != 3 || seconds < 0) state->invalid = 1;
    fprintf(stderr, "C ABI step %d/%d\n", step, steps);
}

int main(int argc, char** argv) {
    int result = 1;
    pictor_error error = {{0}};
    pictor_session_options options;
    pictor_request request;
    pictor_session* session = NULL;
    pictor_image* first = NULL;
    pictor_image* second = NULL;
    pictor_image_info a, b;
    struct progress_state state = {0};
    double load_seconds = 0;
    CHECK(pictor_session_options_init(&options, sizeof(options), &error) == PICTOR_OK);
    options.model_path = argc > 1 ? argv[1] : "models/Anima-P3-Turbo-AIO-Q4_K.gguf";
    CHECK(pictor_session_create(&options, &session, &error) == PICTOR_OK);
    CHECK(pictor_session_load_seconds(session, &load_seconds, &error) == PICTOR_OK && load_seconds > 0);
    CHECK(pictor_request_init(&request, sizeof(request), PICTOR_PRESET_FAST, &error) == PICTOR_OK);
    request.prompt = "masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people";
    request.seed = 666;
    request.cache = PICTOR_CACHE_NONE;
    // Invalid request must leave the resident session usable and output empty.
    request.width = 513;
    CHECK(pictor_session_generate(session, &request, progress, &state, &first, &error) == PICTOR_INVALID_ARGUMENT && first == NULL);
    request.width = 512;
    CHECK(pictor_session_generate(session, &request, progress, &state, &first, &error) == PICTOR_OK);
    CHECK(state.calls > 0 && state.last_step == 3 && !state.invalid);
    CHECK(pictor_session_generate(session, &request, NULL, NULL, &second, &error) == PICTOR_OK);
    // Refuse replacing a live handle, without losing it.
    pictor_image* retained = first;
    CHECK(pictor_session_generate(session, &request, NULL, NULL, &first, &error) == PICTOR_INVALID_ARGUMENT && first == retained);
    pictor_session* retained_session = session;
    CHECK(pictor_session_create(&options, &session, &error) == PICTOR_INVALID_ARGUMENT && session == retained_session);
    pictor_session_destroy(session);
    session = NULL;
    // Both images and their pixel storage must outlive the session.
    CHECK(pictor_image_get_info(first, &a, sizeof(a), &error) == PICTOR_OK);
    CHECK(pictor_image_get_info(second, &b, sizeof(b), &error) == PICTOR_OK);
    CHECK(a.width == 512 && a.height == 768 && a.channels == 3 && a.seed == 666);
    CHECK(a.pixels_len == 512u * 768u * 3u && b.pixels_len == a.pixels_len);
    CHECK(a.pixels != b.pixels && memcmp(a.pixels, b.pixels, a.pixels_len) == 0);
    CHECK(pictor_image_write_png(first, "/dev/null/pictor.png", &error) == PICTOR_IO_ERROR);
    CHECK(pictor_image_write_png(first, argc > 2 ? argv[2] : "outputs/c-abi.png", &error) == PICTOR_OK);
    printf("PASS: C ABI repeated inference, callback, error recovery, image ownership, PNG; load=%.3fs first=%.3fs repeat=%.3fs\n",
           load_seconds, a.generation_seconds, b.generation_seconds);
    result = 0;
cleanup:
    pictor_image_destroy(second);
    pictor_image_destroy(first);
    pictor_session_destroy(session);
    return result;
}
