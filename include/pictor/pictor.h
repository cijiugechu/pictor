#ifndef PICTOR_PICTOR_H
#define PICTOR_PICTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define PICTOR_NOEXCEPT noexcept
extern "C" {
#else
#define PICTOR_NOEXCEPT
#endif

#define PICTOR_ABI_VERSION 1u

typedef int32_t pictor_status;
#define PICTOR_OK 0
#define PICTOR_INVALID_ARGUMENT 1
#define PICTOR_IO_ERROR 2
#define PICTOR_BACKEND_ERROR 3
#define PICTOR_OUT_OF_MEMORY 4

#define PICTOR_PRESET_FAST 0
#define PICTOR_PRESET_BALANCED 1
#define PICTOR_PRESET_QUALITY 2
#define PICTOR_CACHE_NONE 0
#define PICTOR_CACHE_SPECTRUM 1

typedef struct pictor_session pictor_session;
typedef struct pictor_image pictor_image;

/* Optional caller-owned error buffer. Each status-returning call clears it on
 * success or fills a NUL-terminated, possibly truncated message on failure.
 * Do not share an error buffer between concurrent calls. */
typedef struct pictor_error { char message[512]; } pictor_error;

typedef struct pictor_session_options {
    size_t struct_size;
    const char* model_path;
    int32_t threads;
    uint32_t verbose; /* 0 or 1 */
} pictor_session_options;

/* Additive ABI v1 extension for Klein 4B distilled text-to-image.
 * Original Anima options and request layouts remain unchanged. */
typedef struct pictor_flux_klein_options {
    size_t struct_size;
    const char* diffusion_model_path;
    const char* text_encoder_path;
    const char* vae_path;
    int32_t threads;
    uint32_t verbose; /* 0 or 1 */
} pictor_flux_klein_options;

typedef struct pictor_request {
    size_t struct_size;
    const char* prompt;
    const char* negative_prompt; /* NULL means empty */
    int32_t width;
    int32_t height;
    int32_t steps;
    float cfg_scale;
    int64_t seed;
    int32_t cache;
    uint32_t vae_tiling; /* 0 or 1 */
} pictor_request;

typedef struct pictor_image_info {
    int32_t width;
    int32_t height;
    int32_t channels;
    const uint8_t* pixels;
    size_t pixels_len;
    int64_t seed;
    double generation_seconds;
} pictor_image_info;

/* Borrowed immutable, tightly packed RGB8; pixels_len must equal width*height*3.
 * Set struct_size=sizeof(pictor_image_view); valid for the complete edit call. */
typedef struct pictor_image_view {
    size_t struct_size;
    int32_t width;
    int32_t height;
    const uint8_t* pixels;
    size_t pixels_len;
} pictor_image_view;

typedef struct pictor_flux_klein_edit_options {
    size_t struct_size;
    const pictor_image_view* reference_images;
    size_t reference_images_count; /* 1..4, in prompt order */
    uint32_t auto_resize; /* 0 or 1; default 1 */
} pictor_flux_klein_edit_options;

/* Called synchronously on the generating thread; step ranges from 0 to steps.
 * Step 0 is the initial event. Must not re-enter pictor,
 * throw, unwind, longjmp, or let a Rust panic cross this boundary. */
typedef void (*pictor_progress_callback)(int32_t step, int32_t steps, float seconds, void* userdata);

/* ABI v1 layouts/signatures are fixed; future incompatible APIs will be versioned.
 * Strings are UTF-8, NUL-terminated, borrowed only until the call returns.
 * Sized structs must be initialized with these functions and sizeof(*output).
 * All pointers must refer to valid, suitably aligned objects of the stated type.
 * NULL is rejected except for optional error/callback/userdata/negative_prompt.
 * A non-OK status never returns a partial session or image. */
uint32_t pictor_abi_version(void) PICTOR_NOEXCEPT;
pictor_status pictor_session_options_init(pictor_session_options* output, size_t size, pictor_error* error) PICTOR_NOEXCEPT;
pictor_status pictor_request_init(pictor_request* output, size_t size, int32_t preset, pictor_error* error) PICTOR_NOEXCEPT;
/* Validates common fields. Session-specific restrictions are checked by generate. */
pictor_status pictor_request_validate(const pictor_request* request, pictor_error* error) PICTOR_NOEXCEPT;

/* Defaults: 512x512, 4 steps, CFG 1, cache=none; Euler/discrete sampling.
 * The resulting handle uses the existing session/image APIs below.
 * Klein rejects caches other than NONE in this release. */
pictor_status pictor_flux_klein_options_init(pictor_flux_klein_options* output, size_t size, pictor_error* error) PICTOR_NOEXCEPT;
pictor_status pictor_flux_klein_request_init(pictor_request* output, size_t size, pictor_error* error) PICTOR_NOEXCEPT;
pictor_status pictor_flux_klein_session_create(const pictor_flux_klein_options* options, pictor_session** output, pictor_error* error) PICTOR_NOEXCEPT;
pictor_status pictor_flux_klein_edit_options_init(pictor_flux_klein_edit_options* output, size_t size, pictor_error* error) PICTOR_NOEXCEPT;
/* Klein sessions only. Same ownership, output-slot and callback rules as generate.
 * Reference images condition the generation; this is not masked inpainting. */
pictor_status pictor_flux_klein_session_edit(pictor_session* session, const pictor_request* request,
    const pictor_flux_klein_edit_options* options, pictor_progress_callback progress, void* userdata,
    pictor_image** output, pictor_error* error) PICTOR_NOEXCEPT;

/* *output must be NULL; it remains NULL on failure. Do not destroy a session
 * during an active call. Backend calls across all sessions are serialized. */
pictor_status pictor_session_create(const pictor_session_options* options, pictor_session** output, pictor_error* error) PICTOR_NOEXCEPT;
pictor_status pictor_session_load_seconds(const pictor_session* session, double* output, pictor_error* error) PICTOR_NOEXCEPT;
pictor_status pictor_session_generate(pictor_session* session, const pictor_request* request,
    pictor_progress_callback progress, void* userdata, pictor_image** output, pictor_error* error) PICTOR_NOEXCEPT;
void pictor_session_destroy(pictor_session* session) PICTOR_NOEXCEPT; /* NULL is a no-op */

/* Images own immutable, tightly packed RGB8 data independent of their session.
 * get_info borrows pixels until image_destroy; never free pixels yourself.
 * Info is zeroed on failure if its pointer and size are valid. */
pictor_status pictor_image_get_info(const pictor_image* image, pictor_image_info* output, size_t size, pictor_error* error) PICTOR_NOEXCEPT;
/* Creates parent directories and overwrites an existing PNG, like the C++ API. */
pictor_status pictor_image_write_png(const pictor_image* image, const char* path, pictor_error* error) PICTOR_NOEXCEPT;
/* PNG/JPEG -> owned RGB8; *output must be NULL. Alpha is discarded; no EXIF rotation.
 * No model is required. Use get_info to borrow pixels for a reference view. */
pictor_status pictor_image_load(const char* path, pictor_image** output, pictor_error* error) PICTOR_NOEXCEPT;
void pictor_image_destroy(pictor_image* image) PICTOR_NOEXCEPT; /* NULL is a no-op */

#ifdef __cplusplus
}
#endif
#undef PICTOR_NOEXCEPT
#endif
