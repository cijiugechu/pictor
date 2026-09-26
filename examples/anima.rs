// Minimal dependency-free Rust bindings for the ABI v1 subset used below.
// Larger integrations can generate declarations from pictor/pictor.h.
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::{size_of, zeroed};
use std::ptr;

#[repr(C)]
struct Session {
    _private: [u8; 0],
}
#[repr(C)]
struct Image {
    _private: [u8; 0],
}
#[repr(C)]
struct Error {
    message: [c_char; 512],
}
#[repr(C)]
struct Options {
    struct_size: usize,
    model_path: *const c_char,
    threads: i32,
    verbose: u32,
}
#[repr(C)]
struct KleinOptions {
    struct_size: usize,
    diffusion_model_path: *const c_char,
    text_encoder_path: *const c_char,
    vae_path: *const c_char,
    threads: i32,
    verbose: u32,
}
#[repr(C)]
struct Request {
    struct_size: usize,
    prompt: *const c_char,
    negative_prompt: *const c_char,
    width: i32,
    height: i32,
    steps: i32,
    cfg_scale: f32,
    seed: i64,
    cache: i32,
    vae_tiling: u32,
}
#[repr(C)]
struct ImageInfo {
    width: i32,
    height: i32,
    channels: i32,
    pixels: *const u8,
    pixels_len: usize,
    seed: i64,
    generation_seconds: f64,
}
type BatchProgress = Option<unsafe extern "C" fn(i32, i32, i32, i32, f32, *mut c_void)>;
type Progress = Option<unsafe extern "C" fn(i32, i32, f32, *mut c_void)>;

#[repr(C)]
struct ImageView {
    struct_size: usize,
    width: i32,
    height: i32,
    pixels: *const u8,
    pixels_len: usize,
}
#[repr(C)]
struct EditOptions {
    struct_size: usize,
    reference_images: *const ImageView,
    reference_images_count: usize,
    auto_resize: u32,
}

#[link(name = "pictor")]
unsafe extern "C" {
    fn pictor_abi_version() -> u32;
    fn pictor_session_options_init(out: *mut Options, size: usize, error: *mut Error) -> i32;
    fn pictor_request_init(out: *mut Request, size: usize, preset: i32, error: *mut Error) -> i32;
    fn pictor_flux_klein_options_init(
        out: *mut KleinOptions,
        size: usize,
        error: *mut Error,
    ) -> i32;
    fn pictor_flux_klein_request_init(out: *mut Request, size: usize, error: *mut Error) -> i32;
    fn pictor_flux_klein_session_create_with_backend(
        options: *const KleinOptions,
        backend: i32,
        out: *mut *mut Session,
        error: *mut Error,
    ) -> i32;
    fn pictor_request_validate(request: *const Request, error: *mut Error) -> i32;
    fn pictor_session_create(
        options: *const Options,
        out: *mut *mut Session,
        error: *mut Error,
    ) -> i32;
    fn pictor_session_generate(
        session: *mut Session,
        request: *const Request,
        progress: Progress,
        userdata: *mut c_void,
        out: *mut *mut Image,
        error: *mut Error,
    ) -> i32;
    fn pictor_session_generate_batch(
        session: *mut Session,
        request: *const Request,
        count: i32,
        progress: BatchProgress,
        userdata: *mut c_void,
        outputs: *mut *mut Image,
        batch_seconds: *mut f64,
        error: *mut Error,
    ) -> i32;
    fn pictor_flux_klein_session_edit_batch(
        session: *mut Session,
        request: *const Request,
        options: *const EditOptions,
        count: i32,
        progress: BatchProgress,
        userdata: *mut c_void,
        outputs: *mut *mut Image,
        batch_seconds: *mut f64,
        error: *mut Error,
    ) -> i32;
    fn pictor_flux_klein_session_set_hidden_state_compression(
        session: *mut Session,
        enabled: u32,
        error: *mut Error,
    ) -> i32;
    fn pictor_session_destroy(session: *mut Session);
    fn pictor_image_get_info(
        image: *const Image,
        out: *mut ImageInfo,
        size: usize,
        error: *mut Error,
    ) -> i32;
    fn pictor_image_write_png(image: *const Image, path: *const c_char, error: *mut Error) -> i32;
    fn pictor_image_destroy(image: *mut Image);
    fn pictor_image_load(path: *const c_char, out: *mut *mut Image, error: *mut Error) -> i32;
    fn pictor_flux_klein_edit_options_init(
        out: *mut EditOptions,
        size: usize,
        error: *mut Error,
    ) -> i32;
    fn pictor_flux_klein_session_edit(
        session: *mut Session,
        request: *const Request,
        options: *const EditOptions,
        progress: Progress,
        userdata: *mut c_void,
        out: *mut *mut Image,
        error: *mut Error,
    ) -> i32;
}

struct SessionOwner(*mut Session);
impl Drop for SessionOwner {
    fn drop(&mut self) {
        unsafe { pictor_session_destroy(self.0) }
    }
}
struct ImageOwner(*mut Image);
impl Drop for ImageOwner {
    fn drop(&mut self) {
        unsafe { pictor_image_destroy(self.0) }
    }
}

fn check(status: i32, error: &Error) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        // The API guarantees a NUL-terminated error buffer.
        Err(format!(
            "pictor error {status}: {}",
            unsafe { CStr::from_ptr(error.message.as_ptr()) }.to_string_lossy()
        ))
    }
}

unsafe extern "C" fn progress(_: i32, _: i32, _: f32, userdata: *mut c_void) {
    // No allocation, I/O, or panic in the callback. userdata lives through generate.
    unsafe {
        *userdata.cast::<usize>() = (*userdata.cast::<usize>()).saturating_add(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    let mlx = first.as_deref() == Some("--mlx");
    let klein = mlx || first.as_deref() == Some("--klein");
    let model_arg = if klein { args.next() } else { first };
    let reference_path = if klein { args.next() } else { None };
    let model = CString::new(
        model_arg
            .as_deref()
            .unwrap_or("/nonexistent-pictor-rust-model.gguf"),
    )
    .map_err(|e| e.to_string())?;
    let prompt = c"masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people";
    // Unsafe operations remain inside this example's FFI boundary; owners release handles on every return.
    unsafe {
        if pictor_abi_version() != 1 {
            return Err("ABI mismatch".into());
        }
        let mut error: Error = zeroed();
        let mut options: Options = zeroed();
        let mut request: Request = zeroed();
        check(
            pictor_session_options_init(&mut options, size_of::<Options>(), &mut error),
            &error,
        )?;
        if klein {
            check(
                pictor_flux_klein_request_init(&mut request, size_of::<Request>(), &mut error),
                &error,
            )?;
        } else {
            check(
                pictor_request_init(&mut request, size_of::<Request>(), 0, &mut error),
                &error,
            )?;
        }
        request.prompt = prompt.as_ptr();
        if reference_path.is_some() {
            request.prompt =
                c"Change the scene to snowy winter. Keep the subject and composition.".as_ptr();
        }
        request.seed = 666;
        request.cache = 0;
        check(pictor_request_validate(&request, &mut error), &error)?;
        let mut edit: EditOptions = zeroed();
        check(
            pictor_flux_klein_edit_options_init(&mut edit, size_of::<EditOptions>(), &mut error),
            &error,
        )?;
        options.model_path = model.as_ptr();
        let mut session = SessionOwner(ptr::null_mut());
        let status = if klein {
            let directory =
                std::path::Path::new(model_arg.as_deref().unwrap_or("/nonexistent-pictor-klein"));
            let diffusion = CString::new(
                directory
                    .join(if mlx {
                        "transformer"
                    } else {
                        "flux-2-klein-4b-Q4_0.gguf"
                    })
                    .to_string_lossy()
                    .as_bytes(),
            )
            .map_err(|e| e.to_string())?;
            let encoder = CString::new(
                directory
                    .join(if mlx {
                        "text_encoder"
                    } else {
                        "Qwen3-4B-Q4_K_M.gguf"
                    })
                    .to_string_lossy()
                    .as_bytes(),
            )
            .map_err(|e| e.to_string())?;
            let vae = CString::new(
                (if mlx {
                    std::path::Path::new("models/flux2-klein-4b")
                } else {
                    directory
                })
                .join("full_encoder_small_decoder.safetensors")
                .to_string_lossy()
                .as_bytes(),
            )
            .map_err(|e| e.to_string())?;
            let mut paths: KleinOptions = zeroed();
            check(
                pictor_flux_klein_options_init(&mut paths, size_of::<KleinOptions>(), &mut error),
                &error,
            )?;
            paths.diffusion_model_path = diffusion.as_ptr();
            paths.text_encoder_path = encoder.as_ptr();
            paths.vae_path = vae.as_ptr();
            pictor_flux_klein_session_create_with_backend(
                &paths,
                if mlx { 2 } else { 1 },
                &mut session.0,
                &mut error,
            )
        } else {
            pictor_session_create(&options, &mut session.0, &mut error)
        };
        if model_arg.is_none() {
            if status != 1 || !session.0.is_null() || error.message[0] == 0 {
                return Err("unexpected error result".into());
            }
            if pictor_flux_klein_session_set_hidden_state_compression(
                ptr::null_mut(),
                1,
                &mut error,
            ) != 1
            {
                return Err("unexpected HS error result".into());
            }
            let mut images = [ptr::null_mut(); 2];
            let mut batch_seconds = 123.0;
            if pictor_session_generate_batch(
                ptr::null_mut(),
                &request,
                2,
                None,
                ptr::null_mut(),
                images.as_mut_ptr(),
                &mut batch_seconds,
                &mut error,
            ) != 1
                || batch_seconds != 0.0
                || images.iter().any(|image| !image.is_null())
            {
                return Err("unexpected batch error result".into());
            }
            if pictor_flux_klein_session_edit_batch(
                ptr::null_mut(),
                &request,
                &edit,
                2,
                None,
                ptr::null_mut(),
                images.as_mut_ptr(),
                &mut batch_seconds,
                &mut error,
            ) != 1
            {
                return Err("unexpected edit batch error result".into());
            }
            println!("PASS: Rust C ABI layout, validation and errors");
            return Ok(());
        }
        check(status, &error)?;
        let mut image = ImageOwner(ptr::null_mut());
        let mut calls: usize = 0;
        let mut reference = ImageOwner(ptr::null_mut());
        let generated = if let Some(path) = reference_path.as_deref() {
            let path = CString::new(path).map_err(|e| e.to_string())?;
            check(
                pictor_image_load(path.as_ptr(), &mut reference.0, &mut error),
                &error,
            )?;
            let mut info: ImageInfo = zeroed();
            check(
                pictor_image_get_info(reference.0, &mut info, size_of::<ImageInfo>(), &mut error),
                &error,
            )?;
            let view = ImageView {
                struct_size: size_of::<ImageView>(),
                width: info.width,
                height: info.height,
                pixels: info.pixels,
                pixels_len: info.pixels_len,
            };
            edit.reference_images = &view;
            edit.reference_images_count = 1;
            pictor_flux_klein_session_edit(
                session.0,
                &request,
                &edit,
                Some(progress),
                (&mut calls as *mut usize).cast(),
                &mut image.0,
                &mut error,
            )
        } else {
            pictor_session_generate(
                session.0,
                &request,
                Some(progress),
                (&mut calls as *mut usize).cast(),
                &mut image.0,
                &mut error,
            )
        };
        check(generated, &error)?;
        let mut info: ImageInfo = zeroed();
        check(
            pictor_image_get_info(image.0, &mut info, size_of::<ImageInfo>(), &mut error),
            &error,
        )?;
        let expected_bytes = if klein { 512 * 512 * 3 } else { 512 * 768 * 3 };
        if info.pixels.is_null()
            || info.pixels_len != expected_bytes
            || info.seed != 666
            || calls == 0
        {
            return Err("unexpected image".into());
        }
        let pixels = std::slice::from_raw_parts(info.pixels, info.pixels_len); // borrowed from image
        check(
            pictor_image_write_png(
                image.0,
                (if reference_path.is_some() {
                    c"outputs/rust-klein-edit.png"
                } else if klein {
                    c"outputs/rust-klein.png"
                } else {
                    c"outputs/rust.png"
                })
                .as_ptr(),
                &mut error,
            ),
            &error,
        )?;
        println!(
            "wrote {} ({} RGB bytes)",
            if reference_path.is_some() {
                "outputs/rust-klein-edit.png"
            } else if klein {
                "outputs/rust-klein.png"
            } else {
                "outputs/rust.png"
            },
            pixels.len()
        );
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
