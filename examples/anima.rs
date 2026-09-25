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
type Progress = Option<unsafe extern "C" fn(i32, i32, f32, *mut c_void)>;

#[link(name = "pictor")]
unsafe extern "C" {
    fn pictor_abi_version() -> u32;
    fn pictor_session_options_init(out: *mut Options, size: usize, error: *mut Error) -> i32;
    fn pictor_request_init(out: *mut Request, size: usize, preset: i32, error: *mut Error) -> i32;
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
    fn pictor_session_destroy(session: *mut Session);
    fn pictor_image_get_info(
        image: *const Image,
        out: *mut ImageInfo,
        size: usize,
        error: *mut Error,
    ) -> i32;
    fn pictor_image_write_png(image: *const Image, path: *const c_char, error: *mut Error) -> i32;
    fn pictor_image_destroy(image: *mut Image);
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
    let model_arg = std::env::args().nth(1);
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
        check(
            pictor_request_init(&mut request, size_of::<Request>(), 0, &mut error),
            &error,
        )?;
        request.prompt = prompt.as_ptr();
        request.seed = 666;
        request.cache = 0;
        check(pictor_request_validate(&request, &mut error), &error)?;
        options.model_path = model.as_ptr();
        let mut session = SessionOwner(ptr::null_mut());
        let status = pictor_session_create(&options, &mut session.0, &mut error);
        if model_arg.is_none() {
            if status != 1 || !session.0.is_null() || error.message[0] == 0 {
                return Err("unexpected error result".into());
            }
            println!("PASS: Rust C ABI layout, validation and errors");
            return Ok(());
        }
        check(status, &error)?;
        let mut image = ImageOwner(ptr::null_mut());
        let mut calls: usize = 0;
        check(
            pictor_session_generate(
                session.0,
                &request,
                Some(progress),
                (&mut calls as *mut usize).cast(),
                &mut image.0,
                &mut error,
            ),
            &error,
        )?;
        let mut info: ImageInfo = zeroed();
        check(
            pictor_image_get_info(image.0, &mut info, size_of::<ImageInfo>(), &mut error),
            &error,
        )?;
        if info.pixels.is_null()
            || info.pixels_len != 512 * 768 * 3
            || info.seed != 666
            || calls == 0
        {
            return Err("unexpected image".into());
        }
        let pixels = std::slice::from_raw_parts(info.pixels, info.pixels_len); // borrowed from image
        check(
            pictor_image_write_png(image.0, c"outputs/rust.png".as_ptr(), &mut error),
            &error,
        )?;
        println!("wrote outputs/rust.png ({} RGB bytes)", pixels.len());
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
