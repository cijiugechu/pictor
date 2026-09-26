const std = @import("std");
const c = @cImport({
    @cInclude("pictor/pictor.h");
});

fn check(status: c.pictor_status, err: *const c.pictor_error) !void {
    if (status != c.PICTOR_OK) {
        std.debug.print("pictor error {d}: {s}\n", .{ status, std.mem.sliceTo(&err.message, 0) });
        return error.PictorFailure;
    }
}

fn progress(step: i32, steps: i32, _: f32, userdata: ?*anyopaque) callconv(.c) void {
    const count: *usize = @ptrCast(@alignCast(userdata.?));
    count.* += 1;
    std.debug.print("step {d}/{d}\n", .{ step, steps });
}

// With no arguments, checks linking and errors without weights.
// With a model path, generates outputs/zig.png through the C ABI.
// --klein [model-directory] [reference.png] selects Klein generation/editing.
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (c.pictor_abi_version() != c.PICTOR_ABI_VERSION) return error.AbiMismatch;
    var err: c.pictor_error = undefined;
    const klein = args.len > 1 and std.mem.eql(u8, args[1], "--klein");
    const model_index: usize = if (klein) 2 else 1;
    const has_model = args.len > model_index;
    const reference_path = if (klein and args.len > 3) args[3] else null;
    var options: c.pictor_session_options = undefined;
    try check(c.pictor_session_options_init(&options, @sizeOf(@TypeOf(options)), &err), &err);
    var request: c.pictor_request = undefined;
    if (klein) {
        try check(c.pictor_flux_klein_request_init(&request, @sizeOf(@TypeOf(request)), &err), &err);
    } else {
        try check(c.pictor_request_init(&request, @sizeOf(@TypeOf(request)), c.PICTOR_PRESET_FAST, &err), &err);
    }
    request.prompt = "masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people";
    if (reference_path != null) request.prompt = "Change the scene to snowy winter. Keep the subject and composition.";
    request.seed = 666;
    request.cache = c.PICTOR_CACHE_NONE;
    try check(c.pictor_request_validate(&request, &err), &err);
    var edit: c.pictor_flux_klein_edit_options = undefined;
    try check(c.pictor_flux_klein_edit_options_init(&edit, @sizeOf(@TypeOf(edit)), &err), &err);
    options.model_path = if (has_model) args[model_index].ptr else "/nonexistent-pictor-zig-model.gguf";
    var session: ?*c.pictor_session = null;
    defer c.pictor_session_destroy(session);
    const status = if (klein) blk: {
        var paths: c.pictor_flux_klein_options = undefined;
        try check(c.pictor_flux_klein_options_init(&paths, @sizeOf(@TypeOf(paths)), &err), &err);
        const directory = if (has_model) args[model_index] else "/nonexistent-pictor-klein";
        const allocator = init.arena.allocator();
        paths.diffusion_model_path = (try std.fmt.allocPrintSentinel(allocator, "{s}/flux-2-klein-4b-Q4_0.gguf", .{directory}, 0)).ptr;
        paths.text_encoder_path = (try std.fmt.allocPrintSentinel(allocator, "{s}/Qwen3-4B-Q4_K_M.gguf", .{directory}, 0)).ptr;
        paths.vae_path = (try std.fmt.allocPrintSentinel(allocator, "{s}/flux2-vae.safetensors", .{directory}, 0)).ptr;
        break :blk c.pictor_flux_klein_session_create(&paths, &session, &err);
    } else c.pictor_session_create(&options, &session, &err);
    if (!has_model) {
        if (status != c.PICTOR_INVALID_ARGUMENT or session != null or err.message[0] == 0)
            return error.UnexpectedResult;
        std.debug.print("PASS: Zig C ABI import, layout, validation and errors\n", .{});
        return;
    }
    try check(status, &err);
    var image: ?*c.pictor_image = null;
    defer c.pictor_image_destroy(image);
    var progress_calls: usize = 0;
    var reference: ?*c.pictor_image = null;
    defer c.pictor_image_destroy(reference);
    if (reference_path) |path| {
        try check(c.pictor_image_load(path.ptr, &reference, &err), &err);
        var ref_info: c.pictor_image_info = undefined;
        try check(c.pictor_image_get_info(reference, &ref_info, @sizeOf(@TypeOf(ref_info)), &err), &err);
        const view: c.pictor_image_view = .{ .struct_size = @sizeOf(c.pictor_image_view), .width = ref_info.width, .height = ref_info.height, .pixels = ref_info.pixels, .pixels_len = ref_info.pixels_len };
        edit.reference_images = &view;
        edit.reference_images_count = 1;
        try check(c.pictor_flux_klein_session_edit(session, &request, &edit, progress, &progress_calls, &image, &err), &err);
    } else {
        try check(c.pictor_session_generate(session, &request, progress, &progress_calls, &image, &err), &err);
    }
    var info: c.pictor_image_info = undefined;
    try check(c.pictor_image_get_info(image, &info, @sizeOf(@TypeOf(info)), &err), &err);
    const pixels = info.pixels[0..info.pixels_len]; // borrowed until image_destroy
    const expected_bytes: usize = if (klein) 512 * 512 * 3 else 512 * 768 * 3;
    if (pixels.len != expected_bytes or info.seed != 666 or progress_calls == 0)
        return error.UnexpectedImage;
    const output = if (reference_path != null) "outputs/zig-klein-edit.png" else if (klein) "outputs/zig-klein.png" else "outputs/zig.png";
    try check(c.pictor_image_write_png(image, output, &err), &err);
    std.debug.print("wrote {s} ({d} RGB bytes)\n", .{ output, pixels.len });
}
