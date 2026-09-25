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
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (c.pictor_abi_version() != c.PICTOR_ABI_VERSION) return error.AbiMismatch;
    var err: c.pictor_error = undefined;
    var options: c.pictor_session_options = undefined;
    try check(c.pictor_session_options_init(&options, @sizeOf(@TypeOf(options)), &err), &err);
    var request: c.pictor_request = undefined;
    try check(c.pictor_request_init(&request, @sizeOf(@TypeOf(request)), c.PICTOR_PRESET_FAST, &err), &err);
    request.prompt = "masterpiece, best quality, anime landscape, a small cottage by a lake, mountains, blue sky, no people";
    request.seed = 666;
    request.cache = c.PICTOR_CACHE_NONE;
    try check(c.pictor_request_validate(&request, &err), &err);
    options.model_path = if (args.len > 1) args[1].ptr else "/nonexistent-pictor-zig-model.gguf";
    var session: ?*c.pictor_session = null;
    defer c.pictor_session_destroy(session);
    const status = c.pictor_session_create(&options, &session, &err);
    if (args.len == 1) {
        if (status != c.PICTOR_INVALID_ARGUMENT or session != null or err.message[0] == 0)
            return error.UnexpectedResult;
        std.debug.print("PASS: Zig C ABI import, layout, validation and errors\n", .{});
        return;
    }
    try check(status, &err);
    var image: ?*c.pictor_image = null;
    defer c.pictor_image_destroy(image);
    var progress_calls: usize = 0;
    try check(c.pictor_session_generate(session, &request, progress, &progress_calls, &image, &err), &err);
    var info: c.pictor_image_info = undefined;
    try check(c.pictor_image_get_info(image, &info, @sizeOf(@TypeOf(info)), &err), &err);
    const pixels = info.pixels[0..info.pixels_len]; // borrowed until image_destroy
    if (pixels.len != 512 * 768 * 3 or info.seed != 666 or progress_calls == 0)
        return error.UnexpectedImage;
    try check(c.pictor_image_write_png(image, "outputs/zig.png", &err), &err);
    std.debug.print("wrote outputs/zig.png ({d} RGB bytes)\n", .{pixels.len});
}
