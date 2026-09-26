const std = @import("std");
const builtin = @import("builtin");

const Backend = enum { metal, cpu };

const Project = struct {
    b: *std.Build,
    optimize: std.builtin.OptimizeMode,
    prepare: *std.Build.Step,
    sdk: ?[]const u8,

    fn cpp(self: Project, sources: []const []const u8, exceptions: bool) *std.Build.Module {
        const b = self.b;
        const module = b.createModule(.{
            .target = b.graph.host,
            .optimize = self.optimize,
            .link_libc = true,
            .link_libcpp = self.sdk == null,
            .pic = true,
        });
        module.addIncludePath(b.path("include"));
        module.addIncludePath(b.path("src"));
        module.addIncludePath(b.path("vendor/stable-diffusion.cpp/include"));
        module.addSystemIncludePath(b.path("vendor/stable-diffusion.cpp/thirdparty"));
        module.addSystemIncludePath(b.path("vendor/spdlog/include"));
        if (self.sdk) |sdk| {
            // Match Apple's C++ ABI without Zig's bundled libc++.
            module.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/include/c++/v1" }) });
            module.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr/lib/libc++.tbd" }) });
        }
        self.cppSources(module, sources, exceptions);
        return module;
    }

    fn cppSources(self: Project, module: *std.Build.Module, sources: []const []const u8, exceptions: bool) void {
        const b = self.b;
        var flags: std.ArrayList([]const u8) = .empty;
        flags.appendSlice(b.allocator, &.{ "-std=c++17", "-Wall", "-Wextra", "-Wpedantic" }) catch @panic("OOM");
        if (!exceptions) flags.appendSlice(b.allocator, &.{ "-fno-exceptions", "-DSPDLOG_NO_EXCEPTIONS", "-DFMT_EXCEPTIONS=0" }) catch @panic("OOM");
        if (self.sdk != null) {
            flags.append(b.allocator, "-nostdinc++") catch @panic("OOM");
        }
        module.addCSourceFiles(.{ .files = sources, .flags = flags.items });
    }

    fn executable(self: Project, name: []const u8, sources: []const []const u8) *std.Build.Step.Compile {
        const exe = self.b.addExecutable(.{ .name = name, .root_module = self.cpp(sources, false) });
        exe.each_lib_rpath = false;
        exe.step.dependOn(self.prepare);
        exe.root_module.addRPathSpecial(if (self.sdk != null) "@loader_path/../lib" else "$ORIGIN/../lib");
        return exe;
    }

    fn cExecutable(self: Project, name: []const u8, source: []const u8) *std.Build.Step.Compile {
        const module = self.b.createModule(.{ .target = self.b.graph.host, .optimize = self.optimize, .link_libc = true });
        module.addIncludePath(self.b.path("include"));
        module.addCSourceFile(.{ .file = self.b.path(source), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Wpedantic" } });
        module.addRPathSpecial(if (self.sdk != null) "@loader_path/../lib" else "$ORIGIN/../lib");
        const exe = self.b.addExecutable(.{ .name = name, .root_module = module });
        exe.each_lib_rpath = false;
        return exe;
    }

    fn installedRun(self: Project, exe: *std.Build.Step.Compile) *std.Build.Step.Run {
        const install = self.b.addInstallArtifact(exe, .{});
        const run = self.b.addSystemCommand(&.{self.b.getInstallPath(.bin, exe.name)});
        run.step.dependOn(&install.step);
        run.step.dependOn(self.b.getInstallStep());
        run.addFileInput(exe.getEmittedBin());
        return run;
    }
};

pub fn build(b: *std.Build) void {
    const backend = b.option(Backend, "backend", "Inference backend (metal on macOS, cpu elsewhere)") orelse
        (if (builtin.os.tag == .macos) Backend.metal else Backend.cpu);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Project/backend build mode (default ReleaseFast)") orelse .ReleaseFast;
    const jobs = b.option(u32, "jobs", "Upstream compiler parallelism (default 4)") orelse 4;
    if (backend == .metal and builtin.os.tag != .macos) @panic("Metal requires macOS; use -Dbackend=cpu");
    const build_type = switch (optimize) {
        .Debug => "Debug",
        .ReleaseSafe => "RelWithDebInfo",
        .ReleaseFast => "Release",
        .ReleaseSmall => "MinSizeRel",
    };
    const sdk: ?[]const u8 = if (builtin.os.tag == .macos)
        std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }), " \r\n")
    else
        null;
    const prepare = b.addSystemCommand(&.{ "bash", b.pathFromRoot("scripts/prepare-deps.sh") });
    const project: Project = .{ .b = b, .optimize = optimize, .prepare = &prepare.step, .sdk = sdk };

    // Only the upstream dependency uses CMake. All pictor targets below use Zig.
    const backend_dir = b.pathFromRoot(b.fmt(".zig-cache/backend/{s}-{s}", .{ @tagName(backend), build_type }));
    const configure = b.addSystemCommand(&.{
        "cmake",
        "-S",
        b.pathFromRoot("vendor/stable-diffusion.cpp"),
        "-B",
        backend_dir,
        "-G",
        "Ninja",
        b.fmt("-DCMAKE_BUILD_TYPE={s}", .{build_type}),
        b.fmt("-DSD_METAL={s}", .{if (backend == .metal) "ON" else "OFF"}),
        b.fmt("-DGGML_METAL={s}", .{if (backend == .metal) "ON" else "OFF"}),
        b.fmt("-DGGML_METAL_EMBED_LIBRARY={s}", .{if (backend == .metal) "ON" else "OFF"}),
        "-DSD_BUILD_EXAMPLES=OFF",
        "-DSD_BUILD_SHARED_LIBS=ON",
        "-DSD_BUILD_SHARED_GGML_LIB=OFF",
        "-DSD_WEBP=OFF",
        "-DSD_WEBM=OFF",
        "-DGGML_BACKEND_DL=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_INSTALL_NAME_DIR=@rpath",
        "-DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON",
    });
    configure.step.dependOn(&prepare.step);
    const backend_build = b.addSystemCommand(&.{
        "cmake",
        b.fmt("-DNATIVE_DIR={s}", .{backend_dir}),
        b.fmt("-DJOBS={d}", .{jobs}),
        "-DNATIVE_TARGET=stable-diffusion",
        "-P",
        b.pathFromRoot("cmake/BuildNative.cmake"),
    });
    backend_build.step.dependOn(&configure.step);
    const backend_name = if (sdk != null) "libstable-diffusion.dylib" else "libstable-diffusion.so";
    const backend_file: std.Build.LazyPath = .{ .cwd_relative = b.pathJoin(&.{ backend_dir, "bin", backend_name }) };
    const install_backend = b.addInstallFileWithDir(backend_file, .lib, backend_name);
    install_backend.step.dependOn(&backend_build.step);
    b.getInstallStep().dependOn(&install_backend.step);

    const core = b.addLibrary(.{
        .name = "pictor",
        .linkage = .dynamic,
        .root_module = project.cpp(&.{ "src/anima.cpp", "src/flux_klein.cpp", "src/session.cpp", "src/options.cpp", "src/png.cpp", "src/image.cpp", "src/logging.cpp", "src/c_api.cpp" }, false),
    });
    // The upstream C++ backend still throws. Only this small boundary catches it.
    project.cppSources(core.root_module, &.{"src/backend.cpp"}, true);
    core.each_lib_rpath = false;
    core.root_module.addObjectFile(backend_file);
    core.root_module.addRPathSpecial(if (sdk != null) "@loader_path" else "$ORIGIN");
    core.step.dependOn(&backend_build.step);
    b.installArtifact(core);

    const exe = project.executable("pictor", &.{ "src/main.cpp", "src/cli.cpp" });
    // Explicit shared-object linkage keeps build-cache paths out of installed rpaths.
    exe.root_module.addObjectFile(core.getEmittedBin());
    b.installArtifact(exe);
    b.installDirectory(.{ .source_dir = b.path("include/pictor"), .install_dir = .header, .install_subdir = "pictor" });
    b.installFile("THIRD_PARTY_NOTICES.md", "share/pictor/THIRD_PARTY_NOTICES.md");
    const notices = [_][2][]const u8{
        .{ "vendor/stable-diffusion.cpp/LICENSE", "stable-diffusion.cpp.txt" },
        .{ "vendor/stable-diffusion.cpp/ggml/LICENSE", "ggml.txt" },
        .{ "vendor/stable-diffusion.cpp/thirdparty/stb_image_write.h", "stb_image_write.h" },
        .{ "vendor/stable-diffusion.cpp/thirdparty/stb_image.h", "stb_image.h" },
        .{ "vendor/stable-diffusion.cpp/thirdparty/zip.h", "zip.h" },
        .{ "vendor/stable-diffusion.cpp/thirdparty/miniz.h", "miniz.h" },
        .{ "vendor/stable-diffusion.cpp/thirdparty/LICENSE.darts_clone.txt", "darts_clone.txt" },
        .{ "vendor/spdlog/LICENSE", "spdlog.txt" },
    };
    for (notices) |notice| {
        const install_notice = b.addInstallFile(b.path(notice[0]), b.fmt("share/pictor/licenses/{s}", .{notice[1]}));
        install_notice.step.dependOn(&prepare.step);
        b.getInstallStep().dependOn(&install_notice.step);
    }

    const run = b.addSystemCommand(&.{b.getInstallPath(.bin, "pictor")});
    run.step.dependOn(b.getInstallStep());
    if (b.args) |args| run.addArgs(args);
    b.step("run", "Run the installed CLI (arguments after --)").dependOn(&run.step);

    const tests = project.executable("pictor_tests", &.{ "tests/options_test.cpp", "src/options.cpp", "src/cli.cpp" });
    const test_step = b.step("test", "Run C/C++ ABI, backend boundary, option and CLI/logging tests without weights");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    const c_test = project.cExecutable("pictor_c_tests", "tests/c_api_test.c");
    c_test.root_module.addObjectFile(core.getEmittedBin());
    test_step.dependOn(&project.installedRun(c_test).step);
    const cpp_test = project.executable("pictor_cpp_tests", &.{"tests/cpp_api_test.cpp"});
    cpp_test.root_module.addObjectFile(core.getEmittedBin());
    test_step.dependOn(&project.installedRun(cpp_test).step);
    const image_test = project.executable("pictor_image_tests", &.{"tests/image_test.cpp"});
    image_test.root_module.addObjectFile(core.getEmittedBin());
    test_step.dependOn(&project.installedRun(image_test).step);
    const backend_test = project.executable("pictor_backend_tests", &.{"tests/backend_test.cpp"});
    project.cppSources(backend_test.root_module, &.{ "src/backend.cpp", "tests/backend_faults.cpp" }, true);
    test_step.dependOn(&b.addRunArtifact(backend_test).step);

    const session_test = project.executable("pictor_session_tests", &.{ "tests/session_test.cpp", "tests/session_backend.cpp", "src/anima.cpp", "src/flux_klein.cpp", "src/session.cpp", "src/options.cpp", "src/logging.cpp", "src/c_api.cpp", "src/image.cpp", "src/png.cpp" });
    project.cppSources(session_test.root_module, &.{"src/backend.cpp"}, true);
    test_step.dependOn(&b.addRunArtifact(session_test).step);

    const klein_smoke_exe = project.executable("pictor_klein_smoke", &.{"tests/klein_smoke.cpp"});
    klein_smoke_exe.root_module.addObjectFile(core.getEmittedBin());
    const klein_smoke = project.installedRun(klein_smoke_exe);
    if (b.args) |args| klein_smoke.addArgs(args);
    b.step("smoke-klein", "Klein resident/C++/C ABI inference checks (model/output directories after --)").dependOn(&klein_smoke.step);
    const edit_smoke_exe = project.executable("pictor_klein_edit_smoke", &.{"tests/klein_edit_smoke.cpp"});
    edit_smoke_exe.root_module.addObjectFile(core.getEmittedBin());
    const edit_smoke = project.installedRun(edit_smoke_exe);
    if (b.args) |args| edit_smoke.addArgs(args);
    b.step("smoke-klein-edit", "Klein reference edit checks (model dir, reference PNG/JPEG, output dir after --)").dependOn(&edit_smoke.step);

    const c_smoke = project.cExecutable("pictor_c_smoke", "tests/c_api_smoke.c");
    c_smoke.root_module.addObjectFile(core.getEmittedBin());
    const c_smoke_run = project.installedRun(c_smoke);
    if (b.args) |args| c_smoke_run.addArgs(args);
    b.step("smoke-c", "Check C ABI inference/ownership (optional model and PNG paths after --)").dependOn(&c_smoke_run.step);

    const zig_module = b.createModule(.{ .root_source_file = b.path("examples/anima.zig"), .target = b.graph.host, .optimize = optimize, .link_libc = true });
    zig_module.addIncludePath(b.path("include"));
    zig_module.addObjectFile(core.getEmittedBin());
    zig_module.addRPathSpecial(if (sdk != null) "@loader_path/../lib" else "$ORIGIN/../lib");
    const zig_example = b.addExecutable(.{ .name = "pictor_zig_example", .root_module = zig_module });
    zig_example.each_lib_rpath = false;
    const ffi_test = b.step("test-ffi", "Build and run Zig/Rust consumers without weights (requires rustc)");
    ffi_test.dependOn(&project.installedRun(zig_example).step);
    const zig_klein_test = project.installedRun(zig_example);
    zig_klein_test.addArg("--klein");
    ffi_test.dependOn(&zig_klein_test.step);
    const rust = b.addSystemCommand(&.{ "rustc", "--edition=2024" });
    rust.addFileArg(b.path("examples/anima.rs"));
    rust.addArgs(&.{ b.fmt("-Lnative={s}", .{b.getInstallPath(.lib, "")}), b.fmt("-Clink-arg=-Wl,-rpath,{s}", .{if (sdk != null) "@loader_path/../lib" else "$ORIGIN/../lib"}), "-o" });
    const rust_bin = rust.addOutputFileArg("pictor_rust_example");
    rust.step.dependOn(b.getInstallStep());
    rust.addFileInput(core.getEmittedBin());
    const install_rust = b.addInstallFileWithDir(rust_bin, .bin, "pictor_rust_example");
    const rust_run = b.addSystemCommand(&.{b.getInstallPath(.bin, "pictor_rust_example")});
    rust_run.step.dependOn(&install_rust.step);
    rust_run.addFileInput(rust_bin);
    ffi_test.dependOn(&rust_run.step);
    const rust_klein_run = b.addSystemCommand(&.{ b.getInstallPath(.bin, "pictor_rust_example"), "--klein" });
    rust_klein_run.step.dependOn(&install_rust.step);
    rust_klein_run.addFileInput(rust_bin);
    ffi_test.dependOn(&rust_klein_run.step);
    const help = b.addSystemCommand(&.{ b.getInstallPath(.bin, "pictor"), "--help" });
    help.step.dependOn(b.getInstallStep());
    help.addFileInput(exe.getEmittedBin());
    help.expectStdOutMatch("Usage:\n  pictor anima");
    help.expectStdErrEqual("");
    test_step.dependOn(&help.step);
    const invalid = b.addSystemCommand(&.{ b.getInstallPath(.bin, "pictor"), "anima", "-p", "cat", "--width", "513" });
    invalid.step.dependOn(b.getInstallStep());
    invalid.addFileInput(exe.getEmittedBin());
    invalid.expectExitCode(2);
    invalid.expectStdOutEqual("");
    invalid.expectStdErrMatch("[pictor] [error] width and height must be multiples of 16");
    test_step.dependOn(&invalid.step);
    const missing = b.addSystemCommand(&.{
        b.getInstallPath(.bin, "pictor"), "anima",                                             "-p", "cat", "--model", "/nonexistent-pictor-test-model.gguf",
        "--output",                       b.pathFromRoot(".zig-cache/missing-model-test.png"),
    });
    missing.step.dependOn(b.getInstallStep());
    missing.addFileInput(exe.getEmittedBin());
    missing.expectExitCode(2);
    missing.expectStdOutEqual("");
    missing.expectStdErrMatch("[pictor] [error] model file not found");
    test_step.dependOn(&missing.step);
    const missing_ref = b.addSystemCommand(&.{ b.getInstallPath(.bin, "pictor"), "flux-klein", "-p", "winter", "--ref-image", "/nonexistent-pictor-reference.png", "--diffusion-model", "/nonexistent-model.gguf", "--output", b.pathFromRoot(".zig-cache/missing-reference-test.png") });
    missing_ref.step.dependOn(b.getInstallStep());
    missing_ref.addFileInput(exe.getEmittedBin());
    missing_ref.expectExitCode(1);
    missing_ref.expectStdOutEqual("");
    missing_ref.expectStdErrMatch("[pictor] [error] image file not found or inaccessible");
    test_step.dependOn(&missing_ref.step);

    const log_test = project.executable("pictor_logging_tests", &.{ "tests/logging_test.cpp", "src/logging.cpp" });
    const log_run = b.addRunArtifact(log_test);
    log_run.expectStdOutEqual("");
    log_run.expectStdErrMatch("[pictor] [info] literal {braces}");
    log_run.expectStdErrMatch("[sd.cpp] [warning] backend warning");
    log_run.expectStdErrMatch("[sd.cpp] [debug] backend detail");
    test_step.dependOn(&log_run.step);

    const smoke_exe = project.executable("pictor_smoke", &.{"tests/inference_smoke.cpp"});
    smoke_exe.root_module.addObjectFile(core.getEmittedBin());
    const install_smoke = b.addInstallArtifact(smoke_exe, .{});
    const smoke = b.addSystemCommand(&.{b.getInstallPath(.bin, "pictor_smoke")});
    smoke.step.dependOn(&install_smoke.step);
    smoke.step.dependOn(b.getInstallStep());
    if (b.args) |args| smoke.addArgs(args);
    b.step("smoke", "Check repeated inference with weights (optional model/output paths after --)").dependOn(&smoke.step);

    const reference = b.addExecutable(.{ .name = "sd-cli", .root_module = project.cpp(&.{
        "vendor/stable-diffusion.cpp/examples/common/common.cpp",
        "vendor/stable-diffusion.cpp/examples/common/log.cpp",
        "vendor/stable-diffusion.cpp/examples/common/media_io.cpp",
        "vendor/stable-diffusion.cpp/examples/cli/image_metadata.cpp",
        "vendor/stable-diffusion.cpp/examples/cli/main.cpp",
    }, true) });
    reference.each_lib_rpath = false;
    reference.root_module.addRPathSpecial(if (sdk != null) "@loader_path/../lib" else "$ORIGIN/../lib");
    reference.root_module.addIncludePath(b.path("vendor/stable-diffusion.cpp/examples"));
    reference.root_module.addIncludePath(b.path("vendor/stable-diffusion.cpp/ggml/include"));
    reference.root_module.addObjectFile(backend_file);
    reference.step.dependOn(&backend_build.step);
    const install_reference = b.addInstallArtifact(reference, .{});
    install_reference.step.dependOn(&install_backend.step);
    b.step("reference", "Build/install the pinned upstream CLI without the server").dependOn(&install_reference.step);

    const klein_download = b.addSystemCommand(&.{ "bash", b.pathFromRoot("scripts/download-klein-model.sh") });
    b.step("download-klein-model", "Download/verify Klein 4B Q4, Qwen3 4B Q4 and VAE (5.3 GB)").dependOn(&klein_download.step);

    const download = b.addSystemCommand(&.{ "bash", b.pathFromRoot("scripts/download-model.sh") });
    b.step("download-model", "Download and verify Anima P3 Turbo AIO Q4 (1.79 GB)").dependOn(&download.step);
}
