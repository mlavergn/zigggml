const std = @import("std");

const zon = @import("build.zig.zon");
const zigbuildx = @import("zigbuildx");
const ggml = @import("build/ggml.zig");

/// `config.zon`: the library's build-time settings.
const settings: ggml.Settings = @import("config.zon");

/// Declares the build graph for the zigggml library, CLI, tests, and docs.
///
/// Wires up `lib`, `cli`, `run`, `bench`, `test`, and `docs`, plus the
/// standard target and optimize options, the backend options (`-Dcuda*`,
/// `-Dvulkan*`, `-Dblas`; Metal is on for every Apple target), and
/// `-Dtest-filter`. `config.zon` sets `GGML_DEFAULT_N_THREADS`.
///
/// Parameters:
/// - `b`: the build graph the steps and options are registered on.
///
/// Return: nothing on success; propagates errors from option parsing, version
/// parsing, SDK resolution, and reading the vendored tree.
pub fn build(b: *std.Build) !void {
    // Pre-flight: ensure any local dependencies are cloned
    zigbuildx.Deps.cloneMissing(b, zon);

    const name = @tagName(zon.name);
    // The ported ARM kernels are the non-i8mm arms, so the default CPU stays
    // at the baseline the port was built and gated on; `-Dcpu` overrides it.
    const target = b.standardTargetOptions(.{ .default_target = .{ .cpu_model = .baseline } });
    const optimize = b.standardOptimizeOption(.{});
    _ = try std.SemanticVersion.parse(zon.version);

    const test_filter = b.option([]const u8, "test-filter", "Run unit tests that match filter") orelse "";
    const test_filters: []const []const u8 = if (test_filter.len == 0) &.{} else &.{test_filter};

    var xcode_handle: ?zigbuildx.Xcode = null;
    if (target.result.os.tag.isDarwin()) {
        xcode_handle = try zigbuildx.Xcode.initFor(b, target);
        try xcode_handle.?.resolve();
    }

    const opts: ggml.Options = .{
        .target = target,
        .optimize = optimize,
        .xcode = if (xcode_handle) |*x| x else null,
        .settings = settings,
        .cuda = ggml.cudaOptions(b),
        .vulkan = ggml.vulkanOptions(b),
        .blas = ggml.blasOption(b, target),
    };

    // -------------------------------------------------------------------------
    // Library

    const lib = try ggml.add(b, opts);

    const lib_install = b.addInstallArtifact(lib.lib, .{});
    const lib_step = b.step("lib", "Build static library");
    lib_step.dependOn(&lib_install.step);
    b.getInstallStep().dependOn(&lib_install.step);

    // -------------------------------------------------------------------------
    // CLI

    const cli_options = b.addOptions();
    cli_options.addOption(u32, "default_n_threads", settings.default_n_threads);
    cli_options.addOption(bool, "use_cuda", opts.cuda != null);
    cli_options.addOption([]const u8, "version", zon.version);

    const cli_module = b.createModule(.{
        .root_source_file = b.path("cli/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addCliImports(b, cli_module, cli_options, lib, opts);

    const cli = b.addExecutable(.{
        .name = name ++ "-cli",
        .root_module = cli_module,
        .use_llvm = true,
    });
    const cli_install = b.addInstallArtifact(cli, .{});
    const cli_step = b.step("cli", "Build the CLI app");
    cli_step.dependOn(&cli_install.step);
    b.getInstallStep().dependOn(&cli_install.step);

    const cli_run = b.addRunArtifact(cli);
    cli_run.step.dependOn(&cli_install.step);
    cli_run.addPassthruArgs();
    const run_step = b.step("run", "Run the CLI app");
    run_step.dependOn(&cli_run.step);

    // -------------------------------------------------------------------------
    // Tests

    // The library's own tests compile its module; a test executable needs
    // the backends' runtime libraries on its link.
    ggml.addLinkDeps(b, lib.module, opts);
    const lib_tests = b.addTest(.{ .root_module = lib.module, .filters = test_filters, .use_llvm = true });
    const lib_tests_run = b.addRunArtifact(lib_tests);
    lib_tests_run.has_side_effects = true;

    const cli_tests_module = b.createModule(.{
        .root_source_file = b.path("cli/module.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addCliImports(b, cli_tests_module, cli_options, lib, opts);
    const cli_tests = b.addTest(.{ .root_module = cli_tests_module, .filters = test_filters, .use_llvm = true });
    const cli_tests_run = b.addRunArtifact(cli_tests);
    cli_tests_run.has_side_effects = true;

    const tests_step = b.step("test", "Run unit tests");
    tests_step.dependOn(&lib_tests_run.step);
    tests_step.dependOn(&cli_tests_run.step);

    const cli_tests_step = b.step("test-cli", "Run the CLI's unit tests only");
    cli_tests_step.dependOn(&cli_tests_run.step);

    // A CPU smoke run of one small case per op.
    const smoke = b.addRunArtifact(cli);
    smoke.addArgs(&.{ "bench", "--smoke" });
    smoke.has_side_effects = true;
    tests_step.dependOn(&smoke.step);

    // -------------------------------------------------------------------------
    // Docs

    // Autodoc roots a module at `root.zig`, or at the file whose basename
    // matches the artifact name; with `src/module.zig` it can do neither, so
    // docs build from the sibling `root.zig` with the same imports.
    const docs_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    var imports = lib.module.import_table.iterator();
    while (imports.next()) |entry| docs_mod.addImport(entry.key_ptr.*, entry.value_ptr.*);
    const docs_lib = b.addLibrary(.{ .name = name, .root_module = docs_mod, .linkage = .static, .use_llvm = true });
    const docs = b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&docs.step);
}

/// Gives a CLI module what it needs: the `config` options, the translated
/// ggml headers as `ggml_c`, `libggml.a`, and the backends' runtime
/// libraries.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the CLI (or CLI test) module.
/// - `options`: the CLI's `config` options.
/// - `lib`: the ggml artifacts.
/// - `opts`: the options ggml was built with.
fn addCliImports(
    b: *std.Build,
    mod: *std.Build.Module,
    options: *std.Build.Step.Options,
    lib: ggml.Artifacts,
    opts: ggml.Options,
) void {
    mod.addOptions("config", options);
    mod.addImport("ggml_c", lib.c);
    mod.linkLibrary(lib.lib);
    ggml.addLinkDeps(b, mod, opts);
    // NVML, for the bench's GPU-utilization column: the CLI only, CUDA only.
    if (opts.cuda) |cuda| {
        const translate = b.addTranslateC(.{
            .root_source_file = b.addWriteFiles().add("nvml_import.h", "#include <nvml.h>\n"),
            .target = opts.target,
            .optimize = opts.optimize,
        });
        translate.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ cuda.toolkit, "include" }) });
        mod.addImport("nvml_c", translate.createModule());
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda.toolkit, "lib64", "stubs" }) });
        mod.linkSystemLibrary("nvidia-ml", .{});
    }
}
