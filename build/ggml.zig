//! Build graph for the ggml library: the Zig port under `src/`, the vendored
//! upstream inputs under `vendor/ggml/`, and the backend options.
//!
//! Ported from llama.zig's `build/llamacpp.zig` (`addGgml` and the CUDA,
//! Vulkan and Metal helpers) and `build.zig` (`cudaOptions`, `vulkanOptions`,
//! `blasOptions`). Unlike there, nothing reference is built: no upstream host
//! layer, no `*-port` toggles. The `config` options the sources read for those
//! toggles are still provided, fixed to the port.

const std = @import("std");
const zigbuildx = @import("zigbuildx");

/// Where the vendored upstream inputs live, relative to the build root.
pub const vendor = "vendor/ggml";

/// Warnings that fire on upstream's sources under Zig's clang but say nothing
/// about this project.
const quiet = [_][]const u8{
    "-Wno-nullability-completeness",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-variable",
};

const c_flags = [_][]const u8{"-std=c11"} ++ quiet;
const cxx_flags = [_][]const u8{"-std=c++17"} ++ quiet;
// Upstream's Metal sources use manual reference counting, so ARC stays off.
const objc_flags = [_][]const u8{"-std=c11"} ++ quiet;

/// The `-D` flags `GGML_ACCELERATE` adds (ggml/src/ggml-cpu/CMakeLists.txt).
const accelerate_defines = [_][]const u8{
    "-DGGML_USE_ACCELERATE",
    "-DACCELERATE_NEW_LAPACK",
    "-DACCELERATE_LAPACK_ILP64",
};

/// The Metal kernels embedded into the binary, one shader library per entry.
/// Mirrors the `GGML_METAL_LIBS` X-macro in `ggml-metal-device.m`.
const metal_kernels = [_][]const u8{
    "fa_aux",          "fa_f16",      "fa_f32",      "fa_q4_0",
    "fa_q4_1",         "fa_q5_0",     "fa_q5_1",     "fa_q8_0",
    "fa_tensor",       "fa_vec_f16",  "fa_vec_f32",  "fa_vec_q4_0",
    "fa_vec_q4_1",     "fa_vec_q5_0", "fa_vec_q5_1", "fa_vec_q8_0",
    "mul_mv",          "mul_mv_mma",  "mul_mm",      "quantize",
    "softmax",         "norm",        "unary",       "binbcast",
    "reduce",          "tri",         "ssm",         "wkv",
    "gated_delta_net", "solve_tri",   "rope",        "conv",
    "upscale",         "argsort",     "pool",        "misc",
};

/// The Obj-C half of the Metal backend, which stays Obj-C.
const metal_objc_sources = [_][]const u8{
    "src/ggml-metal/ggml-metal-device.m",
    "src/ggml-metal/ggml-metal-context.m",
};

/// `config.zon`: build-time settings for the library.
pub const Settings = struct {
    /// The value `GGML_DEFAULT_N_THREADS` is defined to in the generated
    /// `ggml.h`; upstream's is 4.
    default_n_threads: u32 = 4,
};

/// Where the CUDA toolkit is and which GPU to compile for.
pub const Cuda = struct {
    /// Toolkit root, e.g. `/usr/local/cuda`; `bin/nvcc` must exist under it.
    toolkit: []const u8,
    /// The `compute_`/`sm_` suffix to generate code for, e.g. `121a`.
    arch: []const u8,
    /// Upstream's `GGML_CUDA_FA_QUANTS`: `"all"` or a list such as
    /// `"q8_0-q8_0;q8_0-q4_0"`. See `cudaFaCombinations`.
    fa_quants: []const u8 = "all",
};

/// The Vulkan toolchain.
pub const Vulkan = struct {
    /// The `glslc` every shader is compiled with.
    glslc: []const u8,
    /// Directories holding `vulkan/` and `spirv/`, searched after libc's.
    include_dirs: []const []const u8,
    /// Extra directories to find `libvulkan` in.
    lib_dirs: []const []const u8,
    /// `-D` flags for the shader extensions `glslc` accepts.
    features: []const []const u8,
};

/// Options that shape the ggml build.
pub const Options = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
    /// The resolved Apple SDK; null off Apple targets.
    xcode: ?*const zigbuildx.Xcode = null,
    /// `config.zon`'s settings.
    settings: Settings = .{},
    /// Builds the CUDA backend; null leaves it out.
    cuda: ?Cuda = null,
    /// Builds the Vulkan backend; null leaves it out.
    vulkan: ?Vulkan = null,
    /// Builds the BLAS backend over Accelerate (Apple only).
    blas: bool = false,
};

/// What `add` produces.
pub const Artifacts = struct {
    /// The public `zigggml` module, rooted at `src/module.zig`; also the
    /// library's root module.
    module: *std.Build.Module,
    /// `libggml.a`, exporting the C ABI.
    lib: *std.Build.Step.Compile,
    /// The translated ggml headers, published as the `ggml_c` module.
    c: *std.Build.Module,
    /// The generated include directory (`ggml.h` with the configured
    /// `GGML_DEFAULT_N_THREADS`, plus the other public headers).
    include: std.Build.LazyPath,
};

/// Whether the target gets the Apple configuration: Metal and Accelerate.
fn isApple(opts: Options) bool {
    return opts.target.result.os.tag.isDarwin();
}

/// Whether `src/varargs.c` must define ggml's variadic entry points because
/// Zig cannot for this target. Must agree with `impl.zig_varargs`.
fn needsVarargsShim(opts: Options) bool {
    const t = opts.target.result;
    return switch (t.cpu.arch) {
        .aarch64, .aarch64_be => !(t.os.tag.isDarwin() or t.os.tag == .windows),
        .x86_64 => t.os.tag == .windows,
        else => false,
    };
}

/// Joins flag lists into one slice owned by the build graph.
fn flags(b: *std.Build, parts: []const []const []const u8) []const []const u8 {
    return std.mem.concat(b.allocator, []const u8, parts) catch @panic("OOM");
}

/// Resolves the `-Dcuda` options. Off unless `-Dcuda`. The architecture
/// defaults to what `nvidia-smi` reports for the first GPU, with a Blackwell
/// `12X` rewritten to `12Xa` as upstream's CMake does.
///
/// Parameters:
/// - `b`: the build graph the options are registered on.
///
/// Return: the toolkit and architecture, or null when CUDA is off.
pub fn cudaOptions(b: *std.Build) ?Cuda {
    const enabled = b.option(bool, "cuda", "Build the CUDA backend (Linux; needs nvcc)") orelse false;
    const toolkit = b.option([]const u8, "cuda-toolkit", "CUDA toolkit root (default /usr/local/cuda)") orelse "/usr/local/cuda";
    const arch_opt = b.option([]const u8, "cuda-arch", "GPU architecture to compile for, e.g. 121a (default: from nvidia-smi)");
    const fa_quants = b.option([]const u8, "cuda-fa-quants", "FlashAttention K-V type combinations to compile, \"all\" or e.g. \"q8_0-q8_0;q8_0-q4_0\" (default: all)") orelse "all";
    if (!enabled) return null;

    const arch = arch_opt orelse detect: {
        var code: u8 = 0;
        const out = b.runAllowFail(&.{ "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader" }, &code, .ignore) catch
            std.debug.panic("-Dcuda: nvidia-smi found no GPU; pass -Dcuda-arch=<e.g. 121a>", .{});
        const line = std.mem.trim(u8, std.mem.sliceTo(out, '\n'), " \r\t");
        var digits: std.ArrayList(u8) = .empty;
        for (line) |ch| if (ch != '.') digits.append(b.allocator, ch) catch @panic("OOM");
        // 12X is forward-compatible, 12Xa is not, and the FP4 tensor-core
        // instructions need 12Xa.
        if (digits.items.len == 3 and std.mem.startsWith(u8, digits.items, "12")) digits.append(b.allocator, 'a') catch @panic("OOM");
        break :detect digits.items;
    };
    return .{ .toolkit = toolkit, .arch = arch, .fa_quants = fa_quants };
}

/// Resolves `-Dblas`: ggml's BLAS backend over Apple's Accelerate. Apple only.
///
/// Parameters:
/// - `b`: the build graph, which owns the option.
/// - `target`: the target being built, which must be Darwin when enabled.
///
/// Return: whether BLAS is built.
pub fn blasOption(b: *std.Build, target: std.Build.ResolvedTarget) bool {
    const enabled = b.option(bool, "blas", "Build ggml's BLAS backend over Apple's Accelerate (Apple targets only)") orelse false;
    if (enabled and !target.result.os.tag.isDarwin()) {
        std.debug.panic("-Dblas: the BLAS backend is built over Apple's Accelerate only", .{});
    }
    return enabled;
}

/// Resolves the `-Dvulkan` options and asks `glslc` which shader extensions
/// it accepts, as `ggml-vulkan/CMakeLists.txt` does.
///
/// The toolchain defaults to the pinned one in `.tools/vulkan/`
/// (`make vk-toolchain`) where it exists, else `glslc` on PATH and
/// `/usr/include`. A probe counts as supported unless `glslc` says
/// `extension not supported: <name>`, so `glslc` is shown to run first.
///
/// Parameters:
/// - `b`: the build graph the options are registered on.
///
/// Return: the toolchain, or null when Vulkan is off.
pub fn vulkanOptions(b: *std.Build) ?Vulkan {
    const enabled = b.option(bool, "vulkan", "Build ggml's Vulkan backend (needs glslc, the Vulkan headers and libvulkan; `make vk-toolchain`)") orelse false;
    const glslc_opt = b.option([]const u8, "glslc", "The glslc to compile the Vulkan shaders with (default: .tools/vulkan/bin/glslc, else PATH)");
    const include_opt = b.option([]const []const u8, "vulkan-include", "Directories holding vulkan/ and spirv/ (default: .tools/vulkan/include, else /usr/include)");
    const lib_opt = b.option([]const []const u8, "vulkan-lib", "Extra directories to find libvulkan in (default: .tools/vulkan/lib)");
    if (!enabled) return null;

    const tools = b.root.joinString(b.allocator, ".tools/vulkan") catch @panic("OOM");
    const have_tools = if (std.Io.Dir.accessAbsolute(b.graph.io, b.pathJoin(&.{ tools, "bin", "glslc" }), .{})) true else |_| false;
    const glslc = glslc_opt orelse if (have_tools)
        b.pathJoin(&.{ tools, "bin", "glslc" })
    else
        b.findProgram(.{ .names = &.{"glslc"} }) orelse std.debug.panic("-Dvulkan needs glslc: run `make vk-toolchain` or pass -Dglslc=<path>", .{});
    const include_dirs: []const []const u8 = include_opt orelse if (have_tools)
        b.allocator.dupe([]const u8, &.{b.pathJoin(&.{ tools, "include" })}) catch @panic("OOM")
    else
        &[_][]const u8{"/usr/include"};
    const lib_dirs: []const []const u8 = lib_opt orelse if (have_tools)
        b.allocator.dupe([]const u8, &.{b.pathJoin(&.{ tools, "lib" })}) catch @panic("OOM")
    else
        &[_][]const u8{};

    var code: u8 = 0;
    _ = b.runAllowFail(&.{ glslc, "--version" }, &code, .ignore) catch
        std.debug.panic("-Dvulkan: {s} does not run", .{glslc});

    var features: std.ArrayList([]const u8) = .empty;
    for (vulkan_glslc_features) |probe| {
        const rel = b.fmt(vendor ++ "/src/ggml-vulkan/vulkan-shaders/feature-tests/{s}", .{probe.file});
        // glslc reports a refusal on stderr and exits non-zero, so the shell
        // folds both into a successful stdout.
        const out = b.run(&.{
            "sh",                                                                                       "-c",
            "\"$0\" -o - -fshader-stage=compute --target-env=vulkan1.3 \"$1\" 2>&1 >/dev/null; exit 0", glslc,
            b.root.joinString(b.allocator, rel) catch @panic("OOM"),
        });
        if (std.mem.indexOf(u8, out, b.fmt("extension not supported: {s}", .{probe.extension})) == null) {
            features.append(b.allocator, probe.define) catch @panic("OOM");
        }
    }
    return .{ .glslc = glslc, .include_dirs = include_dirs, .lib_dirs = lib_dirs, .features = features.items };
}

/// The shader extension probes, the `GGML_VULKAN_*_GLSLC_SUPPORT` macros
/// they set, and the `config` options that carry them to the host layer.
const vulkan_glslc_features = [_]struct {
    extension: []const u8,
    file: []const u8,
    define: []const u8,
    option: []const u8,
}{
    .{ .extension = "GL_KHR_cooperative_matrix", .file = "coopmat.comp", .define = "-DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT", .option = "vk_coopmat_glslc" },
    .{ .extension = "GL_NV_cooperative_matrix2", .file = "coopmat2.comp", .define = "-DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT", .option = "vk_coopmat2_glslc" },
    .{ .extension = "GL_NV_cooperative_matrix_decode_vector", .file = "coopmat2_decode_vector.comp", .define = "-DGGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT", .option = "vk_coopmat2_decode_vector_glslc" },
    .{ .extension = "GL_EXT_integer_dot_product", .file = "integer_dot.comp", .define = "-DGGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT", .option = "vk_integer_dot_glslc" },
    .{ .extension = "GL_EXT_bfloat16", .file = "bfloat16.comp", .define = "-DGGML_VULKAN_BFLOAT16_GLSLC_SUPPORT", .option = "vk_bfloat16_glslc" },
    .{ .extension = "GL_EXT_float_e2m1", .file = "float_e2m1.comp", .define = "-DGGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT", .option = "vk_float_e2m1_glslc" },
    .{ .extension = "GL_EXT_float_e4m3", .file = "float_e4m3.comp", .define = "-DGGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT", .option = "vk_float_e4m3_glslc" },
};

fn hasDefine(defines: []const []const u8, define: []const u8) bool {
    for (defines) |d| if (std.mem.eql(u8, d, define)) return true;
    return false;
}

/// The `#define` line in upstream's `ggml.h` that `config.zon` rewrites.
const n_threads_define = "#define GGML_DEFAULT_N_THREADS  4";

/// Generates the public include directory: every vendored header, with
/// `ggml.h`'s `GGML_DEFAULT_N_THREADS` set from `config.zon`.
///
/// The whole directory is copied, not just `ggml.h`, because the other public
/// headers include `"ggml.h"` quoted, which searches their own directory
/// first: an overriding `ggml.h` elsewhere would never be seen by them.
///
/// Parameters:
/// - `b`: the build graph.
/// - `settings`: `config.zon`'s settings.
///
/// Return: the generated directory; fails the build if the `#define` line
/// is not found.
fn generateInclude(b: *std.Build, settings: Settings) !std.Build.LazyPath {
    if (settings.default_n_threads < 1) std.debug.panic("config.zon: .default_n_threads must be at least 1", .{});

    const io = b.graph.io;
    const include = vendor ++ "/include";
    b.dependOnFileContents(b.path(include ++ "/ggml.h"));
    var dir = try b.root.openDir(io, include, .{});
    defer dir.close(io);
    const original = try dir.readFileAlloc(io, "ggml.h", b.allocator, .limited(1 << 20));

    const at = std.mem.find(u8, original, n_threads_define) orelse
        std.debug.panic("{s}/ggml.h: no \"{s}\" line to rewrite from config.zon", .{ include, n_threads_define });
    const rewritten = b.fmt("{s}#define GGML_DEFAULT_N_THREADS  {d}{s}", .{
        original[0..at],
        settings.default_n_threads,
        original[at + n_threads_define.len ..],
    });

    const wf = b.addWriteFiles();
    _ = wf.addCopyDirectory(b.path(include), "", .{ .exclude_extensions = &.{"ggml.h"} });
    _ = wf.add("ggml.h", rewritten);
    return wf.getDirectory();
}

/// Translates `src/c_import.h` against the generated headers: the `ggml_c`
/// module every module rooted in `src/` imports.
fn ggmlCModule(b: *std.Build, opts: Options, include: std.Build.LazyPath) *std.Build.Module {
    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/c_import.h"),
        .target = opts.target,
        .optimize = opts.optimize,
    });
    translate.addIncludePath(include);
    translate.addIncludePath(b.path(vendor ++ "/src"));
    return translate.createModule();
}

/// Declares the `zigggml` module and `libggml.a`.
///
/// Parameters:
/// - `b`: the build graph.
/// - `opts`: target, optimize mode, SDK, settings and backends.
///
/// Return: the module, library, translated headers and include directory;
/// propagates allocation and directory-read failures.
pub fn add(b: *std.Build, opts: Options) !Artifacts {
    const include = try generateInclude(b, opts.settings);
    const c = ggmlCModule(b, opts, include);

    const mod = b.addModule("zigggml", .{
        .root_source_file = b.path("src/module.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .link_libc = true,
        // The vendored C is not UBSan-clean; see llama.zig's `addGgml`.
        .sanitize_c = .off,
    });

    const options = b.addOptions();
    options.addOption(bool, "probe_ported", false);
    options.addOption(bool, "probe_cpu", false);
    options.addOption(bool, "use_metal", isApple(opts));
    options.addOption(bool, "use_cuda", opts.cuda != null);
    // The port is the only host layer here, so each `*_port` follows its
    // backend.
    options.addOption(bool, "cuda_port", opts.cuda != null);
    options.addOption([]const u8, "cuda_arch", if (opts.cuda) |cu| cu.arch else "");
    const fa: CudaFa = if (opts.cuda) |cu| cudaFaCombinations(b, cu) else .{ .define = "", .combinations = &.{} };
    options.addOption([]const u8, "cuda_fa_quants", fa.define);
    options.addOption([]const []const u8, "cuda_fa_combinations", fa.combinations);
    options.addOption(bool, "use_vulkan", opts.vulkan != null);
    options.addOption(bool, "vulkan_port", opts.vulkan != null);
    options.addOption(bool, "use_blas", opts.blas);
    options.addOption(bool, "blas_port", opts.blas);
    for (vulkan_glslc_features) |f| {
        options.addOption(bool, f.option, if (opts.vulkan) |vk| hasDefine(vk.features, f.define) else false);
    }
    options.addOption(u32, "default_n_threads", opts.settings.default_n_threads);
    mod.addOptions("config", options);
    mod.addImport("ggml_c", c);

    if (needsVarargsShim(opts)) mod.addCSourceFile(.{ .file = b.path("src/varargs.c"), .flags = &c_flags });

    mod.addIncludePath(include);
    mod.addIncludePath(b.path(vendor ++ "/src"));

    if (isApple(opts)) {
        const defines = [_][]const u8{
            "-DGGML_USE_CPU",
            "-DGGML_SCHED_MAX_COPIES=4",
            "-DGGML_USE_LLAMAFILE",
            "-DGGML_USE_CPU_REPACK",
            "-DGGML_USE_METAL",
            "-DGGML_METAL_EMBED_LIBRARY",
        } ++ accelerate_defines;
        mod.addIncludePath(b.path(vendor ++ "/src/ggml-metal"));
        mod.addCSourceFiles(.{
            .root = b.path(vendor),
            .files = &metal_objc_sources,
            .flags = flags(b, &.{ &objc_flags, &defines }),
        });
        for (metal_kernels) |kind| mod.addAssemblyFile(embedMetalKernel(b, kind));

        if (opts.xcode) |xcode| xcode.addSdkPaths(mod);
        mod.linkFramework("Foundation", .{});
        mod.linkFramework("Metal", .{});
        mod.linkFramework("MetalKit", .{});
        mod.linkFramework("Accelerate", .{});
    }

    if (opts.cuda) |cuda| try addCuda(b, mod, cuda);
    if (opts.vulkan) |vk| try addVulkan(b, mod, vk);

    // LLVM's backend, always: the port calls LLVM intrinsics by name, which
    // Zig's own backends emit as undefined symbols.
    const lib = b.addLibrary(.{ .name = "ggml", .root_module = mod, .linkage = .static, .use_llvm = true });
    if (opts.xcode) |xcode| {
        if (!opts.target.query.isNative()) try xcode.configure(lib);
    }
    lib.installHeadersDirectory(include, "", .{});

    b.modules.put(b.graph.arena, "ggml_c", c) catch @panic("OOM");
    return .{ .module = mod, .lib = lib, .c = c, .include = include };
}

/// Adds what an executable linking `libggml.a` needs beyond the archive:
/// the CUDA runtime and libstdc++, the Vulkan loader, and the Apple SDK's
/// search paths. Search paths do not travel with a static library.
///
/// Parameters:
/// - `b`: the build graph.
/// - `mod`: the executable's root module.
/// - `opts`: the options the library was built with.
pub fn addLinkDeps(b: *std.Build, mod: *std.Build.Module, opts: Options) void {
    if (opts.cuda) |cuda| {
        // The CUB shim's host object needs a C++ runtime and unwinder; Zig's
        // libc++ supplies both, as for llama.zig's upstream tools
        // (`addCudaRuntime`). cuBLAS loads libstdc++ itself at run time.
        mod.link_libcpp = true;
        addLibgcc(b, mod);
        addCudaLibraries(b, mod, cuda);
    }
    if (opts.vulkan) |vk| {
        for (vk.lib_dirs) |dir| mod.addLibraryPath(.{ .cwd_relative = dir });
        mod.linkSystemLibrary("vulkan", .{});
    }
    if (opts.xcode) |xcode| xcode.addSdkPaths(mod);
}

/// Flattens one Metal kernel and returns the assembly stub that embeds it,
/// running `build/metal_embed.zig` as a build-time tool.
///
/// Parameters:
/// - `b`: the build graph.
/// - `kind`: kernel name, matching a `kernels/<kind>.metal` source.
///
/// Return: the generated `.s` file.
fn embedMetalKernel(b: *std.Build, kind: []const u8) std.Build.LazyPath {
    const tool = b.addExecutable(.{
        .name = "metal_embed",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/metal_embed.zig"),
            .target = b.graph.host,
            .optimize = .debug,
        }),
    });

    const run = b.addRunArtifact(tool);
    run.addArg(kind);
    _ = run.addOutputFileArg2(b.fmt("ggml-metal-embed-{s}.metal", .{kind}), .{});
    const out_asm = run.addOutputFileArg2(b.fmt("ggml-metal-embed-{s}.s", .{kind}), .{});

    const metal = vendor ++ "/src/ggml-metal";
    run.addFileArg(b.path(vendor ++ "/src/ggml-common.h"));
    run.addFileArg(b.path(metal ++ "/ggml-metal-impl.h"));
    run.addFileArg(b.path(metal ++ "/kernels/common.h"));
    run.addFileArg(b.path(metal ++ "/kernels/dequantize.h"));
    run.addFileArg(b.path(metal ++ "/kernels/quantize.h"));
    run.addFileArg(b.path(metal ++ "/kernels/fa_common.metal"));
    run.addFileArg(b.path(metal ++ "/kernels/fa_vec_common.metal"));
    run.addFileArg(b.path(b.fmt(metal ++ "/kernels/{s}.metal", .{kind})));

    return out_asm;
}

// -----------------------------------------------------------------------------
// CUDA

/// The `-D` flags upstream's CMake gives every `ggml-cuda` translation unit
/// for this configuration: graphs on, NCCL off.
const cuda_defines = [_][]const u8{
    "-DGGML_CUDA_USE_GRAPHS",
    "-DGGML_SCHED_MAX_COPIES=4",
    "-D_GNU_SOURCE",
    "-D_XOPEN_SOURCE=600",
    "-DNDEBUG",
};

/// The K and V types a FlashAttention vector kernel can be instantiated for,
/// in the order `ggml_cuda_fattn_vec_instances` iterates them.
const cuda_fa_types = [_][]const u8{ "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "bf16", "f16" };

/// What `cudaFaCombinations` derives from `Cuda.fa_quants`.
const CudaFa = struct {
    /// The value `GGML_CUDA_FA_QUANTS` is defined to.
    define: []const u8,
    /// Every compiled K-V combination, spelled `"<k>-<v>"`.
    combinations: []const []const u8,
};

/// Mirrors `ggml_cuda_fattn_vec_instances`: the K-V type combinations whose
/// FlashAttention vector kernels `GGML_CUDA_FA_QUANTS` selects. `"all"` is
/// every pair, V outermost; otherwise `f16-f16` and each listed combination.
/// A malformed list fails the build.
fn cudaFaCombinations(b: *std.Build, cuda: Cuda) CudaFa {
    const lowered = b.allocator.alloc(u8, cuda.fa_quants.len) catch @panic("OOM");
    _ = std.ascii.lowerString(lowered, cuda.fa_quants);
    const fa_quants = std.mem.trim(u8, lowered, " \t\r\n");
    if (fa_quants.len == 0) std.debug.panic("GGML_CUDA_FA_QUANTS must not be empty", .{});

    var combinations: std.ArrayList([]const u8) = .empty;
    if (std.mem.eql(u8, fa_quants, "all")) {
        for (cuda_fa_types) |type_v| {
            for (cuda_fa_types) |type_k| {
                combinations.append(b.allocator, b.fmt("{s}-{s}", .{ type_k, type_v })) catch @panic("OOM");
            }
        }
    } else {
        combinations.append(b.allocator, "f16-f16") catch @panic("OOM");
        var it = std.mem.tokenizeAny(u8, fa_quants, ";,");
        while (it.next()) |raw| {
            const combination = std.mem.trim(u8, raw, " \t\r\n");
            const dash = std.mem.findScalar(u8, combination, '-') orelse
                std.debug.panic("GGML_CUDA_FA_QUANTS: \"{s}\" is not \"all\" or a <type_K>-<type_V> combination", .{combination});
            for ([_][]const u8{ combination[0..dash], combination[dash + 1 ..] }) |ty| {
                const known = for (cuda_fa_types) |t| {
                    if (std.mem.eql(u8, t, ty)) break true;
                } else false;
                if (!known) std.debug.panic("GGML_CUDA_FA_QUANTS: unknown type \"{s}\" in \"{s}\"", .{ ty, combination });
            }
            const seen = for (combinations.items) |have| {
                if (std.mem.eql(u8, have, combination)) break true;
            } else false;
            if (!seen) combinations.append(b.allocator, combination) catch @panic("OOM");
        }
    }

    const define = b.allocator.dupe(u8, fa_quants) catch @panic("OOM");
    std.mem.replaceScalar(u8, define, ';', ',');
    return .{ .define = define, .combinations = combinations.items };
}

/// The FlashAttention `-D` flags: `GGML_CUDA_FA_QUANTS` as a string, and
/// `GGML_CUDA_FA_<K>_<V>` as 1 or 0 for every type pair.
fn cudaFaDefines(b: *std.Build, cuda: Cuda) []const []const u8 {
    const fa = cudaFaCombinations(b, cuda);
    var defines: std.ArrayList([]const u8) = .empty;
    defines.append(b.allocator, b.fmt("-DGGML_CUDA_FA_QUANTS=\"{s}\"", .{fa.define})) catch @panic("OOM");
    for (cuda_fa_types) |type_v| {
        for (cuda_fa_types) |type_k| {
            const name = b.fmt("{s}-{s}", .{ type_k, type_v });
            const compiled = for (fa.combinations) |have| {
                if (std.mem.eql(u8, have, name)) break true;
            } else false;
            const def = b.fmt("-DGGML_CUDA_FA_{s}_{s}={d}", .{ type_k, type_v, @intFromBool(compiled) });
            _ = std.ascii.upperString(def["-D".len .. def.len - 2], def["-D".len .. def.len - 2]);
            defines.append(b.allocator, def) catch @panic("OOM");
        }
    }
    return defines.items;
}

/// Adds the Zig CUDA host layer's inputs to the ggml module.
///
/// Every upstream translation unit compiles to a fatbin. Its entry symbols
/// are listed with `cuobjdump` and demangled with `cu++filt`, and
/// `build/cuda_kernels.zig` turns the listings into the `cuda_kernels`
/// module, which embeds the fatbins. CUB's device-wide sort and scan come
/// from `src/cuda/cub_shim.cu`, a host object with its device code
/// registered through cudart. No upstream host code is linked.
fn addCuda(b: *std.Build, mod: *std.Build.Module, cuda: Cuda) !void {
    const files = try cudaSources(b, cuda);
    const bin = b.pathJoin(&.{ cuda.toolkit, "bin" });
    const tool = b.addExecutable(.{
        .name = "cuda_kernels",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/cuda_kernels.zig"),
            .target = b.graph.host,
            .optimize = .debug,
        }),
    });
    const gen = b.addRunArtifact(tool);
    const table = gen.addOutputFileArg("cuda_kernels.zig");

    var fatbins: std.ArrayList(std.Build.LazyPath) = .empty;
    for (files) |file| {
        const fatbin = nvccCompile(b, cuda, file, .fatbin);
        try fatbins.append(b.allocator, fatbin);

        // One entry symbol per line, then the same lines demangled beside
        // them. Only entry points (`STO_ENTRY`) are kernels.
        const list = b.addSystemCommand(&.{
            "/bin/sh", "-c",
            \\set -e
            \\"$1/cuobjdump" -symbols "$2" | awk '/STO_ENTRY/ { print $NF }' | sort -u > "$3.mangled"
            \\"$1/cu++filt" < "$3.mangled" | paste "$3.mangled" - > "$3"
            ,
            "sh",      bin,
        });
        list.addFileArg(fatbin);
        const listing = list.addOutputFileArg(b.fmt("{s}.symbols", .{std.fs.path.stem(file)}));

        gen.addArg(std.fs.path.stem(file));
        gen.addFileArg(listing);
    }

    const kernels = b.createModule(.{ .root_source_file = table });
    for (fatbins.items, 0..) |fatbin, i| {
        kernels.addAnonymousImport(b.fmt("cuda_fatbin_{d}", .{i}), .{ .root_source_file = fatbin });
    }
    mod.addImport("cuda_kernels", kernels);

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/cuda/c_import.h"),
        .target = mod.resolved_target.?,
        .optimize = mod.optimize.?,
    });
    translate.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ cuda.toolkit, "include" }) });
    mod.addImport("cuda_c", translate.createModule());

    mod.addObjectFile(nvccCompile(b, cuda, "src/cuda/cub_shim.cu", .object));
}

/// Upstream's `ggml-cuda` source set, sorted: every `*.cu`, plus the
/// `fattn-tile`, `fattn-mma`, `mmq` and `mmf` template instances, and the
/// `fattn-vec` instances of the selected combinations.
fn cudaSources(b: *std.Build, cuda: Cuda) ![]const []const u8 {
    const dir_path = vendor ++ "/src/ggml-cuda";
    var files: std.ArrayList([]const u8) = .empty;
    const instance_prefixes = [_][]const u8{ "fattn-tile", "fattn-mma", "mmq", "mmf" };
    const fa_combinations = cudaFaCombinations(b, cuda).combinations;
    var fa_found: usize = 0;
    for ([_][]const u8{ "", "/template-instances" }) |sub| {
        const path = b.fmt("{s}{s}", .{ dir_path, sub });
        b.dependOnDirectoryMetadata(b.path(path));
        var dir = try b.root.openDir(b.graph.io, path, .{ .iterate = true });
        defer dir.close(b.graph.io);
        var it = dir.iterate();
        while (try it.next(b.graph.io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".cu")) continue;
            if (sub.len != 0) {
                const vec_prefix = "fattn-vec-instance-";
                const wanted = if (std.mem.startsWith(u8, entry.name, vec_prefix)) vec: {
                    const combination = entry.name[vec_prefix.len .. entry.name.len - ".cu".len];
                    for (fa_combinations) |have| {
                        if (std.mem.eql(u8, have, combination)) {
                            fa_found += 1;
                            break :vec true;
                        }
                    }
                    break :vec false;
                } else for (instance_prefixes) |prefix| {
                    if (std.mem.startsWith(u8, entry.name, prefix)) break true;
                } else false;
                if (!wanted) continue;
            }
            try files.append(b.allocator, b.fmt("{s}/{s}", .{ path, entry.name }));
        }
    }
    if (fa_found != fa_combinations.len) {
        std.debug.panic("GGML_CUDA_FA_QUANTS: {d} of {d} FlashAttention template instances exist", .{ fa_found, fa_combinations.len });
    }
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lessThan(_: void, l: []const u8, r: []const u8) bool {
            return std.mem.lessThan(u8, l, r);
        }
    }.lessThan);
    return files.items;
}

/// What one `nvcc` run produces.
const NvccOutput = enum {
    /// `-c`: a host object with the device code registered through cudart.
    object,
    /// `-fatbin`: device code alone, for the Zig host layer to load.
    fatbin,
};

/// Runs `nvcc` on one translation unit with upstream's flags.
fn nvccCompile(b: *std.Build, cuda: Cuda, file: []const u8, kind: NvccOutput) std.Build.LazyPath {
    const nvcc = b.pathJoin(&.{ cuda.toolkit, "bin", "nvcc" });
    const gencode = b.fmt("--generate-code=arch=compute_{s},code=[compute_{s},sm_{s}]", .{ cuda.arch, cuda.arch, cuda.arch });
    const mode = switch (kind) {
        .object => "-c",
        .fatbin => "-fatbin",
    };
    const run = b.addSystemCommand(&.{ nvcc, mode, "-O3", "-std=c++17", gencode, "-use_fast_math", "-extended-lambda", "-compress-mode=size", "-Xcompiler=-Wno-pedantic,-mno-outline-atomics" });
    run.addArgs(&cuda_defines);
    run.addArgs(cudaFaDefines(b, cuda));
    run.addDirectoryArg2(b.path(vendor ++ "/include"), .{ .prefix = "-I" });
    run.addDirectoryArg2(b.path(vendor ++ "/src"), .{ .prefix = "-I" });
    run.addDirectoryArg2(b.path(vendor ++ "/src/ggml-cuda"), .{ .prefix = "-I" });
    // `-MF` must be its own argument: nvcc rejects the joined form.
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2("out.d", .{});
    run.addFileArg(b.path(file));
    run.addArg("-o");
    const ext = switch (kind) {
        .object => "o",
        .fatbin => "fatbin",
    };
    return run.addOutputFileArg2(b.fmt("{s}.{s}", .{ std.fs.path.stem(file), ext }), .{});
}

/// Linux: gcc's static `libgcc.a` on the link, ahead of Zig's
/// `compiler_rt`, as llama.zig's `addLibgcc` does: `nvcc`'s host objects
/// were compiled by g++ and expect gcc's runtime (and with it, `compiler_rt`'s
/// weak `expf`/`logf` stop shadowing glibc's). Native gnu builds only.
fn addLibgcc(b: *std.Build, mod: *std.Build.Module) void {
    const t = mod.resolved_target orelse return;
    const host = b.graph.host.result;
    if (host.os.tag != .linux or t.result.os.tag != .linux) return;
    if (t.result.cpu.arch != host.cpu.arch or !t.result.abi.isGnu()) return;
    var code: u8 = 0;
    const out = b.runAllowFail(&.{ "gcc", "-print-libgcc-file-name" }, &code, .ignore) catch return;
    const path = std.mem.trim(u8, out, " \r\n\t");
    if (!std.fs.path.isAbsolute(path)) return;
    mod.addObjectFile(.{ .cwd_relative = path });
}

/// Adds the CUDA runtime, cuBLAS and the driver to a module's link, with the
/// toolkit's `lib64` as an rpath.
fn addCudaLibraries(b: *std.Build, mod: *std.Build.Module, cuda: Cuda) void {
    const lib64 = b.pathJoin(&.{ cuda.toolkit, "lib64" });
    mod.addLibraryPath(.{ .cwd_relative = lib64 });
    mod.addRPath(.{ .cwd_relative = lib64 });
    for ([_][]const u8{ "cudart", "cublas", "cublasLt", "cuda" }) |name| {
        mod.linkSystemLibrary(name, .{});
    }
}

// -----------------------------------------------------------------------------
// Vulkan

/// Runs upstream's `vulkan-shaders-gen` over every shader file and feeds its
/// output to `build/vulkan_shaders.zig`, whose table the Zig host layer
/// embeds; translates the Vulkan C headers as `vulkan_c`.
///
/// The generator is built for the host with the extension macros, run once
/// with no source for the header and once per `.comp` with `--glslc`. Every
/// `.comp` and `.glsl` is an input of every run, because `glslc` resolves the
/// includes and Zig cannot see them.
fn addVulkan(b: *std.Build, mod: *std.Build.Module, vk: Vulkan) !void {
    const shaders_dir = vendor ++ "/src/ggml-vulkan/vulkan-shaders";

    const gen_mod = b.createModule(.{
        .target = b.graph.host,
        .optimize = .fast,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
    });
    gen_mod.addCSourceFile(.{ .file = b.path(shaders_dir ++ "/vulkan-shaders-gen.cpp"), .flags = flags(b, &.{ &cxx_flags, vk.features }) });
    const gen = b.addExecutable(.{ .name = "vulkan-shaders-gen", .root_module = gen_mod });

    var comp: std.ArrayList([]const u8) = .empty;
    var glsl: std.ArrayList([]const u8) = .empty;
    {
        b.dependOnDirectoryMetadata(b.path(shaders_dir));
        var dir = try b.root.openDir(b.graph.io, shaders_dir, .{ .iterate = true });
        defer dir.close(b.graph.io);
        var it = dir.iterate();
        while (try it.next(b.graph.io)) |entry| {
            if (entry.kind != .file) continue;
            const path = b.fmt("{s}/{s}", .{ shaders_dir, entry.name });
            if (std.mem.endsWith(u8, entry.name, ".comp")) try comp.append(b.allocator, path);
            if (std.mem.endsWith(u8, entry.name, ".glsl")) try glsl.append(b.allocator, path);
        }
    }
    const lessThan = struct {
        fn f(_: void, l: []const u8, r: []const u8) bool {
            return std.mem.lessThan(u8, l, r);
        }
    }.f;
    std.mem.sort([]const u8, comp.items, {}, lessThan);
    std.mem.sort([]const u8, glsl.items, {}, lessThan);

    // One run per shader file, each writing that file's SPIR-V as a `.cpp`.
    const table_tool = b.addExecutable(.{
        .name = "vulkan_shaders",
        .root_module = b.createModule(.{
            .root_source_file = b.path("build/vulkan_shaders.zig"),
            .target = b.graph.host,
            // The largest generated file is 129 MB of hex.
            .optimize = .fast,
        }),
    });
    const table_run = b.addRunArtifact(table_tool);
    const table = table_run.addOutputFileArg("vulkan_shaders.zig");
    const blob = table_run.addOutputFileArg("vulkan_shaders.bin");
    for (comp.items) |file| {
        const run = b.addRunArtifact(gen);
        run.addArgs(&.{ "--glslc", vk.glslc });
        run.addArg("--source");
        run.addFileArg(b.path(file));
        run.addArg("--output-dir");
        _ = run.addOutputDirectoryArg("spv");
        // Only its basename is used, for the generated `#include`.
        run.addArgs(&.{ "--target-hpp", "ggml-vulkan-shaders.hpp" });
        run.addArg("--target-cpp");
        table_run.addFileArg(run.addOutputFileArg(b.fmt("{s}.cpp", .{std.fs.path.basename(file)})));
        for (comp.items) |f| run.addFileInput(b.path(f));
        for (glsl.items) |f| run.addFileInput(b.path(f));
    }

    const shaders_mod = b.createModule(.{ .root_source_file = table });
    shaders_mod.addAnonymousImport("vulkan_shaders_blob", .{ .root_source_file = blob });
    mod.addImport("vulkan_shaders", shaders_mod);

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/vulkan/c_import.h"),
        .target = mod.resolved_target.?,
        .optimize = mod.optimize.?,
    });
    for (vk.include_dirs) |dir| translate.addAfterIncludePath(.{ .cwd_relative = dir });
    mod.addImport("vulkan_c", translate.createModule());
}
