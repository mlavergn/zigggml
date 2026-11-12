const std = @import("std");
const mod = @import("module.zig");

/// One bench case, type-erased: the port of `test-backend-ops`' `test_case`
/// interface.
///
/// A case is a plain struct (see `cases/`). Its fields are the C++
/// constructor's parameters, first, in order, with the same defaults, so
/// `CaseList.add` can build one from positional arguments the way the C++
/// case lists call constructors. It declares:
///
/// - `pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor` (or
///   `!*mod.c.ggml_tensor`): `build_graph`. Required.
/// - `pub fn vars(self: *const Self, w: *std.Io.Writer) !void`: `vars`.
///   Usually `mod.Vars.fields(w, self, .{ "type", "ne" })`.
///
/// and, where the C++ overrides them:
///
/// - `opDesc(self, t, w) !void`: `op_desc`; default `ggml_op_desc(t)`.
/// - `maxNmseErr(self) f64`, `maxNmseErrBackend(self, backend) f64`,
///   `maxErr(self) f64`, `maxErrBackend(self, backend) f64`: the error bound,
///   taken from the first of these declared, in reverse order, else 1e-7.
/// - `err(self, allocator, a, b) f64`: `err`; default NMSE.
/// - `initializeTensors(self, ti: *mod.TensorInit, ctx) void`:
///   `initialize_tensors`; default uniform in [-1, 1).
/// - `reinitPerfIter(self, ti, ctx) void`: `reinit_perf_iter`.
/// - `opSize(self, t) usize`: `op_size`; default the bytes of `t` and its sources.
/// - `opFlops(self, t) u64`: `op_flops`; default 0.
/// - `runWholeGraph(self) bool`, `fusionTestNodes(self) []const *mod.c.ggml_tensor`,
///   `useWeightContext(self) bool`.
/// - `setup(self: *Self) void`: constructor logic beyond field assignment,
///   called once after the fields are set.
///
/// Gradient-mode members (`grad_eps`, `grad_expect`, `grad_nmax`, ...) are
/// not ported: the bench runs check and perf modes only.
pub const Case = struct {
    const Self = @This();

    ptr: *anyopaque,
    vtable: *const VTable,

    /// The operations every case provides, defaults filled in by `of`.
    pub const VTable = struct {
        build_graph: *const fn (*anyopaque, *mod.Graph) anyerror!*mod.c.ggml_tensor,
        vars: *const fn (*const anyopaque, *std.Io.Writer) anyerror!void,
        op_desc: *const fn (*anyopaque, *mod.c.ggml_tensor, *std.Io.Writer) anyerror!void,
        max_err: *const fn (*anyopaque, mod.c.ggml_backend_t) f64,
        err: *const fn (*anyopaque, std.mem.Allocator, []const f32, []const f32) f64,
        initialize_tensors: *const fn (*anyopaque, *mod.TensorInit, *mod.c.ggml_context) void,
        reinit_perf_iter: *const fn (*anyopaque, *mod.TensorInit, *mod.c.ggml_context) void,
        op_size: *const fn (*anyopaque, *mod.c.ggml_tensor) usize,
        op_flops: *const fn (*anyopaque, *mod.c.ggml_tensor) u64,
        run_whole_graph: *const fn (*anyopaque) bool,
        fusion_test_nodes: *const fn (*anyopaque) []const *mod.c.ggml_tensor,
        use_weight_context: *const fn (*anyopaque) bool,
    };

    /// Wraps a case struct.
    ///
    /// - `T`: the case's type.
    /// - `ptr`: the case, which must outlive the handle.
    /// - Returns: the handle.
    pub fn of(comptime T: type, ptr: *T) Self {
        return Self{ .ptr = ptr, .vtable = &Erased(T).vtable };
    }

    /// Builds the case's graph in `g`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output tensor.
    pub fn buildGraph(self: Self, g: *mod.Graph) !*mod.c.ggml_tensor {
        return self.vtable.build_graph(self.ptr, g);
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: Self, w: *std.Io.Writer) !void {
        return self.vtable.vars(self.ptr, w);
    }

    /// Writes the op's description.
    ///
    /// - `self`: the case.
    /// - `t`: the output tensor.
    /// - `w`: the destination.
    pub fn opDesc(self: Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        return self.vtable.op_desc(self.ptr, t, w);
    }

    /// The largest error accepted against the CPU on `backend`.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxErr(self: Self, backend: mod.c.ggml_backend_t) f64 {
        return self.vtable.max_err(self.ptr, backend);
    }

    /// The error between reference `a` and result `b`.
    ///
    /// - `self`: the case.
    /// - `allocator`: scratch allocator.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        return self.vtable.err(self.ptr, allocator, a, b);
    }

    /// Fills the context's tensors.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.vtable.initialize_tensors(self.ptr, ti, ctx);
    }

    /// Re-draws data-dependent inputs between timed runs.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn reinitPerfIter(self: Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.vtable.reinit_perf_iter(self.ptr, ti, ctx);
    }

    /// The bytes one execution of the op moves.
    ///
    /// - `self`: the case.
    /// - `t`: the output tensor.
    /// - Returns: the size.
    pub fn opSize(self: Self, t: *mod.c.ggml_tensor) usize {
        return self.vtable.op_size(self.ptr, t);
    }

    /// The floating-point operations one execution of the op performs, or 0.
    ///
    /// - `self`: the case.
    /// - `t`: the output tensor.
    /// - Returns: the count.
    pub fn opFlops(self: Self, t: *mod.c.ggml_tensor) u64 {
        return self.vtable.op_flops(self.ptr, t);
    }

    /// Whether the comparison runs the whole graph rather than node by node.
    ///
    /// - `self`: the case.
    /// - Returns: the flag.
    pub fn runWholeGraph(self: Self) bool {
        return self.vtable.run_whole_graph(self.ptr);
    }

    /// The nodes a fused comparison checks.
    ///
    /// - `self`: the case.
    /// - Returns: the nodes; empty for none.
    pub fn fusionTestNodes(self: Self) []const *mod.c.ggml_tensor {
        return self.vtable.fusion_test_nodes(self.ptr);
    }

    /// Whether the case builds weights in a second context.
    ///
    /// - `self`: the case.
    /// - Returns: the flag.
    pub fn useWeightContext(self: Self) bool {
        return self.vtable.use_weight_context(self.ptr);
    }

    /// The default `op_size`: the bytes of `t` and of each of its sources.
    ///
    /// - `t`: the tensor.
    /// - Returns: the size.
    pub fn defaultOpSize(t: *const mod.c.ggml_tensor) usize {
        var size = mod.c.ggml_nbytes(t);
        for (t.src) |src| {
            if (src) |s| size += mod.c.ggml_nbytes(s);
        }
        return size;
    }
};

/// The vtable for case type `T`, with `test_case`'s defaults for every
/// member `T` does not declare.
fn Erased(comptime T: type) type {
    return struct {
        const vtable: Case.VTable = .{
            .build_graph = buildGraph,
            .vars = vars,
            .op_desc = opDesc,
            .max_err = maxErr,
            .err = err,
            .initialize_tensors = initializeTensors,
            .reinit_perf_iter = reinitPerfIter,
            .op_size = opSize,
            .op_flops = opFlops,
            .run_whole_graph = runWholeGraph,
            .fusion_test_nodes = fusionTestNodes,
            .use_weight_context = useWeightContext,
        };

        fn self(ptr: *anyopaque) *T {
            return @ptrCast(@alignCast(ptr));
        }

        fn buildGraph(ptr: *anyopaque, g: *mod.Graph) anyerror!*mod.c.ggml_tensor {
            return T.buildGraph(self(ptr), g);
        }

        fn vars(ptr: *const anyopaque, w: *std.Io.Writer) anyerror!void {
            if (@hasDecl(T, "vars")) return T.vars(@ptrCast(@alignCast(ptr)), w);
        }

        fn opDesc(ptr: *anyopaque, t: *mod.c.ggml_tensor, w: *std.Io.Writer) anyerror!void {
            if (@hasDecl(T, "opDesc")) return T.opDesc(self(ptr), t, w);
            try w.writeAll(std.mem.span(mod.c.ggml_op_desc(t)));
        }

        fn maxErr(ptr: *anyopaque, backend: mod.c.ggml_backend_t) f64 {
            if (@hasDecl(T, "maxErrBackend")) return T.maxErrBackend(self(ptr), backend);
            if (@hasDecl(T, "maxErr")) return T.maxErr(self(ptr));
            if (@hasDecl(T, "maxNmseErrBackend")) return T.maxNmseErrBackend(self(ptr), backend);
            if (@hasDecl(T, "maxNmseErr")) return T.maxNmseErr(self(ptr));
            return mod.Compare.default_max_nmse_err;
        }

        fn err(ptr: *anyopaque, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
            if (@hasDecl(T, "err")) return T.err(self(ptr), allocator, a, b);
            return mod.Compare.nmse(a, b);
        }

        fn initializeTensors(ptr: *anyopaque, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
            if (@hasDecl(T, "initializeTensors")) return T.initializeTensors(self(ptr), ti, ctx);
            ti.all(ctx);
        }

        fn reinitPerfIter(ptr: *anyopaque, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
            if (@hasDecl(T, "reinitPerfIter")) T.reinitPerfIter(self(ptr), ti, ctx);
        }

        fn opSize(ptr: *anyopaque, t: *mod.c.ggml_tensor) usize {
            if (@hasDecl(T, "opSize")) return T.opSize(self(ptr), t);
            return Case.defaultOpSize(t);
        }

        fn opFlops(ptr: *anyopaque, t: *mod.c.ggml_tensor) u64 {
            if (@hasDecl(T, "opFlops")) return T.opFlops(self(ptr), t);
            return 0;
        }

        fn runWholeGraph(ptr: *anyopaque) bool {
            if (@hasDecl(T, "runWholeGraph")) return T.runWholeGraph(self(ptr));
            return false;
        }

        fn fusionTestNodes(ptr: *anyopaque) []const *mod.c.ggml_tensor {
            if (@hasDecl(T, "fusionTestNodes")) return T.fusionTestNodes(self(ptr));
            return &.{};
        }

        fn useWeightContext(ptr: *anyopaque) bool {
            if (@hasDecl(T, "useWeightContext")) return T.useWeightContext(self(ptr));
            return false;
        }
    };
}
