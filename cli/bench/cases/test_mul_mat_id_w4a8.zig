//! Port of `test_mul_mat_id_w4a8` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT_ID.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_MUL_MAT_ID on the FP4 W4A8 path (`GGML_PREC_Q8` on src1):
/// `test_mul_mat_id` by composition.
pub const TestMulMatIdW4a8 = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_NVFP4),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    n_mats: i32 = 8,
    n_used: i32 = 2,
    /// Broadcast the b matrix.
    b: bool = false,
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 256,

    // State.
    /// The base case, built from the parameters in `setup`.
    base: mod.TestMulMatId = .{},

    /// Builds the base case.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.base = .{ .type_a = self.type_a, .type_b = self.type_b, .n_mats = self.n_mats, .n_used = self.n_used, .b = self.b, .m = self.m, .n = self.n, .k = self.k };
        self.base.setup();
    }

    /// Writes the case's parameters, as the base does.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try self.base.vars(w);
    }

    /// The base's bound.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        return self.base.maxNmseErrBackend(backend);
    }

    /// The base's FLOP count.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        return self.base.opFlops(t);
    }

    /// Builds the base graph and sets Q8 precision on src1 of every
    /// `MUL_MAT_ID`, which disallows 4-bit activations.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const out = self.base.buildGraph(g);
        var it = mod.Helpers.tensors(g.ctx);
        while (it.next()) |t| {
            if (t.op == mod.c.GGML_OP_MUL_MAT_ID) _ = mod.c.ggml_prec_set_src(t, mod.c.GGML_PREC_Q8, 1);
        }
        return out;
    }

    /// Fills the tensors as the base does.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.base.initializeTensors(ti, ctx);
    }

    /// Re-draws the ids between timed runs, as the base does.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn reinitPerfIter(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.base.reinitPerfIter(ti, ctx);
    }

    /// Always `MUL_MAT_ID_W4A8`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_ID_W4A8");
    }
};
