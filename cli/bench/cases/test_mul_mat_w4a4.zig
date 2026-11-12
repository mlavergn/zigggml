//! Port of `test_mul_mat_w4a4` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_MUL_MAT on the native FP4 W4A4 path (default precision):
/// `test_mul_mat` by composition.
pub const TestMulMatW4a4 = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_NVFP4),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 256,
    /// Dims 3 and 4.
    bs: [2]i64 = .{ 1, 1 },
    /// Repeat in dims 3 and 4.
    nr: [2]i64 = .{ 1, 1 },

    // State.
    /// The base case, built from the parameters in `setup`.
    base: mod.TestMulMat = .{},

    /// Builds the base case.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.base = .{ .type_a = self.type_a, .type_b = self.type_b, .m = self.m, .n = self.n, .k = self.k, .bs = self.bs, .nr = self.nr };
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

    /// Whether the base compares the whole graph.
    ///
    /// - `self`: the case.
    /// - Returns: the flag.
    pub fn runWholeGraph(self: *Self) bool {
        return self.base.runWholeGraph();
    }

    /// Builds the base graph.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        return self.base.buildGraph(g);
    }

    /// Always `MUL_MAT_W4A4`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_W4A4");
    }
};
