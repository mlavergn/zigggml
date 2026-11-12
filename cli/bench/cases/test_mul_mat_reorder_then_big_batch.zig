//! Port of `test_mul_mat_reorder_then_big_batch` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT.

const std = @import("std");
const mod = @import("module.zig");

/// A small-batch GGML_OP_MUL_MAT (which may install a reordered weight
/// layout) followed by a big-batch one on the same weight.
pub const TestMulMatReorderThenBigBatch = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_IQ3_S),
    /// Columns of the big batch.
    n: i32 = 64,
    /// Rows.
    m: i32 = 64,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "n", "m" });
    }

    /// The bound of `test_mul_mat`.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `repeat(mul_mat(w, x1), mm2) + mm2` with `mm2 = mul_mat(w, x2)`,
    /// so the small-batch mul_mat runs first.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const w = g.newTensor2d(ctx, self.type, 1 * 256, self.m);
        mod.Graph.setName(w, "w");
        const x1 = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, 1 * 256, 1);
        mod.Graph.setName(x1, "x1");
        const x2 = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, 1 * 256, self.n);
        mod.Graph.setName(x2, "x2");

        const mm1: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, w, x1);
        mod.Graph.setName(mm1, "mm1");
        const mm2: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, w, x2);
        mod.Graph.setName(mm2, "mm2");

        // mm1 is the first source of the final ADD, so the reorder-installing
        // small-batch mul_mat is computed before the big-batch one.
        const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, mod.c.ggml_repeat(ctx, mm1, mm2), mm2);
        return out;
    }
};
