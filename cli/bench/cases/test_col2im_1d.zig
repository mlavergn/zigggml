//! Port of `test_col2im_1d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_COL2IM_1D.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_COL2IM_1D: overlap-add of `t_in` columns of `k * oc` values.
pub const TestCol2im1d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Kernel size (upstream `K`).
    k: i64 = 4,
    /// Output channels (upstream `OC`).
    oc: i64 = 3,
    /// Input length, the number of columns (upstream `T_in`).
    t_in: i64 = 7,
    /// Stride.
    s0: i32 = 2,
    /// Padding cropped from both sides.
    p0: i32 = 0,

    /// Writes the case's parameters, under upstream's names.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type", self.type);
        try v.add("K", self.k);
        try v.add("OC", self.oc);
        try v.add("T_in", self.t_in);
        try v.add("s0", self.s0);
        try v.add("p0", self.p0);
    }

    /// Exact for F32; looser for reduced-precision types.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return if (self.type.raw() == mod.c.GGML_TYPE_F32) 1e-7 else 5e-4;
    }

    /// Builds `col2im_1d(cols)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const cols = g.newTensor2d(ctx, self.type, self.k * self.oc, self.t_in);
        mod.Graph.setName(cols, "cols");
        const out: *mod.c.ggml_tensor = mod.c.ggml_col2im_1d(ctx, cols, self.s0, @intCast(self.oc), self.p0);
        mod.Graph.setName(out, "out");
        return out;
    }
};
