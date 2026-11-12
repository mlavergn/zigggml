//! Port of `test_out_prod` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_OUT_PROD.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_OUT_PROD, with `b` optionally transposed and broadcast in dims 3 and 4.
pub const TestOutProd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 32,
    /// Dims 3 and 4.
    bs: [2]i64 = .{ 10, 10 },
    /// Repeat in dims 3 and 4.
    nr: [2]i64 = .{ 2, 2 },
    trans_b: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_a", "type_b", "m", "n", "k", "bs", "nr", "trans_b" });
    }

    /// The bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `out_prod(a, b)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor4d(ctx, self.type_a, self.m, self.k, self.bs[0], self.bs[1]);
        mod.Graph.setName(a, "a");

        var b: *mod.c.ggml_tensor = undefined;
        if (self.trans_b) {
            b = g.newTensor4d(ctx, self.type_b, self.k, self.n, self.bs[0] * self.nr[0], self.bs[1] * self.nr[1]);
            b = mod.c.ggml_transpose(ctx, b);
        } else {
            b = g.newTensor4d(ctx, self.type_b, self.n, self.k, self.bs[0] * self.nr[0], self.bs[1] * self.nr[1]);
        }
        mod.Graph.setName(b, "b");

        const out: *mod.c.ggml_tensor = mod.c.ggml_out_prod(ctx, a, b);
        mod.Graph.setName(out, "out");
        return out;
    }
};
