//! Port of `test_norm_mul_add` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_NORM + GGML_OP_MUL + GGML_OP_ADD.

const std = @import("std");
const mod = @import("module.zig");

/// `norm(a) * w + b`, compared as one fused graph.
pub const TestNormMulAdd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 128, 2, 1, 1 },
    eps: f32 = 1e-5,
    /// Whether `a` is larger than `w` and `b`, which broadcast over it.
    broadcast: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "eps", "broadcast" });
    }

    /// Always `NORM_MUL_ADD`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("NORM_MUL_ADD");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `norm(a + w + b) * w + b`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const broadcast_dims = [4]i64{ ne[0], ne[1] * 2, ne[2] * 2, ne[3] * 2 };

        var a = g.newTensor(ctx, self.type, 4, if (self.broadcast) &broadcast_dims else &self.ne);
        const w = g.newTensor(ctx, self.type, 4, &self.ne);
        const b = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        mod.Graph.setName(w, "w");
        mod.Graph.setName(b, "b");

        // Use a, w and b early to avoid OP_NONE in the graph.
        a = mod.c.ggml_add(ctx, mod.c.ggml_add(ctx, a, w), b);

        const n: *mod.c.ggml_tensor = mod.c.ggml_norm(ctx, a, self.eps);
        const m: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, n, w);
        const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, m, b);
        mod.Graph.setName(out, "out");
        return out;
    }
};
