//! Port of `test_l2_norm_batch` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_L2_NORM.

const std = @import("std");
const mod = @import("module.zig");

/// `n_norms` independent same-shape GGML_OP_L2_NORMs in one graph (strided
/// qkv views or contiguous tensors), their consuming adds nested so the
/// norms stay adjacent in the graph.
pub const TestL2NormBatch = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 128, 16, 16, 1 },
    n_norms: i32 = 4,
    eps: f32 = 1e-12,
    strided: bool = true,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "n_norms", "eps", "strided" });
    }

    /// Always `L2_NORM_BATCH`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("L2_NORM_BATCH");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds the norms and the sum of them.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        std.debug.assert(self.n_norms >= 2 and self.n_norms <= 8);
        const ctx = g.ctx;
        const ne = self.ne;
        const n_norms: usize = @intCast(self.n_norms);
        var parent: ?*mod.c.ggml_tensor = null;
        if (self.strided) {
            // qkv buffer
            parent = g.newTensor4d(ctx, self.type, ne[0], ne[1] * self.n_norms, ne[2], ne[3]);
        }
        var norms: [8]*mod.c.ggml_tensor = undefined;
        for (norms[0..n_norms], 0..) |*norm, t| {
            var src: *mod.c.ggml_tensor = undefined;
            if (parent) |p| {
                src = mod.c.ggml_view_4d(ctx, p, ne[0], ne[1], ne[2], ne[3], p.nb[1], p.nb[2], p.nb[3], t * @as(usize, @intCast(ne[1])) * p.nb[1]);
            } else {
                src = g.newTensor(ctx, self.type, 4, &ne);
            }
            norm.* = mod.c.ggml_l2_norm(ctx, src, self.eps);
        }
        var out = norms[n_norms - 1];
        var t: usize = n_norms - 1;
        while (t > 0) {
            t -= 1;
            out = mod.c.ggml_add(ctx, norms[t], out);
        }
        mod.Graph.setName(out, "out");
        return out;
    }
};
