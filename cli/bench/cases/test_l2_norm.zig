//! Port of `test_l2_norm` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_L2_NORM.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_L2_NORM of each row, of an optionally viewed or row-permuted input.
pub const TestL2Norm = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 64, 320, 1 },
    eps: f32 = 1e-12,
    /// A view of half of each dimension.
    v: bool = false,
    /// Rows that are not contiguous (a permuted input).
    noncontig_rows: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "eps", "v", "noncontig_rows" });
    }

    /// Builds `l2_norm(a, eps)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const ne_a: [4]i64 = if (self.noncontig_rows) .{ ne[1], ne[0], ne[2], ne[3] } else ne;
        var a = g.newTensor(ctx, self.type, 4, &ne_a);
        mod.Graph.setName(a, "a");

        if (self.noncontig_rows) {
            a = mod.c.ggml_permute(ctx, a, 1, 0, 2, 3);
            mod.Graph.setName(a, "permuted a");
        }
        if (self.v) {
            a = mod.c.ggml_view_4d(ctx, a, @divTrunc(a.ne[0], 2), @divTrunc(a.ne[1], 2), @divTrunc(a.ne[2], 2), @divTrunc(a.ne[3], 2), a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view of a");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_l2_norm(ctx, a, self.eps);
        mod.Graph.setName(out, "out");
        return out;
    }
};
