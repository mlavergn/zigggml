//! Port of `test_norm` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_NORM.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_NORM on a contiguous, viewed or row-permuted input.
pub const TestNorm = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    /// Whether `a` is a non-contiguous view.
    v: bool = false,
    eps: f32 = 1e-6,
    /// Whether `a`'s rows are non-contiguous (dims 0 and 1 permuted).
    noncontig_rows: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "v", "eps", "noncontig_rows" });
    }

    /// Builds `norm(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne_a: [4]i64 = if (self.noncontig_rows) .{ self.ne[1], self.ne[0], self.ne[2], self.ne[3] } else self.ne;
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

        const out: *mod.c.ggml_tensor = mod.c.ggml_norm(ctx, a, self.eps);
        mod.Graph.setName(out, "out");
        return out;
    }
};
