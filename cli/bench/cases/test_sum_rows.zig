//! Port of `test_sum_rows` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SUM_ROWS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SUM_ROWS of an optionally sliced and permuted input.
pub const TestSumRows = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },
    permute: bool = false,
    slice: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "permute", "slice" });
    }

    /// Builds `sum_rows(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        var a = g.newTensor(ctx, self.type, 4, &ne);
        mod.Graph.setName(a, "a");

        if (self.slice) {
            a = mod.c.ggml_view_4d(ctx, a, ne[0], ne[1], @divTrunc(ne[2], 2), ne[3] - 1, a.nb[1], a.nb[2] * 2, a.nb[3], a.nb[3]);
        }
        if (self.permute) {
            a = mod.c.ggml_permute(ctx, a, 0, 2, 3, 1);
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_sum_rows(ctx, a);
        mod.Graph.setName(out, "out");
        return out;
    }
};
