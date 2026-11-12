//! Port of `test_sqr` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SQR.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SQR on a contiguous tensor.
pub const TestSqr = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Builds `a²`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_sqr(ctx, a);
        mod.Graph.setName(out, "out");
        return out;
    }
};
