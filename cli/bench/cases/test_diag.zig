//! Port of `test_diag` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_DIAG.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_DIAG: a diagonal matrix from each row vector.
pub const TestDiag = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Row vectors: ne[1] == 1.
    ne: [4]i64 = .{ 10, 1, 4, 3 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Builds `diag(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        std.debug.assert(self.ne[1] == 1);
        const a = g.newTensor4d(ctx, self.type, self.ne[0], self.ne[1], self.ne[2], self.ne[3]);
        mod.Graph.setName(a, "a");

        const out: *mod.c.ggml_tensor = mod.c.ggml_diag(ctx, a);
        mod.Graph.setName(out, "out");
        return out;
    }
};
