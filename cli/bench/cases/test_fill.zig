//! Port of `test_fill` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_FILL.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_FILL: sets every element to a constant.
pub const TestFill = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// The fill value.
    c: f32 = 0.0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 4, 3 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "c" });
    }

    /// Builds `fill(a, c)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor4d(ctx, self.type, self.ne[0], self.ne[1], self.ne[2], self.ne[3]);
        mod.Graph.setName(a, "a");

        const out: *mod.c.ggml_tensor = mod.c.ggml_fill(ctx, a, self.c);
        mod.Graph.setName(out, "out");
        return out;
    }
};
