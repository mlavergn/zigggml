//! Port of `test_soft_max_back` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SOFT_MAX_BACK.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SOFT_MAX_BACK with a scale and ALiBi bias.
pub const TestSoftMaxBack = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },
    scale: f32 = 1.0,
    max_bias: f32 = 0.0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "scale", "max_bias" });
    }

    /// Builds `soft_max_ext_back(a, b, scale, max_bias)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        // Upstream names `a` twice and leaves `b` unnamed.
        const b = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_soft_max_ext_back(ctx, a, b, self.scale, self.max_bias);
        mod.Graph.setName(out, "out");
        return out;
    }
};
