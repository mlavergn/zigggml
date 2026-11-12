//! Port of `test_leaky_relu` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_LEAKY_RELU.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_LEAKY_RELU, in place.
pub const TestLeakyRelu = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 10, 5, 4, 3 },
    negative_slope: f32 = 0.1,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "negative_slope" });
    }

    /// Builds `leaky_relu(a, negative_slope, inplace)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_leaky_relu(ctx, a, self.negative_slope, true);
        mod.Graph.setName(out, "out");
        return out;
    }
};
