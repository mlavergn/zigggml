//! Port of `test_ssm_conv` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SSM_CONV.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SSM_CONV: the Mamba depthwise causal convolution.
pub const TestSsmConv = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 10, 10, 10, 1 },
    ne_b: [4]i64 = .{ 3, 3, 1, 1 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "ne_b" });
    }

    /// Builds `ssm_conv(a, b)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        const b = g.newTensor(ctx, self.type, 4, &self.ne_b);
        return mod.c.ggml_ssm_conv(ctx, a, b);
    }
};
