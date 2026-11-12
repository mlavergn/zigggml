//! Port of `test_ssm_conv_bias_silu` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SSM_CONV + GGML_OP_ADD + GGML_UNARY_OP_SILU.

const std = @import("std");
const mod = @import("module.zig");

/// `silu(ssm_conv(a, b) [+ bias])`, with an optional channel-wise bias,
/// compared as one fused graph.
pub const TestSsmConvBiasSilu = struct {
    const Self = @This();

    // Constructor parameters, in order (none has a C++ default).
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 0, 0, 0, 0 },
    ne_b: [4]i64 = .{ 0, 0, 0, 0 },
    /// Add a channel-wise bias between the convolution and the SiLU.
    fuse_bias: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "ne_b", "fuse_bias" });
    }

    /// Always `SSM_CONV_BIAS_SILU`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("SSM_CONV_BIAS_SILU");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `silu(ssm_conv(a, b) + bias)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        const b = g.newTensor(ctx, self.type, 4, &self.ne_b);
        mod.Graph.setName(a, "a");
        mod.Graph.setName(b, "b");

        var out: *mod.c.ggml_tensor = mod.c.ggml_ssm_conv(ctx, a, b);

        if (self.fuse_bias) {
            const bias = g.newTensor1d(ctx, self.type, out.ne[0]);
            mod.Graph.setName(bias, "bias");
            out = mod.c.ggml_add(ctx, out, bias);
        }

        out = mod.c.ggml_silu(ctx, out);
        mod.Graph.setName(out, "out");
        return out;
    }
};
