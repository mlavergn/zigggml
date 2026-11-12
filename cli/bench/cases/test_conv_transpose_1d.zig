//! Port of `test_conv_transpose_1d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONV_TRANSPOSE_1D.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CONV_TRANSPOSE_1D on F32 input and kernel.
pub const TestConvTranspose1d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// [input_width, input_channels, 1 (asserted by the CPU kernel), 1 (should be batch)].
    ne_input: [4]i64 = .{ 197, 32, 1, 1 },
    /// [kernel_width, output_channels, input_channels, 1 (should be batch)].
    ne_kernel: [4]i64 = .{ 16, 32, 32, 1 },
    /// Stride.
    s0: i32 = 1,
    /// Padding.
    p0: i32 = 0,
    /// Dilation.
    d0: i32 = 1,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "ne_input", "ne_kernel", "s0", "p0", "d0" });
    }

    /// Builds `conv_transpose_1d(kernel, input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const input = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne_input);
        mod.Graph.setName(input, "input");
        const kernel = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne_kernel);
        mod.Graph.setName(kernel, "kernel");
        const out: *mod.c.ggml_tensor = mod.c.ggml_conv_transpose_1d(ctx, kernel, input, self.s0, self.p0, self.d0);
        mod.Graph.setName(out, "out");
        return out;
    }
};
