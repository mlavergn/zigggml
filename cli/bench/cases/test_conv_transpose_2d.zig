//! Port of `test_conv_transpose_2d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONV_TRANSPOSE_2D.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CONV_TRANSPOSE_2D, unpadded, on an F32 input.
pub const TestConvTranspose2d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// [input_width, input_height, input_channels, 1].
    ne_input: [4]i64 = .{ 10, 10, 3, 1 },
    /// [kernel_width, kernel_height, input_channels, 1].
    ne_kernel: [4]i64 = .{ 3, 3, 3, 1 },
    stride: i32 = 1,
    kernel_type: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "kernel_type", "ne_input", "ne_kernel", "stride" });
    }

    /// The default 1e-7 is too small for Vulkan.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `conv_transpose_2d_p0(kernel, input, stride)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const input = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne_input);
        mod.Graph.setName(input, "input");
        const kernel = g.newTensor(ctx, self.kernel_type, 4, &self.ne_kernel);
        mod.Graph.setName(kernel, "kernel");
        const out: *mod.c.ggml_tensor = mod.c.ggml_conv_transpose_2d_p0(ctx, kernel, input, self.stride);
        mod.Graph.setName(out, "out");
        return out;
    }
};
