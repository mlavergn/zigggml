//! Port of `test_conv_2d_dw` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONV_2D_DW.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CONV_2D_DW (direct), WHCN or (with `cwhn`) channel-contiguous.
pub const TestConv2dDw = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    ne_input: [4]i64 = .{ 64, 64, 16, 1 },
    ne_kernel: [4]i64 = .{ 3, 3, 1, 16 },
    type_kernel: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    stride: i32 = 1,
    padding: i32 = 0,
    dilation: i32 = 1,
    /// The inputs are contiguous in the channel dim rather than the width dim.
    cwhn: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "ne_input", "ne_kernel", "type_kernel", "stride", "padding", "dilation", "cwhn" });
    }

    /// Builds `conv_2d_dw_direct(kernel, input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var input = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne_input);
        mod.Graph.setName(input, "input");
        var kernel = g.newTensor(ctx, self.type_kernel, 4, &self.ne_kernel);
        mod.Graph.setName(kernel, "kernel");

        if (self.cwhn) {
            // Change the memory layout to channel-most-contiguous (CWHN), then
            // permute it back so ne matches the original input.
            input = mod.c.ggml_cont(ctx, mod.c.ggml_permute(ctx, input, 1, 2, 0, 3));
            input = mod.c.ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = mod.c.ggml_cont(ctx, mod.c.ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = mod.c.ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_conv_2d_dw_direct(ctx, kernel, input, self.stride, self.stride, self.padding, self.padding, self.dilation, self.dilation);
        mod.Graph.setName(out, "out");
        return out;
    }
};
