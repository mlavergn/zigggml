//! Port of `test_conv_1d_dw` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_IM2COL + GGML_OP_MUL_MAT.

const std = @import("std");
const mod = @import("module.zig");

/// A depthwise 1D convolution, built from GGML_OP_IM2COL and GGML_OP_MUL_MAT.
/// The im2col of a BF16 kernel is F32, which puts F32 in src0 and BF16 in
/// src1 of the mul_mat.
pub const TestConv1dDw = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// T, C.
    ne_input: [4]i64 = .{ 64, 16, 1, 1 },
    /// K, 1, C.
    ne_kernel: [4]i64 = .{ 3, 1, 16, 1 },
    type_kernel: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    stride: i32 = 1,
    padding: i32 = 0,
    dilation: i32 = 1,

    /// Always `CONV_1D_DW`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("CONV_1D_DW");
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "ne_input", "ne_kernel", "type_kernel", "stride", "padding", "dilation" });
    }

    /// The bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `conv_1d_dw(kernel, input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const input = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne_input);
        mod.Graph.setName(input, "input");
        const kernel = g.newTensor(ctx, self.type_kernel, 4, &self.ne_kernel);
        mod.Graph.setName(kernel, "kernel");
        const out: *mod.c.ggml_tensor = mod.c.ggml_conv_1d_dw(ctx, kernel, input, self.stride, self.padding, self.dilation);
        mod.Graph.setName(out, "out");
        return out;
    }
};
