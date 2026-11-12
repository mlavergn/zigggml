//! Port of `test_im2col` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_IM2COL.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_IM2COL, 1D or 2D.
pub const TestIm2col = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_input: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_kernel: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    dst_type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// [input_width, input_height, input_channels, 1].
    ne_input: [4]i64 = .{ 10, 10, 3, 1 },
    /// [kernel_width, kernel_height, input_channels, 1].
    ne_kernel: [4]i64 = .{ 3, 3, 3, 1 },
    /// Stride.
    s0: i32 = 1,
    s1: i32 = 1,
    /// Padding.
    p0: i32 = 1,
    p1: i32 = 1,
    /// Dilation.
    d0: i32 = 1,
    d1: i32 = 1,
    /// 2D mode (upstream `is_2D`).
    is_2d: bool = true,

    /// Writes the case's parameters, under upstream's names.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_input", "type_kernel", "dst_type", "ne_input", "ne_kernel", "s0", "s1", "p0", "p1", "d0", "d1" });
        var v = mod.Vars.init(w);
        v.first = false;
        try v.add("is_2D", self.is_2d);
    }

    /// Builds `im2col(kernel, input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const input = g.newTensor(ctx, self.type_input, 4, &self.ne_input);
        mod.Graph.setName(input, "input");
        const kernel = g.newTensor(ctx, self.type_kernel, 4, &self.ne_kernel);
        mod.Graph.setName(kernel, "kernel");
        const out: *mod.c.ggml_tensor = mod.c.ggml_im2col(ctx, kernel, input, self.s0, self.s1, self.p0, self.p1, self.d0, self.d1, self.is_2d, self.dst_type.raw());
        mod.Graph.setName(out, "out");
        return out;
    }
};
