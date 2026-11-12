//! Port of `test_im2col_3d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_IM2COL_3D.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_IM2COL_3D, on a contiguous or (with `v`) non-contiguous input.
pub const TestIm2col3d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_input: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_kernel: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    dst_type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// [N*IC, ID, IH, IW] (upstream's comment).
    ne_input: [4]i64 = .{ 10, 10, 10, 9 },
    /// [OC*IC, KD, KH, KW] (upstream's comment).
    ne_kernel: [4]i64 = .{ 3, 3, 3, 1 },
    /// Input channels (upstream `IC`).
    ic: i64 = 3,
    /// Stride.
    s0: i32 = 1,
    s1: i32 = 1,
    s2: i32 = 1,
    /// Padding.
    p0: i32 = 1,
    p1: i32 = 1,
    p2: i32 = 1,
    /// Dilation.
    d0: i32 = 1,
    d1: i32 = 1,
    d2: i32 = 1,
    /// View: a non-contiguous input.
    v: bool = false,

    /// Writes the case's parameters, under upstream's names.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type_input", self.type_input);
        try v.add("type_kernel", self.type_kernel);
        try v.add("dst_type", self.dst_type);
        try v.add("ne_input", self.ne_input);
        try v.add("ne_kernel", self.ne_kernel);
        try v.add("IC", self.ic);
        try v.add("s0", self.s0);
        try v.add("s1", self.s1);
        try v.add("s2", self.s2);
        try v.add("p0", self.p0);
        try v.add("p1", self.p1);
        try v.add("p2", self.p2);
        try v.add("d0", self.d0);
        try v.add("d1", self.d1);
        try v.add("d2", self.d2);
        try v.add("v", self.v);
    }

    /// Builds `im2col_3d(kernel, input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne_input;
        var input = g.newTensor(ctx, self.type_input, 4, &ne);
        mod.Graph.setName(input, "input");
        if (self.v) {
            input = mod.c.ggml_view_4d(ctx, input, ne[0] - 2, ne[1] - 2, ne[2] - 2, ne[3] - 2, input.nb[1], input.nb[2], input.nb[3], 0);
            mod.Graph.setName(input, "view_of_input");
        }
        const kernel = g.newTensor(ctx, self.type_kernel, 4, &self.ne_kernel);
        mod.Graph.setName(kernel, "kernel");
        const out: *mod.c.ggml_tensor = mod.c.ggml_im2col_3d(ctx, kernel, input, self.ic, self.s0, self.s1, self.s2, self.p0, self.p1, self.p2, self.d0, self.d1, self.d2, self.dst_type.raw());
        mod.Graph.setName(out, "out");
        return out;
    }
};
