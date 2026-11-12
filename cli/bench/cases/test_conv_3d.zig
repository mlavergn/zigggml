//! Port of `test_conv_3d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONV_3D.

const std = @import("std");
const mod = @import("module.zig");

/// `ggml_calc_conv_output_size` from ggml.c.
///
/// - `ins`: the input size.
/// - `ks`: the kernel size.
/// - `s`: the stride.
/// - `p`: the padding.
/// - `d`: the dilation.
/// - Returns: the output size.
fn calcConvOutputSize(ins: i64, ks: i64, s: i32, p: i32, d: i32) i64 {
    return @divTrunc(ins + 2 * @as(i64, p) - @as(i64, d) * (ks - 1) - 1, @as(i64, s)) + 1;
}

/// GGML_OP_CONV_3D (direct) over logical 5D shapes, with an optionally
/// offset kernel view.
pub const TestConv3d = struct {
    const Self = @This();

    // Constructor parameters, in order (only `kernel_offset` has a C++
    // default). Upstream names them in capitals: N, IC, ID, IH, IW, OC, KD, KH, KW.
    n: i64 = 0,
    ic: i64 = 0,
    id: i64 = 0,
    ih: i64 = 0,
    iw: i64 = 0,
    oc: i64 = 0,
    kd: i64 = 0,
    kh: i64 = 0,
    kw: i64 = 0,
    s0: i32 = 0,
    s1: i32 = 0,
    s2: i32 = 0,
    p0: i32 = 0,
    p1: i32 = 0,
    p2: i32 = 0,
    d0: i32 = 0,
    d1: i32 = 0,
    d2: i32 = 0,
    type_kernel: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Elements before the kernel in its storage: 0 for a plain tensor.
    kernel_offset: i32 = 0,

    /// Always `CONV_3D`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("CONV_3D");
    }

    /// Writes the case's parameters, under upstream's names.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("N", self.n);
        try v.add("IC", self.ic);
        try v.add("ID", self.id);
        try v.add("IH", self.ih);
        try v.add("IW", self.iw);
        try v.add("OC", self.oc);
        try v.add("KD", self.kd);
        try v.add("KH", self.kh);
        try v.add("KW", self.kw);
        try v.add("s0", self.s0);
        try v.add("s1", self.s1);
        try v.add("s2", self.s2);
        try v.add("p0", self.p0);
        try v.add("p1", self.p1);
        try v.add("p2", self.p2);
        try v.add("d0", self.d0);
        try v.add("d1", self.d1);
        try v.add("d2", self.d2);
        try v.add("type_kernel", self.type_kernel);
        try v.add("kernel_offset", self.kernel_offset);
    }

    /// The bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// N·OC·OD·OH·OW·(2·IC·KD·KH·KW - 1).
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        const od = calcConvOutputSize(self.id, self.kd, self.s2, self.p2, self.d2);
        const oh = calcConvOutputSize(self.ih, self.kh, self.s1, self.p1, self.d1);
        const ow = calcConvOutputSize(self.iw, self.kw, self.s0, self.p0, self.d0);
        const per_out = @max(0, 2 * self.ic * self.kd * self.kh * self.kw - 1);
        return @bitCast(self.n *% self.oc *% od *% oh *% ow *% per_out);
    }

    /// Builds `conv_3d_direct(kernel, input)` on the packed 4D tensors.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        // The input is packed as [W, H, D, C*N].
        const ne_input = [4]i64{ self.iw, self.ih, self.id, self.ic * self.n };
        const input = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &ne_input);
        mod.Graph.setName(input, "input");

        // The kernel is packed as [KW, KH, KD, IC*OC].
        const ne_kernel = [4]i64{ self.kw, self.kh, self.kd, self.ic * self.oc };
        var kernel: *mod.c.ggml_tensor = undefined;
        if (self.kernel_offset == 0) {
            kernel = g.newTensor(ctx, self.type_kernel, 4, &ne_kernel);
        } else {
            const storage = g.newTensor1d(ctx, self.type_kernel, self.kw * self.kh * self.kd * self.ic * self.oc + self.kernel_offset);
            const es = mod.c.ggml_type_size(self.type_kernel.raw());
            const kw: usize = @intCast(self.kw);
            const kh: usize = @intCast(self.kh);
            const kd: usize = @intCast(self.kd);
            const off: usize = @intCast(self.kernel_offset);
            kernel = mod.c.ggml_view_4d(ctx, storage, self.kw, self.kh, self.kd, self.ic * self.oc, kw * es, kw * kh * es, kw * kh * kd * es, off * es);
        }
        mod.Graph.setName(kernel, "kernel");

        const out: *mod.c.ggml_tensor = mod.c.ggml_conv_3d_direct(ctx, kernel, input, self.s0, self.s1, self.s2, self.p0, self.p1, self.p2, self.d0, self.d1, self.d2, @intCast(self.ic), @intCast(self.n), @intCast(self.oc));
        mod.Graph.setName(out, "out");
        return out;
    }
};
