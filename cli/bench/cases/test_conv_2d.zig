//! Port of `test_conv_2d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONV_2D.

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

/// GGML_OP_CONV_2D (direct), WHCN or (with `cwhn`) channel-contiguous, with
/// an optionally offset kernel view.
pub const TestConv2d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    ne_input: [4]i64 = .{ 64, 64, 16, 1 },
    ne_kernel: [4]i64 = .{ 3, 3, 1, 16 },
    type_kernel: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    stride0: i32 = 1,
    stride1: i32 = 1,
    padding0: i32 = 0,
    padding1: i32 = 0,
    dilation0: i32 = 1,
    dilation1: i32 = 1,
    /// The inputs are contiguous in the channel dim rather than the width dim.
    cwhn: bool = false,
    /// Elements before the kernel in its storage: 0 for a plain tensor.
    kernel_offset: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "ne_input", "ne_kernel", "type_kernel", "stride0", "stride1", "padding0", "padding1", "dilation0", "dilation1", "cwhn", "kernel_offset" });
    }

    /// The bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// The matmul cost only: KxCRS @ CRSxNPQ = KxNPQ, KxNPQx(2·CRS - 1).
    /// Both output sizes use the dim-0 stride, padding and dilation, as upstream does.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        const w_in = self.ne_input[0];
        const h_in = self.ne_input[1];
        const kw = self.ne_kernel[0];
        const kh = self.ne_kernel[1];
        const cin = self.ne_kernel[2];
        const cout = self.ne_kernel[3];
        const n = self.ne_input[3];
        const oh = calcConvOutputSize(h_in, kh, self.stride0, self.padding0, self.dilation0);
        const ow = calcConvOutputSize(w_in, kw, self.stride0, self.padding0, self.dilation0);
        const crs = cin * kh * kw;
        const npq = n * oh * ow;
        return @bitCast(cout * npq * (2 * crs - 1));
    }

    /// Builds `conv_2d_direct(kernel, input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var input = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne_input);
        mod.Graph.setName(input, "input");

        const nk = self.ne_kernel;
        var kernel: *mod.c.ggml_tensor = undefined;
        if (self.kernel_offset == 0) {
            kernel = g.newTensor(ctx, self.type_kernel, 4, &nk);
        } else {
            const nelem = nk[0] * nk[1] * nk[2] * nk[3];
            const storage = g.newTensor1d(ctx, self.type_kernel, nelem + self.kernel_offset);
            const es = mod.c.ggml_type_size(self.type_kernel.raw());
            const k0: usize = @intCast(nk[0]);
            const k1: usize = @intCast(nk[1]);
            const k2: usize = @intCast(nk[2]);
            const off: usize = @intCast(self.kernel_offset);
            kernel = mod.c.ggml_view_4d(ctx, storage, nk[0], nk[1], nk[2], nk[3], k0 * es, k0 * k1 * es, k0 * k1 * k2 * es, off * es);
        }
        mod.Graph.setName(kernel, "kernel");

        if (self.cwhn) {
            // Change the memory layout to channel-most-contiguous (CWHN), then
            // permute it back so ne matches the original input.
            input = mod.c.ggml_cont(ctx, mod.c.ggml_permute(ctx, input, 1, 2, 0, 3));
            input = mod.c.ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = mod.c.ggml_cont(ctx, mod.c.ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = mod.c.ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_conv_2d_direct(ctx, kernel, input, self.stride0, self.stride1, self.padding0, self.padding1, self.dilation0, self.dilation1);
        mod.Graph.setName(out, "out");
        return out;
    }
};
