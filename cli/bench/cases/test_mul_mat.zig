//! Port of `test_mul_mat` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_MUL_MAT, C^T = A * B^T: (k, m) * (k, n) => (m, n), with optional
/// permuted, overlapping, strided and padded inputs, and `o` outputs summed.
pub const TestMulMat = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 32,
    /// Dims 3 and 4.
    bs: [2]i64 = .{ 10, 10 },
    /// Repeat in dims 3 and 4.
    nr: [2]i64 = .{ 2, 2 },
    /// Permutation of dimensions.
    per: [4]i64 = .{ 0, 1, 2, 3 },
    /// Size of k in memory: a non-contiguous view for k_v > k, none for 0.
    k_v: i64 = 0,
    /// Number of outputs.
    o: u32 = 1,
    /// a and b are overlapping views of the same tensor.
    src_overlap: bool = false,
    /// Rows of a in memory: the batches of a are strided for m_v > m, none for 0.
    m_v: i64 = 0,
    /// Bytes after the m_v rows of each batch of a.
    pad: i64 = 0,

    // State.
    gf: ?*mod.c.ggml_cgraph = null,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_a", "type_b", "m", "n", "k", "bs", "nr", "per", "k_v", "o", "src_overlap", "m_v", "pad" });
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// A looser bound where Blackwell quantizes activations to FP4, or a
    /// backend computes natively in FP8.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        const ta = self.type_a.raw();
        if ((ta == mod.c.GGML_TYPE_MXFP4 or ta == mod.c.GGML_TYPE_NVFP4) and
            !mod.Helpers.graphMulMatHiPrecAct(self.gf, mod.c.GGML_OP_MUL_MAT) and
            mod.Helpers.backendHasFeature(backend, "BLACKWELL_NATIVE_FP4")) return 2e-2;
        if (ta == mod.c.GGML_TYPE_F8_E4M3 and mod.Helpers.backendHasFeature(backend, "NATIVE_FP8")) return 5e-3;
        return self.maxNmseErr();
    }

    /// 2·m·n·k per batch.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        return @intCast(2 * self.m * self.n * self.k * self.bs[0] * self.nr[0] * self.bs[1] * self.nr[1]);
    }

    /// Builds `mul_mat(a, b)`, summed `o` times.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        self.gf = g.gf;
        var a: *mod.c.ggml_tensor = undefined;
        var b: *mod.c.ggml_tensor = undefined;
        const per = self.per;
        const npermuted = @as(u32, @intFromBool(per[0] != 0)) + @intFromBool(per[1] != 1) + @intFromBool(per[2] != 2) + @intFromBool(per[3] != 3);
        if (npermuted > 0) {
            std.debug.assert(npermuted == 2);
            std.debug.assert(self.k_v == 0);
            std.debug.assert(self.m_v == 0);
            // Create tensors with the permuted dimensions, then permute them back.
            const ne_a = [4]i64{ self.k, self.m, self.bs[0], self.bs[1] };
            const ne_b = [4]i64{ self.k, self.n, self.bs[0] * self.nr[0], self.bs[1] * self.nr[1] };
            const p = [4]usize{ @intCast(per[0]), @intCast(per[1]), @intCast(per[2]), @intCast(per[3]) };
            a = g.newTensor4d(ctx, self.type_a, ne_a[p[0]], ne_a[p[1]], ne_a[p[2]], ne_a[p[3]]);
            b = g.newTensor4d(ctx, self.type_b, ne_b[p[0]], ne_b[p[1]], ne_b[p[2]], ne_b[p[3]]);
            mod.Graph.setName(a, "a");
            mod.Graph.setName(b, "b");
            a = mod.c.ggml_permute(ctx, a, @intCast(per[0]), @intCast(per[1]), @intCast(per[2]), @intCast(per[3]));
            b = mod.c.ggml_permute(ctx, b, @intCast(per[0]), @intCast(per[1]), @intCast(per[2]), @intCast(per[3]));
            mod.Graph.setName(a, "a_permuted");
            mod.Graph.setName(b, "b_permuted");
        } else if (self.src_overlap) {
            std.debug.assert(self.type_a == self.type_b);
            // a and b are interleaved views of one tensor (fused QKV).
            const base = g.newTensor4d(ctx, self.type_a, 2 * self.k, @max(self.m, self.n), self.bs[0] * self.nr[0], self.bs[1] * self.nr[1]);
            mod.Graph.setName(base, "base");
            a = mod.c.ggml_view_4d(ctx, base, self.k, self.m, self.bs[0], self.bs[1], base.nb[1], base.nb[2], base.nb[3], 0);
            b = mod.c.ggml_view_4d(ctx, base, self.k, self.n, self.bs[0] * self.nr[0], self.bs[1] * self.nr[1], base.nb[1], base.nb[2], base.nb[3], @as(usize, @intCast(self.k)) * mod.c.ggml_type_size(self.type_a.raw()));
            mod.Graph.setName(a, "a");
            mod.Graph.setName(b, "b");
        } else {
            const k_physical = if (self.k_v == 0) self.k else self.k_v;
            const m_physical = if (self.m_v == 0) self.m else self.m_v + @intFromBool(self.pad != 0);
            a = g.newTensor4d(ctx, self.type_a, k_physical, m_physical, self.bs[0], self.bs[1]);
            b = g.newTensor4d(ctx, self.type_b, k_physical, self.n, self.bs[0] * self.nr[0], self.bs[1] * self.nr[1]);
            if (self.k_v != 0) {
                std.debug.assert(self.k_v > self.k);
                a = mod.c.ggml_view_4d(ctx, a, self.k, self.m, self.bs[0], self.bs[1], a.nb[1], a.nb[2], a.nb[3], 0);
                b = mod.c.ggml_view_4d(ctx, b, self.k, self.n, self.bs[0] * self.nr[0], self.bs[1] * self.nr[1], b.nb[1], b.nb[2], b.nb[3], 0);
            }
            if (self.m_v != 0) {
                std.debug.assert(self.m_v > self.m);
                a = mod.c.ggml_view_4d(ctx, a, self.k, self.m, self.bs[0], self.bs[1], a.nb[1], @as(usize, @intCast(self.m_v)) * a.nb[1] + @as(usize, @intCast(self.pad)), a.nb[3], 0);
            }
            mod.Graph.setName(a, "a");
            mod.Graph.setName(b, "b");
        }

        var out: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, a, b);
        mod.Graph.setName(out, "out");
        var i: u32 = 1;
        while (i < self.o) : (i += 1) {
            const out2: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, a, b);
            mod.Graph.setName(out2, "out2");
            out = mod.c.ggml_add(ctx, out, out2);
        }
        return out;
    }

    /// Several outputs are compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true when `o > 1`.
    pub fn runWholeGraph(self: *Self) bool {
        return self.o > 1;
    }

    /// Always `MUL_MAT`, whatever the last node is.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll(std.mem.span(mod.c.ggml_op_name(mod.c.GGML_OP_MUL_MAT)));
    }
};
