//! Port of `test_flash_attn_ext` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_FLASH_ATTN_EXT.

const std = @import("std");
const mod = @import("module.zig");

/// `GGML_PAD`: `x` rounded up to a multiple of `n`, a power of two.
fn pad(x: i64, n: i64) i64 {
    return (x + n - 1) & ~(n - 1);
}

/// GGML_OP_FLASH_ATTN_EXT with optional mask, sinks, ALiBi, softcap,
/// grouped-query repeats, permuted inputs and K/V cache views.
pub const TestFlashAttnExt = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// K head size.
    hsk: i64 = 128,
    /// V head size.
    hsv: i64 = 128,
    /// Number of heads.
    nh: i64 = 32,
    /// Repeat in dims 2 and 3, for grouped-query attention.
    nr23: [2]i64 = .{ 1, 1 },
    /// KV size.
    kv: i64 = 96,
    /// Batch size.
    nb: i64 = 8,
    /// Use a mask.
    mask: bool = true,
    /// Use sinks.
    sinks: bool = false,
    /// ALiBi.
    max_bias: f32 = 0.0,
    /// Gemma 2.
    logit_softcap: f32 = 0.0,
    prec: mod.GgmlPrec = .of(mod.c.GGML_PREC_F32),
    type_k: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    type_v: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    permute: [4]i32 = .{ 0, 1, 2, 3 },
    /// Create K/V as views of a larger buffer, like a KV cache.
    kv_view: bool = true,
    v_is_view_of_k: bool = false,
    n_kv_max: i64 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("hsk", self.hsk);
        try v.add("hsv", self.hsv);
        try v.add("nh", self.nh);
        try v.add("nr23", self.nr23);
        try v.add("kv", self.kv);
        try v.add("nb", self.nb);
        try v.add("mask", self.mask);
        try v.add("sinks", self.sinks);
        try v.add("max_bias", self.max_bias);
        try v.add("logit_softcap", self.logit_softcap);
        try v.add("prec", self.prec);
        try v.add("type_K", self.type_k);
        try v.add("type_V", self.type_v);
        try v.add("permute", self.permute);
        try v.add("kv_view", self.kv_view);
        try v.add("v_is_view_of_k", self.v_is_view_of_k);
        try v.add("n_kv_max", self.n_kv_max);
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// The matmul costs: Q*K^T and P*V per head.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        return @intCast((2 * self.nh * self.nr23[0] * self.nb * (self.hsk + self.hsv) * self.kv) * self.nr23[1]);
    }

    /// Builds `flash_attn_ext(q, k, v, m)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const hsk_padded = pad(self.hsk, mod.c.ggml_blck_size(self.type_k.raw()));
        const hsv_padded = pad(self.hsv, mod.c.ggml_blck_size(self.type_v.raw()));

        const q = self.createPermuted(g, mod.c.GGML_TYPE_F32, hsk_padded, self.nb, self.nh * self.nr23[0], self.nr23[1], false);
        mod.Graph.setName(q, "q");

        // The K tensor is usually a view of the K cache.
        const k = self.createPermuted(g, self.type_k.raw(), hsk_padded, self.kv, self.nh, self.nr23[1], self.kv_view);
        mod.Graph.setName(k, "k");

        var v: *mod.c.ggml_tensor = undefined;
        if (self.v_is_view_of_k) {
            // The V cache is a sub-view of the K cache, as in some MLA-based models.
            std.debug.assert(self.type_k == self.type_v and hsv_padded <= hsk_padded);
            v = mod.c.ggml_view_4d(ctx, k, hsv_padded, self.kv, self.nh, self.nr23[1], k.nb[1], k.nb[2], k.nb[3], 0);
        } else {
            // The V tensor is usually a view of the V cache.
            v = self.createPermuted(g, self.type_v.raw(), hsv_padded, self.kv, self.nh, self.nr23[1], self.kv_view);
        }
        mod.Graph.setName(v, "v");

        var m: ?*mod.c.ggml_tensor = null;
        if (self.mask) {
            const mt = g.newTensor4d(ctx, mod.c.GGML_TYPE_F16, self.kv, self.nb, 1, self.nr23[1]);
            mod.Graph.setName(mt, "m");
            m = mt;
        }

        var s: ?*mod.c.ggml_tensor = null;
        if (self.sinks) {
            const st = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, q.ne[2]);
            mod.Graph.setName(st, "s");
            s = st;
        }

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.hsk)));
        const out: *mod.c.ggml_tensor = mod.c.ggml_flash_attn_ext(ctx, q, k, v, m, scale, self.max_bias, self.logit_softcap);
        mod.c.ggml_flash_attn_ext_add_sinks(out, s);
        mod.c.ggml_flash_attn_ext_set_n_kv_max(out, @intCast(self.n_kv_max));
        _ = mod.c.ggml_prec_set_acc(out, self.prec.raw());
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the sinks over a wider range, the mask with a KQ mask (sparse
    /// when `n_kv_max > 0`), and the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            const name = std.mem.sliceTo(&t.name, 0);
            if (std.mem.eql(u8, name, "s")) {
                // More noticeable sink values, so a wrong implementation fails.
                ti.uniform(t, -10.0, 10.0);
            } else if (std.mem.eql(u8, name, "m")) {
                if (self.n_kv_max > 0) {
                    ti.kqMaskSparse(t, self.n_kv_max);
                } else {
                    ti.kqMask(t, -1.0, 1.0);
                }
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }

    /// A tensor of extents `ne0..ne3` after `permute`, created with the
    /// permuted extents (as a view of a buffer twice as tall for `is_view`)
    /// and permuted back.
    fn createPermuted(self: *Self, g: *mod.Graph, ty: mod.c.enum_ggml_type, ne0: i64, ne1: i64, ne2: i64, ne3: i64, is_view: bool) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = [4]i64{ ne0, ne1, ne2, ne3 };
        var ne_perm: [4]i64 = undefined;
        for (0..4) |i| ne_perm[@intCast(self.permute[i])] = ne[i];
        var t: *mod.c.ggml_tensor = undefined;
        if (is_view) {
            const t0 = g.newTensor4d(ctx, ty, ne_perm[0], 2 * ne_perm[1], ne_perm[2], ne_perm[3]);
            t = mod.c.ggml_view_4d(ctx, t0, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3], t0.nb[1], t0.nb[2], t0.nb[3], 0);
        } else {
            t = g.newTensor4d(ctx, ty, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3]);
        }
        if (!std.mem.eql(i32, &self.permute, &.{ 0, 1, 2, 3 })) {
            t = mod.c.ggml_permute(ctx, t, self.permute[0], self.permute[1], self.permute[2], self.permute[3]);
        }
        return t;
    }
};
