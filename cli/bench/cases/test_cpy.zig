//! Port of `test_cpy` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CPY.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CPY between types, from an optionally permuted or transposed
/// source to an optionally reshaped, permuted or strided destination.
pub const TestCpy = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_src: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_dst: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_src: [4]i64 = .{ 10, 10, 10, 1 },
    /// The destination's shape; all -1 for the source's.
    ne_dst: [4]i64 = .{ -1, -1, -1, -1 },
    permute_src: [4]i64 = .{ 0, 0, 0, 0 },
    permute_dst: [4]i64 = .{ 0, 0, 0, 0 },
    /// Upstream's `_src_transpose`.
    transpose_src: bool = false,
    /// When set, dst is a view into a larger buffer of this shape (strided).
    dst_alloc: [4]i64 = .{ 0, 0, 0, 0 },

    // State, derived in `setup`.
    src_use_permute: bool = false,
    dst_use_permute: bool = false,
    use_dst_shape: bool = false,
    use_dst_alloc: bool = false,

    /// Derives the flags from the parameters.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        const ps = self.permute_src;
        const pd = self.permute_dst;
        const nd = self.ne_dst;
        self.src_use_permute = ps[0] + ps[1] + ps[2] + ps[3] > 0;
        self.dst_use_permute = pd[0] + pd[1] + pd[2] + pd[3] > 0;
        self.use_dst_shape = nd[0] >= 0 and nd[1] >= 0 and nd[2] >= 0 and nd[3] >= 0;
        self.use_dst_alloc = self.dst_alloc[0] > 0;
    }

    /// Writes the case's parameters; `ne_dst` and `dst_alloc` only when used.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type_src", self.type_src);
        try v.add("type_dst", self.type_dst);
        try v.add("ne_src", self.ne_src);
        if (self.use_dst_alloc or self.use_dst_shape) try v.add("ne_dst", self.ne_dst);
        try v.add("permute_src", self.permute_src);
        try v.add("permute_dst", self.permute_dst);
        try v.add("_src_transpose", self.transpose_src);
        if (self.use_dst_alloc) try v.add("dst_alloc", self.dst_alloc);
    }

    fn totalElements(self: *const Self) i64 {
        return self.ne_src[0] * self.ne_src[1] * self.ne_src[2] * self.ne_src[3];
    }

    /// Exact for same-type copies; for quantized destinations, an estimate
    /// of one quantized value off by one; 1e-6 otherwise.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        if (self.type_src == self.type_dst) return 0.0;
        const td = self.type_dst.raw();
        if (td == mod.c.GGML_TYPE_Q4_0 or td == mod.c.GGML_TYPE_Q4_1 or td == mod.c.GGML_TYPE_IQ4_NL or
            td == mod.c.GGML_TYPE_Q5_0 or td == mod.c.GGML_TYPE_Q5_1 or td == mod.c.GGML_TYPE_Q8_0)
        {
            // The values are in [-150, 150]: roughly (150 * 2 / 2^bits)^2,
            // divided by the mean square of the reference, about
            // 0.25 * 150^2 times the count.
            var err_estimate: f64 = 1.0 / 8.0 * 150.0;
            // iq4_nl values are a bit more spread out.
            if (td == mod.c.GGML_TYPE_IQ4_NL) err_estimate *= 2.0;
            if (td == mod.c.GGML_TYPE_Q5_0 or td == mod.c.GGML_TYPE_Q5_1) err_estimate /= 2.0;
            if (td == mod.c.GGML_TYPE_Q8_0) err_estimate /= 8.0;
            err_estimate *= err_estimate;
            const denom: f32 = (150.0 * 150.0 * 0.25) * @as(f32, @floatFromInt(self.totalElements()));
            err_estimate /= denom;
            return err_estimate;
        }
        return 1e-6;
    }

    /// The output's bytes plus the source's.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the size.
    pub fn opSize(self: *Self, t: *mod.c.ggml_tensor) usize {
        _ = self;
        return mod.c.ggml_nbytes(t) + mod.c.ggml_nbytes(t.src[0]);
    }

    /// Builds `cpy(src, dst)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var src = g.newTensor(ctx, self.type_src, 4, &self.ne_src);
        mod.Graph.setName(src, "src");

        if (self.src_use_permute) {
            const p = self.permute_src;
            src = mod.c.ggml_permute(ctx, src, @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
            mod.Graph.setName(src, "src_permuted");
        }

        if (self.transpose_src) {
            src = mod.c.ggml_transpose(ctx, src);
            mod.Graph.setName(src, "src_transposed");
        }

        const dst_ne: [4]i64 = if (self.use_dst_shape) self.ne_dst else src.ne;
        var dst: *mod.c.ggml_tensor = undefined;

        if (self.use_dst_alloc) {
            // View a sub-block of a larger buffer: a strided dst.
            const dst_buf = g.newTensor(ctx, self.type_dst, 4, &self.dst_alloc);
            mod.Graph.setName(dst_buf, "dst_buf");
            dst = mod.c.ggml_view_4d(ctx, dst_buf, dst_ne[0], dst_ne[1], dst_ne[2], dst_ne[3], dst_buf.nb[1], dst_buf.nb[2], dst_buf.nb[3], 0);
            mod.Graph.setName(dst, "dst_view");
        } else {
            dst = g.newTensor(ctx, self.type_dst, 4, &dst_ne);
            mod.Graph.setName(dst, "dst");
            if (self.dst_use_permute) {
                const p = self.permute_dst;
                dst = mod.c.ggml_permute(ctx, dst, @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
                mod.Graph.setName(dst, "dst_permuted");
            }
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_cpy(ctx, src, dst);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs over an extended range, to check that f32/i32 casts
    /// agree.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -150.0, 150.0);
    }
};
