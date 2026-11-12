//! Port of `test_bin_bcast` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ADD, GGML_OP_SUB, GGML_OP_MUL, GGML_OP_DIV.

const std = @import("std");
const mod = @import("module.zig");

/// A broadcasting binary op: a pointer to `ggml_add`, `ggml_sub`,
/// `ggml_mul` or `ggml_div` (upstream's `op_t`).
pub const BinOp = @TypeOf(&mod.c.ggml_add);

/// A broadcasting binary op of `a` (`ne * nr`) and `b` (`ne`), applied `nf`
/// times in a row (fused), with `b` optionally permuted, overlapping `a`, of
/// another type, or in the weight context.
pub const TestBinBcast = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// The op; upstream has no default.
    op: BinOp = &mod.c.ggml_add,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 1, 1 },
    nr: [4]i32 = .{ 1, 2, 1, 1 },
    /// Number of fused ops; 1 for a single op (no fusion).
    nf: i32 = 1,
    /// Permute src1.
    perm1: bool = false,
    /// src0 and src1 are overlapping views of the same buffer.
    src_overlap: bool = false,
    /// Type of src1; `GGML_TYPE_COUNT` for the same as src0.
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_COUNT),
    /// Build src1 in the weight context.
    weights: bool = false,

    /// Writes the case's parameters; `type_b` and `weights` only when set.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "nr", "nf", "perm1", "src_overlap" });
        var v = mod.Vars.init(w);
        v.first = false;
        if (self.type_b.raw() != mod.c.GGML_TYPE_COUNT) try v.add("type_b", self.type_b);
        if (self.weights) try v.add("weights", true);
    }

    /// Fused ops are compared as one graph.
    ///
    /// - `self`: the case.
    /// - Returns: true when `nf > 1`.
    pub fn runWholeGraph(self: *Self) bool {
        return self.nf > 1;
    }

    /// Three times the output's bytes.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the size.
    pub fn opSize(self: *Self, t: *mod.c.ggml_tensor) usize {
        _ = self;
        return mod.c.ggml_nbytes(t) * 3;
    }

    /// Whether src1 lives in the weight context.
    ///
    /// - `self`: the case.
    /// - Returns: `weights`.
    pub fn useWeightContext(self: *Self) bool {
        return self.weights;
    }

    /// Fused ADDs can keep FP32 intermediates where the CPU rounds each one
    /// to FP16/BF16.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        if (self.op == &mod.c.ggml_add and self.nf > 1) {
            if (self.type.raw() == mod.c.GGML_TYPE_F16) return 1e-6;
            if (self.type.raw() == mod.c.GGML_TYPE_BF16) return 1e-4;
        }
        return mod.Compare.default_max_nmse_err;
    }

    /// Builds `op(...op(a, b0)..., b[nf-1])`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        std.debug.assert(self.nf <= 16);
        const ne = self.ne;
        const nr = self.nr;

        const a = g.newTensor4d(ctx, self.type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
        mod.Graph.setName(a, "a");

        const tb = if (self.type_b.raw() == mod.c.GGML_TYPE_COUNT) self.type else self.type_b;
        const ctx_b = if (self.weights) g.ctx_weights.? else ctx;

        const nf: usize = @intCast(self.nf);
        var b: [16]*mod.c.ggml_tensor = undefined;
        for (0..nf) |i| {
            if (self.perm1) {
                // Hardcoded for now.
                const p = [4]usize{ 1, 2, 0, 3 };
                b[i] = g.newTensor4d(ctx_b, tb, ne[p[0]], ne[p[1]], ne[p[2]], ne[p[3]]);
                b[i] = mod.c.ggml_permute(ctx_b, b[i], @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
            } else if (self.src_overlap) {
                const third = @divTrunc(ne[3], 3);
                b[i] = mod.c.ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], 2 * third, a.nb[1], a.nb[2], a.nb[3], @as(usize, @intCast(third)) * a.nb[3]);
            } else {
                b[i] = g.newTensor(ctx_b, tb, 4, &self.ne);
            }
            var buf: [16]u8 = undefined;
            const name = std.fmt.bufPrintSentinel(&buf, "b{d}", .{i}, 0) catch @panic("tensor name too long");
            mod.Graph.setName(b[i], name.ptr);
        }

        var out: *mod.c.ggml_tensor = a;
        if (self.src_overlap) {
            out = mod.c.ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], 2 * @divTrunc(ne[3], 3), a.nb[1], a.nb[2], a.nb[3], 0);
        }

        for (b[0..nf]) |bi| {
            out = self.op(ctx, out, bi);
        }

        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs in [0.9, 1.1) for MUL and DIV, which have numerical
    /// issues around zero; uniformly in [-1, 1) otherwise.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        if (self.op == &mod.c.ggml_mul or self.op == &mod.c.ggml_div) {
            ti.allRange(ctx, 0.9, 1.1);
        } else {
            ti.all(ctx);
        }
    }
};
