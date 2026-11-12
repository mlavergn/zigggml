//! Port of `test_unary_mul` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_UNARY + GGML_OP_MUL.

const std = @import("std");
const mod = @import("module.zig");

/// `unary(a) * b` (GELU, SILU, SIGMOID or SOFTPLUS), over several operand
/// layouts and with an optional extra consumer, compared as one fused graph.
pub const TestUnaryMul = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// No C++ default.
    op: mod.c.enum_ggml_unary_op = 0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 128, 2, 2, 2 },
    /// The unary result is the second MUL operand.
    swap: bool = false,
    /// Operand layout: `packed`, `pad_unary`, `pad_other`, `halves`,
    /// `strided_dim1`, `bcast`, `rep_ne0`, `view_mid` or `gate`.
    layout: []const u8 = "packed",
    /// Extra consumer past the MUL: empty, `reuse` or `consumer`.
    tail: []const u8 = "",

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "swap", "layout", "tail" });
    }

    /// `<UNARY>_MUL`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = t;
        try w.print("{s}_MUL", .{std.mem.span(mod.c.ggml_unary_op_name(self.op))});
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The fused kernel elides the rounding of the unary result that the
    /// CPU chain performs; GELU shaders use the exp form where the CPU uses
    /// tanhf.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return switch (self.type.raw()) {
            mod.c.GGML_TYPE_F16 => 5e-5,
            mod.c.GGML_TYPE_BF16 => 1e-4,
            else => if (self.op == mod.c.GGML_UNARY_OP_GELU) 5e-7 else 1e-7,
        };
    }

    /// Builds `unary(a) * b` in the configured layout.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const layout = self.layout;
        // a: the unary source; b: the other MUL operand.
        var a: *mod.c.ggml_tensor = undefined;
        var b: ?*mod.c.ggml_tensor = null;

        if (std.mem.eql(u8, layout, "packed")) {
            a = g.newTensor(ctx, self.type, 4, &ne);
            b = g.newTensor(ctx, self.type, 4, &ne);
        } else if (std.mem.eql(u8, layout, "pad_unary")) {
            a = self.padded(g, "a", 3, 0);
            b = g.newTensor(ctx, self.type, 4, &ne);
        } else if (std.mem.eql(u8, layout, "pad_other")) {
            a = g.newTensor(ctx, self.type, 4, &ne);
            b = self.padded(g, "b", 3, 0);
        } else if (std.mem.eql(u8, layout, "halves")) {
            // The shape the Conformer audio encoders build: one tensor split in two.
            var ne_w = ne;
            ne_w[0] *= 2;
            const base = g.newTensor(ctx, self.type, 4, &ne_w);
            mod.Graph.setName(base, "base");
            b = mod.c.ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base.nb[1], base.nb[2], base.nb[3], 0);
            a = mod.c.ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base.nb[1], base.nb[2], base.nb[3], @as(usize, @intCast(ne[0])) * base.nb[0]);
        } else if (std.mem.eql(u8, layout, "strided_dim1")) {
            // Contiguous rows but a strided dim 1: not ggml_is_contiguous_1, must not fuse.
            var ne_w = ne;
            ne_w[1] *= 3;
            const base = g.newTensor(ctx, self.type, 4, &ne_w);
            mod.Graph.setName(base, "a");
            a = mod.c.ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base.nb[1], base.nb[2], base.nb[3], 0);
            b = g.newTensor(ctx, self.type, 4, &ne);
        } else if (std.mem.eql(u8, layout, "bcast")) {
            a = g.newTensor(ctx, self.type, 4, &ne);
            b = g.newTensor4d(ctx, self.type, ne[0], 1, 1, 1);
        } else if (std.mem.eql(u8, layout, "rep_ne0")) {
            // Repeat on dim 0.
            a = g.newTensor(ctx, self.type, 4, &ne);
            var ne_b = ne;
            ne_b[0] = @divTrunc(ne_b[0], 4);
            b = g.newTensor(ctx, self.type, 4, &ne_b);
        } else if (std.mem.eql(u8, layout, "view_mid")) {
            // VIEW between UNARY and MUL.
            a = g.newTensor(ctx, self.type, 4, &ne);
            b = null;
        } else if (std.mem.eql(u8, layout, "gate")) {
            // Small gate on src1.
            const ne_gate = [4]i64{ 1, ne[1], ne[2], ne[3] };
            a = g.newTensor(ctx, self.type, 4, &ne_gate);
            b = g.newTensor(ctx, self.type, 4, &ne);
        } else {
            std.debug.panic("unknown layout {s}", .{layout});
        }
        mod.Graph.setName(a, "a");
        if (b) |bt| mod.Graph.setName(bt, "b");

        const u: *mod.c.ggml_tensor = mod.c.ggml_unary(ctx, a, self.op);
        mod.Graph.setName(u, "unary");

        // A broadcasting operand can only be the second one.
        const second = std.mem.eql(u8, layout, "gate") or
            (self.swap and !std.mem.eql(u8, layout, "bcast") and !std.mem.eql(u8, layout, "view_mid"));
        if (std.mem.eql(u8, layout, "view_mid")) {
            var ne_base = ne;
            ne_base[0] *= 2;
            const base = g.newTensor(ctx, self.type, 4, &ne_base);
            mod.Graph.setName(base, "base");
            const bt: *mod.c.ggml_tensor = mod.c.ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base.nb[1], base.nb[2], base.nb[3], 0);
            mod.Graph.setName(bt, "b");
            b = bt;
        }
        const other = b.?;
        var out: *mod.c.ggml_tensor = if (second) mod.c.ggml_mul(ctx, other, u) else mod.c.ggml_mul(ctx, u, other);

        if (std.mem.eql(u8, self.tail, "reuse")) {
            // A second read of the unary result must block the fusion.
            mod.Graph.setName(out, "mul");
            out = mod.c.ggml_add(ctx, out, u);
        } else if (std.mem.eql(u8, self.tail, "consumer")) {
            // Fusion still applies; catches a dispatcher that skips one node too many.
            mod.Graph.setName(out, "mul");
            out = mod.c.ggml_add(ctx, out, other);
        } else if (self.tail.len != 0) {
            std.debug.panic("unknown tail {s}", .{self.tail});
        }
        mod.Graph.setName(out, "out");
        return out;
    }

    /// `ne` viewed out of a wider tensor: rows stay contiguous, but the
    /// stride exceeds the width.
    fn padded(self: *Self, g: *mod.Graph, name: [*:0]const u8, mul0: i64, off0: i64) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        var ne_w = ne;
        ne_w[0] *= mul0;
        const base = g.newTensor(ctx, self.type, 4, &ne_w);
        mod.Graph.setName(base, name);
        return mod.c.ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base.nb[1], base.nb[2], base.nb[3], @as(usize, @intCast(off0)) * base.nb[0]);
    }
};
