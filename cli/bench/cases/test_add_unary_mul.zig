//! Port of `test_add_unary_mul` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ADD + GGML_OP_UNARY + GGML_OP_MUL.

const std = @import("std");
const mod = @import("module.zig");

/// `unary(a + bias) * scale` (SILU, SIGMOID or SOFTPLUS) with the bias and
/// scale broadcast over dim 0: the delta-net alpha gate,
/// `softplus(alpha + dt) * a_coeff`. Compared as one fused graph.
pub const TestAddUnaryMul = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// No C++ default.
    op: mod.c.enum_ggml_unary_op = 0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 32, 7, 1, 1 },
    /// The unary result is the second MUL operand.
    swap: bool = false,
    /// Bias/scale layout: `bcast`, `same_shape` or `rep_ne0`.
    layout: []const u8 = "bcast",
    /// Extra consumer past the MUL: empty, `reuse` or `consumer`.
    tail: []const u8 = "",

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "swap", "layout", "tail" });
    }

    /// `ADD_<UNARY>_MUL`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = t;
        try w.print("ADD_{s}_MUL", .{std.mem.span(mod.c.ggml_unary_op_name(self.op))});
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// F16 never fuses, so its bound is the unfused chain's rounding drift;
    /// GELU never fuses either, and backends' exp form drifts from the CPU's
    /// tanhf.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        if (self.type.raw() == mod.c.GGML_TYPE_F16) return 5e-5;
        return if (self.op == mod.c.GGML_UNARY_OP_GELU) 5e-7 else 1e-7;
    }

    /// Builds `unary(a + bias) * scale`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const a = g.newTensor(ctx, self.type, 4, &ne);
        mod.Graph.setName(a, "a");

        var ne_v = [4]i64{ ne[0], 1, 1, 1 };
        if (std.mem.eql(u8, self.layout, "bcast")) {
            // One ne0 row each, broadcast over the outer dims: the alpha-gate form.
        } else if (std.mem.eql(u8, self.layout, "same_shape")) {
            // No broadcast at all; fuses only while the activation is a single row.
            ne_v = ne;
        } else if (std.mem.eql(u8, self.layout, "rep_ne0")) {
            // Repeat on dim 0, which bias[col] cannot address, so this must not fuse.
            ne_v[0] = @divTrunc(ne[0], 4);
        } else {
            std.debug.panic("unknown layout {s}", .{self.layout});
        }

        const bias = g.newTensor(ctx, self.type, 4, &ne_v);
        mod.Graph.setName(bias, "bias");

        const scale = g.newTensor(ctx, self.type, 4, &ne_v);
        mod.Graph.setName(scale, "scale");

        const s: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, a, bias);
        mod.Graph.setName(s, "add");

        const u: *mod.c.ggml_tensor = mod.c.ggml_unary(ctx, s, self.op);
        mod.Graph.setName(u, "unary");

        // A broadcasting operand can only be the second one, so swap needs same-shape operands.
        var out: *mod.c.ggml_tensor = if (self.swap) mod.c.ggml_mul(ctx, scale, u) else mod.c.ggml_mul(ctx, u, scale);

        if (std.mem.eql(u8, self.tail, "reuse")) {
            // A second read of the add result must block the fusion.
            mod.Graph.setName(out, "mul");
            out = mod.c.ggml_add(ctx, out, s);
        } else if (std.mem.eql(u8, self.tail, "consumer")) {
            // Fusion still applies; catches a dispatcher that skips one node too many.
            mod.Graph.setName(out, "mul");
            out = mod.c.ggml_add(ctx, out, scale);
        } else if (self.tail.len != 0) {
            std.debug.panic("unknown tail {s}", .{self.tail});
        }
        mod.Graph.setName(out, "out");
        return out;
    }
};
