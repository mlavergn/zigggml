//! Port of `test_snake_fuse` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL + GGML_OP_SIN + GGML_OP_SQR + GGML_OP_MUL + GGML_OP_ADD.

const std = @import("std");
const mod = @import("module.zig");

/// The SNAKE activation, `y = x + sin(a*x)^2 * inv_b`, as the naive 5-op
/// chain a backend may fuse into one kernel. Compared as one graph.
pub const TestSnakeFuse = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// [T, C, D2, D3].
    ne: [4]i64 = .{ 256, 192, 1, 1 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Always `SNAKE_FUSE`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("SNAKE_FUSE");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Looser bounds for BF16 and F16, matching the roundoff drift between
    /// the naive CPU chain and a fused kernel.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return switch (self.type.raw()) {
            mod.c.GGML_TYPE_BF16 => 5e-3,
            mod.c.GGML_TYPE_F16 => 5e-5,
            else => 1e-7,
        };
    }

    /// Builds `x + sqr(sin(x * a)) * inv_b`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const x = g.newTensor4d(ctx, self.type, ne[0], ne[1], ne[2], ne[3]);
        mod.Graph.setName(x, "x");

        const a = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, 1, ne[1]);
        mod.Graph.setName(a, "a");

        const inv_b = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, 1, ne[1]);
        mod.Graph.setName(inv_b, "inv_b");

        // The exact 5-op chain that BigVGAN / Vocos frontends emit.
        const ax: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, x, a);
        const sin_ax: *mod.c.ggml_tensor = mod.c.ggml_sin(ctx, ax);
        const sin_sq: *mod.c.ggml_tensor = mod.c.ggml_sqr(ctx, sin_ax);
        const scaled: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, sin_sq, inv_b);
        const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, x, scaled);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills `x` in [-pi, pi], to exercise sin's periodicity, and the
    /// parameters in [-1, 1).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (std.mem.eql(u8, std.mem.span(mod.c.ggml_get_name(t)), "x")) {
                ti.uniform(t, -3.14159, 3.14159);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
