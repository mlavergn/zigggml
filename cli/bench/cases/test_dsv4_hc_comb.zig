//! Port of `test_dsv4_hc_comb` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_DSV4_HC_COMB.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_DSV4_HC_COMB: the DSV4 hyper-connection combination weights.
pub const TestDsv4HcComb = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    n_tokens: i64 = 17,
    n_iter: i32 = 4,
    eps: f32 = 1e-6,

    // State.
    /// The `test_dsv4_hc` base.
    base: mod.TestDsv4Hc = .{},

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "n_tokens", "n_iter", "eps" });
    }

    /// Always `DSV4_HC_COMB`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("DSV4_HC_COMB");
    }

    /// Builds `dsv4_hc_comb(mixes, scale, base)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const hc = mod.TestDsv4Hc.hc;
        const mixes = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, (2 + hc) * hc, self.n_tokens);
        mod.Graph.setName(mixes, "mixes");

        const scale = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, 3);
        mod.Graph.setName(scale, "scale");

        const base = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, (2 + hc) * hc);
        mod.Graph.setName(base, "base");

        const out: *mod.c.ggml_tensor = mod.c.ggml_dsv4_hc_comb(ctx, mixes, scale, base, self.eps, self.n_iter);
        mod.Graph.setName(out, "out");
        self.base.out = out;
        return out;
    }

    /// Fills the inputs as the base does.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.base.initializeTensors(ti, ctx);
    }
};
