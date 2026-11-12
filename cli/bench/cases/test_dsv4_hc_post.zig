//! Port of `test_dsv4_hc_post` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_DSV4_HC_POST.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_DSV4_HC_POST: the DSV4 hyper-connection expansion, with or
/// without a combination matrix, and optionally with the gated `post` chain.
pub const TestDsv4HcPost = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    n_embd: i64 = 31,
    n_tokens: i64 = 17,
    /// No combination matrix.
    identity: bool = false,
    /// `post = 2*sigmoid(post/hc)`, as qwen4exp builds it.
    gated: bool = false,

    // State.
    /// The `test_dsv4_hc` base.
    base: mod.TestDsv4Hc = .{},

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "n_embd", "n_tokens", "identity", "gated" });
    }

    /// Always `DSV4_HC_POST`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("DSV4_HC_POST");
    }

    /// The gated chain is compared as one graph, so backends can fuse it.
    ///
    /// - `self`: the case.
    /// - Returns: `gated`.
    pub fn runWholeGraph(self: *Self) bool {
        return self.gated;
    }

    /// Builds `dsv4_hc_post(x, residual, post, comb)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const hc = mod.TestDsv4Hc.hc;
        const x = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, self.n_embd, self.n_tokens);
        mod.Graph.setName(x, "x");

        const residual = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, self.n_embd, hc, self.n_tokens);
        mod.Graph.setName(residual, "residual");

        var post = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, hc, self.n_tokens);
        mod.Graph.setName(post, "post");

        if (self.gated) {
            post = mod.c.ggml_scale(ctx, mod.c.ggml_sigmoid(ctx, mod.c.ggml_scale(ctx, post, 1.0 / @as(f32, @floatFromInt(hc)))), 2.0);
        }

        var comb: ?*mod.c.ggml_tensor = null;
        if (!self.identity) {
            const ct = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, hc, hc, self.n_tokens);
            mod.Graph.setName(ct, "comb");
            comb = ct;
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_dsv4_hc_post(ctx, x, residual, post, comb);
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
