//! Port of `test_dsv4_hc_pre` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_DSV4_HC_PRE.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_DSV4_HC_PRE: the DSV4 hyper-connection reduction, weighted or
/// (with `gated`) gated per element.
pub const TestDsv4HcPre = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    n_embd: i64 = 31,
    n_hc: i64 = 4,
    n_tokens: i64 = 17,
    gated: bool = false,

    // State.
    /// The `test_dsv4_hc` base.
    base: mod.TestDsv4Hc = .{},

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "n_embd", "n_hc", "n_tokens", "gated" });
    }

    /// Always `DSV4_HC_PRE`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("DSV4_HC_PRE");
    }

    /// Builds `dsv4_hc_pre(x, weights)` or `dsv4_hc_pre_gated(x, gate, 1/n_hc)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const x = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, self.n_embd, self.n_hc, self.n_tokens);
        mod.Graph.setName(x, "x");

        var out: *mod.c.ggml_tensor = undefined;
        if (self.gated) {
            const gate = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, self.n_embd, self.n_hc, self.n_tokens);
            mod.Graph.setName(gate, "gate");
            out = mod.c.ggml_dsv4_hc_pre_gated(ctx, x, gate, 1.0 / @as(f32, @floatFromInt(self.n_hc)));
        } else {
            const weights = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, self.n_hc, self.n_tokens);
            mod.Graph.setName(weights, "weights");
            out = mod.c.ggml_dsv4_hc_pre(ctx, x, weights);
        }
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
