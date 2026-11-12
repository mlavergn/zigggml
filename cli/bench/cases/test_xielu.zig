//! Port of `test_xielu` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_UNARY (XIELU).

const std = @import("std");
const mod = @import("module.zig");

/// xIELU activation with fixed alpha_n, alpha_p, beta and eps.
pub const TestXielu = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Builds `xielu(a, 4, 20, 0.5, 1e-7)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor4d(ctx, self.type, self.ne[0], self.ne[1], self.ne[2], self.ne[3]);
        mod.Graph.setName(a, "a");

        const alpha_n: f32 = 4.0;
        const alpha_p: f32 = 20.0;
        const beta: f32 = 0.5;
        const eps: f32 = 0.0000001;

        const out: *mod.c.ggml_tensor = mod.c.ggml_xielu(ctx, a, alpha_n, alpha_p, beta, eps);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills every tensor uniformly in [-1, 1).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -1.0, 1.0);
    }
};
