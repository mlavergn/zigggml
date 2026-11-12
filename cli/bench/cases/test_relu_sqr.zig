//! Port of `test_relu_sqr` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_UNARY(RELU) + GGML_OP_SQR.

const std = @import("std");
const mod = @import("module.zig");

/// `sqr(relu(a))`, compared as one fused graph.
pub const TestReluSqr = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 128, 2, 2, 2 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Always `RELU_SQR`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("RELU_SQR");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `sqr(relu(a))`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        const r: *mod.c.ggml_tensor = mod.c.ggml_relu(ctx, a);
        mod.Graph.setName(r, "relu");

        const out: *mod.c.ggml_tensor = mod.c.ggml_sqr(ctx, r);
        mod.Graph.setName(out, "out");
        return out;
    }
};
