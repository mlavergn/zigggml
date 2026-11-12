//! Port of `test_softcap` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE.

const std = @import("std");
const mod = @import("module.zig");

/// Softcapping, `softcap * tanh(a / softcap)`, compared as one fused graph.
pub const TestSoftcap = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 10, 10 },
    softcap: f32 = 30.0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "softcap" });
    }

    /// Always `SOFTCAP`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("SOFTCAP");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `scale(tanh(scale(a, 1/softcap)), softcap)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        const out: *mod.c.ggml_tensor = mod.c.ggml_scale(ctx, mod.c.ggml_tanh(ctx, mod.c.ggml_scale(ctx, a, 1.0 / self.softcap)), self.softcap);
        mod.Graph.setName(out, "out");
        return out;
    }
};
