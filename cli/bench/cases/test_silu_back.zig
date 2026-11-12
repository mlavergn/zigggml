//! Port of `test_silu_back` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SILU_BACK.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SILU_BACK: the gradient of SiLU at `a`, given `grad`.
pub const TestSiluBack = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    eps: f32 = 1e-6,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "eps" });
    }

    /// Builds `silu_back(a, grad)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        const grad = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(grad, "grad");

        const out: *mod.c.ggml_tensor = mod.c.ggml_silu_back(ctx, a, grad);
        mod.Graph.setName(out, "out");
        return out;
    }
};
