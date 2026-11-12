//! Port of `test_rms_norm_back` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_RMS_NORM_BACK.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_RMS_NORM_BACK: the gradient of RMS norm.
pub const TestRmsNormBack = struct {
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

    /// Builds `rms_norm_back(a, b)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        const b = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(b, "b");

        const out: *mod.c.ggml_tensor = mod.c.ggml_rms_norm_back(ctx, a, b, self.eps);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs uniformly in [-10, 10).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -10.0, 10.0);
    }
};
