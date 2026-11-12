//! Port of `test_norm_scale` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_NORM/GGML_OP_RMS_NORM + GGML_OP_SCALE.

const std = @import("std");
const mod = @import("module.zig");

/// `norm(a) * scale` or `rms_norm(a) * scale`, compared as one fused graph.
pub const TestNormScale = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    eps: f32 = 1e-6,
    /// RMS_NORM instead of NORM.
    rms: bool = false,
    scale: f32 = 1.5,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "eps", "rms", "scale" });
    }

    /// `RMS_NORM_SCALE` or `NORM_SCALE`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = t;
        try w.writeAll(if (self.rms) "RMS_NORM_SCALE" else "NORM_SCALE");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `scale(norm(a))`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        const n: *mod.c.ggml_tensor = if (self.rms) mod.c.ggml_rms_norm(ctx, a, self.eps) else mod.c.ggml_norm(ctx, a, self.eps);
        const out: *mod.c.ggml_tensor = mod.c.ggml_scale(ctx, n, self.scale);
        mod.Graph.setName(out, "out");
        return out;
    }
};
