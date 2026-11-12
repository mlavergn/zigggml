//! Port of `test_rms_norm` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_RMS_NORM.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_RMS_NORM on a contiguous or viewed input, optionally in place.
pub const TestRmsNorm = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    /// Whether `a` is a non-contiguous view.
    v: bool = false,
    eps: f32 = 1e-6,
    /// Whether the operation runs in place.
    inplace: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "v", "eps", "inplace" });
    }

    /// Builds `rms_norm(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        if (self.v) {
            a = mod.c.ggml_view_4d(ctx, a, @divTrunc(a.ne[0], 2), @divTrunc(a.ne[1], 2), @divTrunc(a.ne[2], 2), @divTrunc(a.ne[3], 2), a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view of a");
        }

        const out: *mod.c.ggml_tensor = if (self.inplace)
            mod.c.ggml_rms_norm_inplace(ctx, a, self.eps)
        else
            mod.c.ggml_rms_norm(ctx, a, self.eps);
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
