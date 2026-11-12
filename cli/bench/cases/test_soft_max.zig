//! Port of `test_soft_max` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SOFT_MAX.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SOFT_MAX with an optional mask, sinks, scale and ALiBi bias,
/// broadcasting the mask over dims 2 and 3.
pub const TestSoftMax = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },
    mask: bool = false,
    sinks: bool = false,
    m_prec: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Broadcast only dims 2 and 3.
    nr23: [2]i64 = .{ 1, 1 },
    scale: f32 = 1.0,
    max_bias: f32 = 0.0,
    inplace: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "mask", "sinks", "m_prec", "nr23", "scale", "max_bias", "inplace" });
    }

    /// The 1024-wide test with bias occasionally fails the default bound.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 1e-6;
    }

    /// Builds `soft_max_ext(a, mask, scale, max_bias)` with optional sinks.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const a = g.newTensor4d(ctx, self.type, ne[0], ne[1], ne[2] * self.nr23[0], ne[3] * self.nr23[1]);
        mod.Graph.setName(a, "a");

        var mask: ?*mod.c.ggml_tensor = null;
        if (self.mask) {
            const m = g.newTensor4d(ctx, self.m_prec, ne[0], ne[1], ne[2], ne[3]);
            mod.Graph.setName(m, "mask");
            mask = m;
        }

        var sinks: ?*mod.c.ggml_tensor = null;
        if (self.sinks) {
            const s = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, ne[2] * self.nr23[0]);
            mod.Graph.setName(s, "sinks");
            sinks = s;
        }

        const out: *mod.c.ggml_tensor = if (self.inplace)
            mod.c.ggml_soft_max_ext_inplace(ctx, a, mask, self.scale, self.max_bias)
        else
            mod.c.ggml_soft_max_ext(ctx, a, mask, self.scale, self.max_bias);
        mod.c.ggml_soft_max_add_sinks(out, sinks);
        mod.Graph.setName(out, "out");
        return out;
    }
};
