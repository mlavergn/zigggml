//! Port of `test_scale` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SCALE.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SCALE with bias, `a * scale + bias`, optionally in place and on a
/// non-contiguous input.
pub const TestScale = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 10, 10 },
    scale: f32 = 2.0,
    bias: f32 = 0.0,
    inplace: bool = false,
    /// View: 1 for a non-contiguous `a`.
    v: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "scale", "bias", "inplace", "v" });
    }

    /// Builds `scale_bias(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var ne_a = self.ne;
        if (self.v & 1 != 0) {
            ne_a[0] *= 3;
            ne_a[1] *= 2;
            ne_a[2] *= 5;
            ne_a[3] *= 4;
        }
        var a = g.newTensor(ctx, self.type, 4, &ne_a);
        mod.Graph.setName(a, "a");

        if (self.v & 1 != 0) {
            a = mod.c.ggml_view_4d(ctx, a, self.ne[0], self.ne[1], self.ne[2], self.ne[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
        }

        const out: *mod.c.ggml_tensor = if (self.inplace)
            mod.c.ggml_scale_bias_inplace(ctx, a, self.scale, self.bias)
        else
            mod.c.ggml_scale_bias(ctx, a, self.scale, self.bias);
        mod.Graph.setName(out, "out");
        return out;
    }
};
