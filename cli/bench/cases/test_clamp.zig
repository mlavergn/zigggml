//! Port of `test_clamp` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CLAMP.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CLAMP on a contiguous or (with `v & 1`) non-contiguous input.
pub const TestClamp = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },
    min: f32 = -0.5,
    max: f32 = 0.5,
    /// View: 1 for a non-contiguous `a`.
    v: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "min", "max", "v" });
    }

    /// Builds `clamp(a, min, max)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a: *mod.c.ggml_tensor = undefined;
        if (self.v & 1 != 0) {
            var ne_a = self.ne;
            ne_a[0] *= 3;
            ne_a[1] *= 2;
            ne_a[2] *= 5;
            ne_a[3] *= 4;
            a = g.newTensor(ctx, self.type, 4, &ne_a);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, self.ne[0], self.ne[1], self.ne[2], self.ne[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
        } else {
            a = g.newTensor(ctx, self.type, 4, &self.ne);
            mod.Graph.setName(a, "a");
        }
        const out: *mod.c.ggml_tensor = mod.c.ggml_clamp(ctx, a, self.min, self.max);
        mod.Graph.setName(out, "out");
        return out;
    }
};
