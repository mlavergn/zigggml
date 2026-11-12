//! Port of `test_swiglu_clamp` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GLU.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GLU (SWIGLU with clamp) on two inputs, contiguous or (with
/// `v & 1`) non-contiguous.
pub const TestSwigluClamp = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 128, 2, 2, 2 },
    /// View: 1 for non-contiguous inputs.
    v: i32 = 0,
    limit: f32 = 7.0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "v", "limit" });
    }

    /// A looser bound in F16, for the roundoff of the sigmoid/clamp chain.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return if (self.type.raw() == mod.c.GGML_TYPE_F16) 5e-6 else 1e-7;
    }

    /// Builds `swiglu_clamp(a, b, limit)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a: *mod.c.ggml_tensor = undefined;
        var b: *mod.c.ggml_tensor = undefined;
        if (self.v & 1 != 0) {
            var ne = self.ne_a;
            ne[0] *= 3;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, self.ne_a[0], self.ne_a[1], self.ne_a[2], self.ne_a[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
            b = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(b, "b");
            b = mod.c.ggml_view_4d(ctx, b, self.ne_a[0], self.ne_a[1], self.ne_a[2], self.ne_a[3], b.nb[1], b.nb[2], b.nb[3], 0);
            mod.Graph.setName(b, "view_of_b");
        } else {
            a = g.newTensor(ctx, self.type, 4, &self.ne_a);
            mod.Graph.setName(a, "a");
            b = g.newTensor(ctx, self.type, 4, &self.ne_a);
            mod.Graph.setName(b, "b");
        }
        const out: *mod.c.ggml_tensor = mod.c.ggml_swiglu_clamp(ctx, a, b, self.limit);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs over an extended range.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -150.0, 150.0);
    }
};
