//! Port of `test_concat` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONCAT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CONCAT of `a` and `b` along `dim`, with optional non-contiguous
/// or transposed inputs.
pub const TestConcat = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 10, 5, 5, 5 },
    ne_b_d: i64 = 5,
    dim: i32 = 2,
    /// View: 1 non-cont a (first 3 dims), 2 non-cont b (first 3 dims),
    /// 4 non-cont a (last 2 dims), 8 non-cont b (last 2 dims), 16 transposed b.
    v: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "ne_b_d", "dim", "v" });
    }

    /// Builds `concat(a, b, dim)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var ne_b = self.ne_a;
        ne_b[@intCast(self.dim)] = self.ne_b_d;
        const ne_a = self.ne_a;

        var a: *mod.c.ggml_tensor = undefined;
        if (self.v & 1 != 0) {
            var ne = ne_a;
            ne[0] *= 2;
            ne[1] *= 4;
            ne[2] *= 3;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
        } else if (self.v & 4 != 0) {
            var ne = ne_a;
            ne[2] *= 2;
            ne[3] *= 4;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
        } else {
            a = g.newTensor(ctx, self.type, 4, &ne_a);
            mod.Graph.setName(a, "a");
        }

        var b: *mod.c.ggml_tensor = undefined;
        if (self.v & 2 != 0) {
            var ne = ne_b;
            ne[0] *= 3;
            ne[1] *= 2;
            ne[2] *= 4;
            b = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(b, "b");
            b = mod.c.ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b.nb[1], b.nb[2], b.nb[3], 0);
            mod.Graph.setName(b, "view_of_b");
        } else if (self.v & 8 != 0) {
            var ne = ne_b;
            ne[2] *= 3;
            ne[3] *= 2;
            b = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(b, "b");
            b = mod.c.ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b.nb[1], b.nb[2], b.nb[3], 0);
            mod.Graph.setName(b, "view_of_b");
        } else if (self.v & 16 != 0) {
            var ne = ne_b;
            std.mem.swap(i64, &ne[0], &ne[1]);
            b = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(b, "b");
            b = mod.c.ggml_transpose(ctx, b);
            mod.Graph.setName(b, "transpose_of_b");
        } else {
            b = g.newTensor(ctx, self.type, 4, &ne_b);
            mod.Graph.setName(b, "b");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_concat(ctx, a, b, self.dim);
        mod.Graph.setName(out, "out");
        return out;
    }
};
