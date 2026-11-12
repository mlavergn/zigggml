//! Port of `test_solve_tri` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SOLVE_TRI.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SOLVE_TRI: solves a lower-triangular system a·x = b from the left.
pub const TestSolveTri = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_lhs: [4]i64 = .{ 10, 10, 4, 3 },
    ne_rhs: [4]i64 = .{ 3, 10, 4, 3 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_lhs", "ne_rhs" });
    }

    /// n·(n+1)/2 non-zero elements of lhs, 2 FLOPs each, per column of rhs
    /// and per batch.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        const n = self.ne_lhs[0];
        const k = self.ne_rhs[0];
        const batch = self.ne_lhs[2] * self.ne_lhs[3];
        return @intCast(n * (n + 1) * k * batch);
    }

    /// Builds `solve_tri(a, b, left=true, lower=true, uni=false)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const l = self.ne_lhs;
        const r = self.ne_rhs;
        const a = g.newTensor4d(ctx, self.type, l[0], l[1], l[2], l[3]);
        mod.Graph.setName(a, "a");

        const b = g.newTensor4d(ctx, self.type, r[0], r[1], r[2], r[3]);
        mod.Graph.setName(b, "b");

        const out: *mod.c.ggml_tensor = mod.c.ggml_solve_tri(ctx, a, b, true, true, false);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills `a` with a lower triangle in [0.1, 1), avoiding zeros on the
    /// diagonal, and the rest uniformly in [-1, 1).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (std.mem.eql(u8, std.mem.sliceTo(&t.name, 0), "a")) {
                ti.tril(t, 0.1, 1.0);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
