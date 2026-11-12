//! Port of `test_tri` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_TRI.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_TRI: keeps one triangle of each square matrix.
pub const TestTri = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    tri_type: mod.c.enum_ggml_tri_type = 0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Square matrices: ne[0] == ne[1].
    ne: [4]i64 = .{ 10, 10, 4, 3 },

    /// Checks the matrices are square.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        std.debug.assert(self.ne[0] == self.ne[1]);
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "tri_type" });
    }

    /// Builds `tri(a, tri_type)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor4d(ctx, self.type, self.ne[0], self.ne[1], self.ne[2], self.ne[3]);
        mod.Graph.setName(a, "a");

        const out: *mod.c.ggml_tensor = mod.c.ggml_tri(ctx, a, self.tri_type);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills every tensor uniformly in [-1, 1).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -1.0, 1.0);
    }
};
