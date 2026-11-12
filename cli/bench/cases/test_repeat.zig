//! Port of `test_repeat` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_REPEAT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_REPEAT of `src` `nr` times along each dimension.
pub const TestRepeat = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },
    nr: [4]i32 = .{ 2, 2, 2, 2 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "nr" });
    }

    /// Twice the output's bytes.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the size.
    pub fn opSize(self: *Self, t: *mod.c.ggml_tensor) usize {
        _ = self;
        return mod.c.ggml_nbytes(t) * 2;
    }

    /// Builds `repeat(src, target)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const nr = self.nr;
        const target = g.newTensor4d(ctx, self.type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
        mod.Graph.setName(target, "target");
        const src = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(src, "src");
        const out: *mod.c.ggml_tensor = mod.c.ggml_repeat(ctx, src, target);
        mod.Graph.setName(out, "out");
        return out;
    }
};
