//! Port of `test_repeat_back` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_REPEAT_BACK.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_REPEAT_BACK, summing `nr` repeats per dimension, from an
/// optionally non-contiguous source.
pub const TestRepeatBack = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 8, 6, 4, 2 },
    nr: [4]i32 = .{ 2, 2, 2, 2 },
    /// Whether src is a non-contiguous view.
    v: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "nr", "v" });
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

    /// Builds `repeat_back(src, target)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const nr = self.nr;
        var src = g.newTensor4d(ctx, self.type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2], ne[3] * nr[3]);
        mod.Graph.setName(src, "src");

        if (self.v) {
            for (0..4) |i| {
                std.debug.assert(@mod(ne[i], 2) == 0);
                std.debug.assert(@mod(nr[i], 2) == 0 or nr[i] == 1);
            }
            const ne00 = if (nr[0] == 1) src.ne[0] else @divTrunc(src.ne[0], 2);
            const ne01 = if (nr[1] == 1) src.ne[1] else @divTrunc(src.ne[1], 2);
            const ne02 = if (nr[2] == 1) src.ne[2] else @divTrunc(src.ne[2], 2);
            const ne03 = if (nr[3] == 1) src.ne[3] else @divTrunc(src.ne[3], 2);
            src = mod.c.ggml_view_4d(ctx, src, ne00, ne01, ne02, ne03, src.nb[1], src.nb[2], src.nb[3], 0);
        }

        const target = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(target, "target");

        const out: *mod.c.ggml_tensor = mod.c.ggml_repeat_back(ctx, src, target);
        mod.Graph.setName(out, "out");
        return out;
    }
};
