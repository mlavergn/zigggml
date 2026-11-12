//! Port of `test_group_norm` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GROUP_NORM.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GROUP_NORM over `num_groups` groups.
pub const TestGroupNorm = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 64, 320, 1 },
    num_groups: i32 = 32,
    eps: f32 = 1e-6,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "num_groups", "eps" });
    }

    /// Builds `group_norm(a, num_groups, eps)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_group_norm(ctx, a, self.num_groups, self.eps);
        mod.Graph.setName(out, "out");
        return out;
    }
};
