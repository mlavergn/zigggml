//! Port of `test_group_norm_mul_add` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GROUP_NORM + GGML_OP_MUL + GGML_OP_ADD.

const std = @import("std");
const mod = @import("module.zig");

/// `group_norm(a) * w + b`, which backends may fuse.
pub const TestGroupNormMulAdd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 128, 1, 1, 1 },
    num_groups: i32 = 4,
    eps: f32 = 1e-5,

    /// Always `GROUP_NORM_MUL_ADD`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("GROUP_NORM_MUL_ADD");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "num_groups", "eps" });
    }

    /// Builds `group_norm(a, num_groups, eps) * w + b`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        const w = g.newTensor(ctx, self.type, 4, &self.ne);
        const b = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        mod.Graph.setName(w, "w");
        mod.Graph.setName(b, "b");
        const n: *mod.c.ggml_tensor = mod.c.ggml_group_norm(ctx, a, self.num_groups, self.eps);
        const m: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, n, w);
        const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, m, b);
        mod.Graph.setName(out, "out");
        return out;
    }
};
