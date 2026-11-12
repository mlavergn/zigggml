//! Port of `test_timestep_embedding` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_TIMESTEP_EMBEDDING.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_TIMESTEP_EMBEDDING of `dim` dimensions.
pub const TestTimestepEmbedding = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 2, 1, 1, 1 },
    dim: i32 = 320,
    max_period: i32 = 10000,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "dim", "max_period" });
    }

    /// Builds `timestep_embedding(a, dim, max_period)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_timestep_embedding(ctx, a, self.dim, self.max_period);
        mod.Graph.setName(out, "out");
        return out;
    }
};
