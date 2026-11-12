//! Port of `test_pool1d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_POOL_1D.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_POOL_1D, average or max.
pub const TestPool1d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    pool_type: mod.OpPool = .of(mod.c.GGML_OP_POOL_AVG),
    type_input: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_input: [4]i64 = .{ 10, 1, 1, 1 },
    /// Kernel size.
    k0: i32 = 3,
    /// Stride.
    s0: i32 = 3,
    /// Padding.
    p0: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "pool_type", "type_input", "ne_input", "k0", "s0", "p0" });
    }

    /// Builds `pool_1d(input)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const input = g.newTensor(ctx, self.type_input, 4, &self.ne_input);
        mod.Graph.setName(input, "input");
        const out: *mod.c.ggml_tensor = mod.c.ggml_pool_1d(ctx, input, self.pool_type.raw(), self.k0, self.s0, self.p0);
        mod.Graph.setName(out, "out");
        return out;
    }
};
