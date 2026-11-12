//! Port of `test_pad_reflect_1d` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_PAD_REFLECT_1D.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_PAD_REFLECT_1D of a 2-D input (the first two of `ne_a`).
pub const TestPadReflect1d = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 512, 34, 2, 1 },
    pad_0: i32 = 10,
    pad_1: i32 = 9,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "pad_0", "pad_1" });
    }

    /// Builds `pad_reflect_1d(a, pad_0, pad_1)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 2, &self.ne_a);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_pad_reflect_1d(ctx, a, self.pad_0, self.pad_1);
        mod.Graph.setName(out, "out");
        return out;
    }
};
