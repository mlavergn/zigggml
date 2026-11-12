//! Port of `test_pad` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_PAD.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_PAD at the end of dims 0 and 1, with zeros or circularly.
pub const TestPad = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 512, 512, 1, 1 },
    pad_0: i32 = 1,
    pad_1: i32 = 1,
    circular: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "pad_0", "pad_1", "circular" });
    }

    /// Builds `pad(a, pad_0, pad_1, 0, 0)` or its circular form.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = if (self.circular)
            mod.c.ggml_pad_circular(ctx, a, self.pad_0, self.pad_1, 0, 0)
        else
            mod.c.ggml_pad(ctx, a, self.pad_0, self.pad_1, 0, 0);
        mod.Graph.setName(out, "out");
        return out;
    }
};
