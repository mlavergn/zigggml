//! Port of `test_arange` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ARANGE.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ARANGE from `start` to `stop` by `step`.
pub const TestArange = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    start: f32 = 0.0,
    stop: f32 = 10.0,
    step: f32 = 1.0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "start", "stop", "step" });
    }

    /// Builds `arange(start, stop, step)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const out: *mod.c.ggml_tensor = mod.c.ggml_arange(g.ctx, self.start, self.stop, self.step);
        mod.Graph.setName(out, "out");
        return out;
    }
};
