//! Port of `test_upscale` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_UPSCALE.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_UPSCALE by an integer factor, of an optionally transposed input.
pub const TestUpscale = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 512, 512, 3, 1 },
    scale_factor: i32 = 2,
    mode: mod.ScaleMode = .of(mod.c.GGML_SCALE_MODE_NEAREST),
    transpose: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "scale_factor", "mode", "transpose" });
    }

    /// Builds `upscale(a, scale_factor, mode)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        if (self.transpose) {
            a = mod.c.ggml_transpose(ctx, a);
            mod.Graph.setName(a, "a_transposed");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_upscale(ctx, a, self.scale_factor, @intCast(self.mode.raw()));
        mod.Graph.setName(out, "out");
        return out;
    }
};
