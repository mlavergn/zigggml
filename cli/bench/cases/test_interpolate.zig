//! Port of `test_interpolate` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_UPSCALE.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_UPSCALE through `ggml_interpolate` to a target shape, of an
/// optionally permuted input.
pub const TestInterpolate = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 2, 5, 7, 11 },
    ne_tgt: [4]i64 = .{ 5, 7, 11, 13 },
    mode: mod.ScaleMode = .of(mod.c.GGML_SCALE_MODE_NEAREST),
    permute: [4]i64 = .{ 0, 0, 0, 0 },

    // State.
    /// Whether `permute` is not all zero.
    use_permute: bool = false,

    /// Derives `use_permute` from `permute`.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.use_permute = self.permute[0] + self.permute[1] + self.permute[2] + self.permute[3] > 0;
    }

    /// Writes the case's parameters; `permute` only when used.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type", self.type);
        try v.add("ne", self.ne);
        try v.add("ne_tgt", self.ne_tgt);
        try v.add("mode", self.mode);
        if (self.use_permute) try v.add("permute", self.permute);
    }

    /// Builds `interpolate(a, ne_tgt, mode)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");

        if (self.use_permute) {
            const p = self.permute;
            a = mod.c.ggml_permute(ctx, a, @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
            mod.Graph.setName(a, "a_permuted");
        }

        const t = self.ne_tgt;
        const out: *mod.c.ggml_tensor = mod.c.ggml_interpolate(ctx, a, t[0], t[1], t[2], t[3], self.mode.raw());
        mod.Graph.setName(out, "out");
        return out;
    }
};
