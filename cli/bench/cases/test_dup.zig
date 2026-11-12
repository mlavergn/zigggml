//! Port of `test_dup` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_DUP.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_DUP of an optionally permuted source.
pub const TestDup = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 20, 1 },
    /// Permutation of the source; all zeros for none.
    permute: [4]i64 = .{ 0, 0, 0, 0 },

    // State.
    /// Whether `permute` is set (upstream's `_use_permute`).
    use_permute: bool = false,

    /// Derives `use_permute` from `permute`.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.use_permute = self.permute[0] + self.permute[1] + self.permute[2] + self.permute[3] > 0;
    }

    /// Writes the case's parameters; `permute` only when set.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type", self.type);
        try v.add("ne", self.ne);
        if (self.use_permute) try v.add("permute", self.permute);
    }

    /// Builds `dup(src)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var src = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(src, "src");
        if (self.use_permute) {
            const p = self.permute;
            src = mod.c.ggml_permute(ctx, src, @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
            mod.Graph.setName(src, "src_permuted");
        }
        const out: *mod.c.ggml_tensor = mod.c.ggml_dup(ctx, src);
        mod.Graph.setName(out, "out");
        return out;
    }
};
