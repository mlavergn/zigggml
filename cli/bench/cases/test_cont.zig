//! Port of `test_cont` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CONT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CONT of a permuted, view-sliced or transposed source. A zero
/// `permute` means no permutation.
pub const TestCont = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 10, 1 },
    use_view_slice: bool = false,
    permute: [4]i64 = .{ 0, 0, 0, 0 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "use_view_slice", "permute" });
    }

    /// Builds `cont(view of src)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const src = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(src, "src");

        const p = self.permute;
        const permuted = p[0] != 0 or p[1] != 0 or p[2] != 0 or p[3] != 0;

        var dst: *mod.c.ggml_tensor = undefined;
        if (permuted) {
            dst = mod.c.ggml_permute(ctx, src, @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
            mod.Graph.setName(dst, "src_permuted");
        } else if (self.use_view_slice) {
            const offset = src.nb[0] * @as(usize, @intCast(src.ne[1] - 1));
            dst = mod.c.ggml_view_4d(ctx, src, src.ne[0], 1, src.ne[2], src.ne[3], src.nb[1], src.nb[2], src.nb[3], offset);
            mod.Graph.setName(dst, "src_view_slice");
        } else {
            dst = mod.c.ggml_transpose(ctx, src);
            mod.Graph.setName(dst, "src_transposed");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, dst);
        mod.Graph.setName(out, "out");
        return out;
    }
};
