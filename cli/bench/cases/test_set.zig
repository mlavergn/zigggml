//! Port of `test_set` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SET.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SET of `src` into the middle of a `dst` doubled along the first
/// `dim` dimensions, optionally in place.
pub const TestSet = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_src: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_dst: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 6, 5, 4, 3 },
    dim: i32 = 1,
    inplace: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_src", "type_dst", "ne", "dim", "inplace" });
    }

    /// The output's bytes plus `dst`'s.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the size.
    pub fn opSize(self: *Self, t: *mod.c.ggml_tensor) usize {
        _ = self;
        return mod.c.ggml_nbytes(t) + mod.c.ggml_nbytes(t.src[0]);
    }

    /// Builds `set(dst, src)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const dim: usize = @intCast(self.dim);
        const src = g.newTensor(ctx, self.type_src, 4, &self.ne);
        mod.Graph.setName(src, "src");

        var ne_dst = self.ne;
        for (ne_dst[0..dim]) |*n| n.* *= 2;
        const dst = g.newTensor(ctx, self.type_dst, 4, &ne_dst);
        mod.Graph.setName(dst, "dst");

        var offset: usize = 0;
        for (0..dim) |i| {
            offset += @as(usize, @intCast(@divTrunc(ne_dst[i] - self.ne[i], 2))) * dst.nb[i];
        }
        // The backward pass requires setting a contiguous region.
        const out: *mod.c.ggml_tensor = if (self.inplace)
            mod.c.ggml_set_inplace(ctx, dst, src, src.nb[1], src.nb[2], src.nb[3], offset)
        else
            mod.c.ggml_set(ctx, dst, src, src.nb[1], src.nb[2], src.nb[3], offset);
        mod.Graph.setName(out, "out");
        return out;
    }
};
