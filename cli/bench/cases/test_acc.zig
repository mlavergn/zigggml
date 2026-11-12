//! Port of `test_acc` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ACC.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ACC of `b` into `a` with `a`'s strides, `b` optionally a view at
/// a non-zero offset.
pub const TestAcc = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 256, 17, 2, 3 },
    ne_b: [4]i64 = .{ 256, 16, 2, 3 },
    /// The dimension `b` is padded in, 1 to 3, for a view at a non-zero
    /// offset; anything else for none. The C++ parameter is a `uint64_t`
    /// defaulting to -1, stored in an `int64_t`.
    stride_dim: i64 = -1,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "ne_b", "stride_dim" });
    }

    /// Builds `acc(a, b)` with `a`'s strides.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        mod.Graph.setName(a, "a");

        const ne_b = self.ne_b;
        var b: *mod.c.ggml_tensor = undefined;
        if (self.stride_dim == 1 or self.stride_dim == 2 or self.stride_dim == 3) {
            // A larger tensor viewed at a non-zero offset checks that the
            // backend handles b's data offset.
            var ne_b_pad = ne_b;
            ne_b_pad[@intCast(self.stride_dim)] += 1;
            const b_pad = g.newTensor(ctx, self.type, 4, &ne_b_pad);
            mod.Graph.setName(b_pad, "b_pad");
            // The view skips the first row, so b has a non-zero byte offset.
            b = mod.c.ggml_view_4d(ctx, b_pad, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b_pad.nb[1], b_pad.nb[2], b_pad.nb[3], b_pad.nb[1]);
        } else {
            b = g.newTensor(ctx, self.type, 4, &ne_b);
        }
        mod.Graph.setName(b, "b");

        // When ne_b[0] < ne_a[0], a.nb[1] != b.nb[1], so the strides given
        // to acc don't match b's natural stride.
        const out: *mod.c.ggml_tensor = mod.c.ggml_acc(ctx, a, b, a.nb[1], a.nb[2], a.nb[3], 0);
        mod.Graph.setName(out, "out");
        return out;
    }
};
