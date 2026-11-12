//! Port of `test_pad_ext` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_PAD.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_PAD on both sides of every dimension, with zeros or circularly,
/// of an optionally viewed or permuted input.
pub const TestPadExt = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 512, 512, 3, 1 },
    lp0: i32 = 1,
    rp0: i32 = 1,
    lp1: i32 = 1,
    rp1: i32 = 1,
    lp2: i32 = 1,
    rp2: i32 = 1,
    lp3: i32 = 1,
    rp3: i32 = 1,
    /// Transform: 0 none, 1 non-contiguous, 2 permuted.
    tfrm: i32 = 0,
    circular: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "lp0", "rp0", "lp1", "rp1", "lp2", "rp2", "lp3", "rp3", "tfrm", "circular" });
    }

    /// Builds `pad_ext(a, ...)` or its circular form.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a = g.newTensor(ctx, self.type, 4, &self.ne_a);
        mod.Graph.setName(a, "a");

        if (self.tfrm == 1) {
            a = mod.c.ggml_view_4d(ctx, a, @divTrunc(a.ne[0] + 1, 2), @divTrunc(a.ne[1] + 1, 2), @divTrunc(a.ne[2] + 1, 2), @divTrunc(a.ne[3] + 1, 2), a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view of a");
        } else if (self.tfrm == 2) {
            a = mod.c.ggml_permute(ctx, a, 2, 1, 0, 3);
            mod.Graph.setName(a, "permuted a");
        }

        const out: *mod.c.ggml_tensor = if (self.circular)
            mod.c.ggml_pad_ext_circular(ctx, a, self.lp0, self.rp0, self.lp1, self.rp1, self.lp2, self.rp2, self.lp3, self.rp3)
        else
            mod.c.ggml_pad_ext(ctx, a, self.lp0, self.rp0, self.lp1, self.rp1, self.lp2, self.rp2, self.lp3, self.rp3);
        mod.Graph.setName(out, "out");
        return out;
    }
};
