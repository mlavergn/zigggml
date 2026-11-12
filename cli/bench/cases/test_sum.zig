//! Port of `test_sum` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SUM.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SUM of all elements, of an optionally sliced and permuted input.
pub const TestSum = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },
    permute: [4]i64 = .{ 0, 0, 0, 0 },
    slice: bool = false,

    // State.
    /// Whether `permute` is not all zero.
    use_permute: bool = false,

    /// Derives `use_permute` from `permute`.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.use_permute = self.permute[0] + self.permute[1] + self.permute[2] + self.permute[3] > 0;
    }

    /// Writes the case's parameters; `permute` and `slice` only when used.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type", self.type);
        try v.add("ne", self.ne);
        if (self.use_permute) try v.add("permute", self.permute);
        if (self.slice) try v.add("slice", self.slice);
    }

    /// Builds `sum(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        var a = g.newTensor(ctx, self.type, 4, &ne);
        mod.Graph.setName(a, "a");

        if (self.slice) {
            a = mod.c.ggml_view_4d(ctx, a, ne[0], ne[1], @divTrunc(ne[2], 2), ne[3] - 1, a.nb[1], a.nb[2] * 2, a.nb[3], a.nb[3]);
            mod.Graph.setName(a, "a_sliced");
        }

        if (self.use_permute) {
            const p = self.permute;
            a = mod.c.ggml_permute(ctx, a, @intCast(p[0]), @intCast(p[1]), @intCast(p[2]), @intCast(p[3]));
            mod.Graph.setName(a, "a_permuted");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_sum(ctx, a);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills off-center, in [-0.9, 1.1), to avoid catastrophic cancellation.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -0.9, 1.1);
    }
};
