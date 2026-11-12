//! Port of `test_mul_mat_id_fusion` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT_ID.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_MUL_MAT_ID followed by GGML_OP_ADD (`o` outputs summed) or
/// GGML_OP_MUL (`mul`), compared as one fused graph.
pub const TestMulMatIdFusion = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    n_mats: i32 = 8,
    n_used: i32 = 2,
    /// Broadcast the b matrix.
    b: bool = false,
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 32,
    /// Number of outputs.
    o: u32 = 1,
    mul: bool = false,
    /// Set some ids to a negative value.
    skip: bool = false,

    /// Checks the constructor's invariants.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        std.debug.assert(self.n_used <= self.n_mats);
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_a", "type_b", "n_mats", "n_used", "b", "m", "n", "k", "o", "mul", "skip" });
    }

    /// The bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// 2·m·k·n·n_used.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        return @intCast(2 * self.m * self.k * self.n * self.n_used);
    }

    /// Builds `mul_mat_id(as, b, ids)`, plus `o - 1` more summed, times a
    /// per-row scale when `mul`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const as_t = g.newTensor3d(ctx, self.type_a, self.k, self.m, self.n_mats);
        mod.Graph.setName(as_t, "as");

        var ids = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.n_mats, self.n);
        mod.Graph.setName(ids, "ids");
        if (self.n_used != self.n_mats) {
            ids = mod.c.ggml_view_2d(ctx, ids, self.n_used, self.n, ids.nb[1], 0);
            mod.Graph.setName(ids, "view_of_ids");
        }

        const b_t = g.newTensor3d(ctx, self.type_b, self.k, if (self.b) 1 else self.n_used, self.n);
        mod.Graph.setName(b_t, "b");

        var out: *mod.c.ggml_tensor = mod.c.ggml_mul_mat_id(ctx, as_t, b_t, ids);
        mod.Graph.setName(out, "out");

        var i: u32 = 1;
        while (i < self.o) : (i += 1) {
            const a2 = g.newTensor3d(ctx, self.type_a, self.k, self.m, self.n_mats);
            const out2: *mod.c.ggml_tensor = mod.c.ggml_mul_mat_id(ctx, a2, b_t, ids);
            mod.Graph.setName(out2, "out2");
            out = mod.c.ggml_add(ctx, out, out2);
        }

        if (self.mul) {
            const ne = [4]i64{ 1, out.ne[1], out.ne[2], out.ne[3] };
            const scale = g.newTensor(ctx, out.type, 4, &ne);
            out = mod.c.ggml_mul(ctx, out, scale);
        }
        return out;
    }

    /// Fills the matrices uniformly and the ids with shuffled experts.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        mod.Helpers.initMulMatIdTensors(ti, ctx, self.n_mats, self.skip);
    }

    /// The fused ops are compared as one graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Always `MUL_MAT_ID_FUSION`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_ID_FUSION");
    }
};
