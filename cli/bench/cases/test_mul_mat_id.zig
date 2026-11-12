//! Port of `test_mul_mat_id` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT_ID.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_MUL_MAT_ID, C^T = A * B^T per selected expert: (k, m) * (k, n) => (m, n).
pub const TestMulMatId = struct {
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
    /// Rows of `as` in memory: the experts of `as` are strided for m_v > m, no view for 0.
    m_v: i64 = 0,
    ids_offset: bool = false,
    /// Set some ids to a negative value.
    skip: bool = false,

    // State.
    gf: ?*mod.c.ggml_cgraph = null,

    /// Checks the constructor's invariants.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        std.debug.assert(self.n_used <= self.n_mats);
        std.debug.assert(self.m_v == 0 or self.m_v > self.m);
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_a", "type_b", "n_mats", "n_used", "b", "m", "n", "k", "m_v", "ids_offset", "skip" });
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// A looser bound where Blackwell quantizes activations to FP4, or a
    /// backend computes natively in FP8.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        const ta = self.type_a.raw();
        if ((ta == mod.c.GGML_TYPE_MXFP4 or ta == mod.c.GGML_TYPE_NVFP4) and
            !mod.Helpers.graphMulMatHiPrecAct(self.gf, mod.c.GGML_OP_MUL_MAT_ID) and
            mod.Helpers.backendHasFeature(backend, "BLACKWELL_NATIVE_FP4")) return 2e-2;
        if (ta == mod.c.GGML_TYPE_F8_E4M3 and mod.Helpers.backendHasFeature(backend, "NATIVE_FP8")) return 5e-3;
        return self.maxNmseErr();
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

    /// Builds `mul_mat_id(as, b, ids)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        self.gf = g.gf;
        var as_t = g.newTensor3d(ctx, self.type_a, self.k, if (self.m_v == 0) self.m else self.m_v, self.n_mats);
        if (self.m_v != 0) {
            as_t = mod.c.ggml_view_3d(ctx, as_t, self.k, self.m, self.n_mats, as_t.nb[1], as_t.nb[2], 0);
        }
        mod.Graph.setName(as_t, "as");

        var ids = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.n_mats + @intFromBool(self.ids_offset), self.n);
        mod.Graph.setName(ids, "ids");
        if (self.n_used != self.n_mats or self.ids_offset) {
            ids = mod.c.ggml_view_2d(ctx, ids, self.n_used, self.n, ids.nb[1], if (self.ids_offset) @sizeOf(i32) else 0);
            mod.Graph.setName(ids, "view_of_ids");
        }

        const b_t = g.newTensor3d(ctx, self.type_b, self.k, if (self.b) 1 else self.n_used, self.n);
        mod.Graph.setName(b_t, "b");

        const out: *mod.c.ggml_tensor = mod.c.ggml_mul_mat_id(ctx, as_t, b_t, ids);
        mod.Graph.setName(out, "out");
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

    /// Re-draws the ids between timed runs.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn reinitPerfIter(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        mod.Helpers.initMulMatIdIds(ti, ctx, self.n_mats, self.skip);
    }
};
