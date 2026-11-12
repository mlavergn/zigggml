//! Port of `test_mul_mat_vec_fusion` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID + GGML_OP_GLU.

const std = @import("std");
const mod = @import("module.zig");

/// A gated FFN lane on mat-vec products (dense or by expert id), with
/// optional bias, lane scale and gate, which backends may fuse.
pub const TestMulMatVecFusion = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// The C++ constructor's `op`.
    glu_op: mod.c.enum_ggml_glu_op = 0,
    m: i64 = 0,
    n: i64 = 0,
    k: i64 = 0,
    use_id: bool = false,
    n_mats: i32 = 1,
    n_used: i32 = 1,
    /// Broadcast the b matrix (only for use_id).
    b: bool = false,
    with_bias: bool = false,
    with_gate: bool = true,
    with_lane_scale: bool = false,
    batch_dims: [2]i64 = .{ 4, 2 },
    /// Set some ids to a negative value (only for use_id).
    skip: bool = false,

    /// Checks the constructor's invariant.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        if (self.use_id) std.debug.assert(self.n_used <= self.n_mats);
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "glu_op", "m", "n", "k", "use_id", "n_mats", "n_used", "b", "with_bias", "with_gate", "with_lane_scale", "batch_dims", "skip" });
    }

    /// Always `MUL_MAT_VEC_FUSION`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_VEC_FUSION");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The per-expert lane scale lives in the weights context.
    ///
    /// - `self`: the case.
    /// - Returns: true for `use_id` with `with_lane_scale`.
    pub fn useWeightContext(self: *Self) bool {
        return self.use_id and self.with_lane_scale;
    }

    /// Builds the lane.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        if (!self.use_id) {
            const channels = self.batch_dims[0];
            const samples = self.batch_dims[1];
            const ne = [4]i64{ self.k, self.m, channels, samples };
            const ne0 = [4]i64{ self.k, self.n, channels, samples };

            const cur = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &ne);
            const gate: ?*mod.c.ggml_tensor = if (self.with_gate) g.newTensor(ctx, self.type, 4, &ne0) else null;
            const up = g.newTensor(ctx, self.type, 4, &ne0);

            const ffn_up = self.buildLaneDense(g, up, cur);
            const out0 = if (self.with_gate) self.buildGate(ctx, self.buildLaneDense(g, gate.?, cur), ffn_up) else ffn_up;

            const bias2_ne = [4]i64{ out0.ne[0], 1, channels, samples };
            const bias2 = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &bias2_ne);
            const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, out0, bias2);
            mod.Graph.setName(out, "out");
            return out;
        }

        const gates = g.newTensor3d(ctx, self.type, self.k, self.n, self.n_mats);
        const ups = g.newTensor3d(ctx, self.type, self.k, self.n, self.n_mats);
        var ids = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.n_mats, self.m);
        if (self.n_used != self.n_mats) {
            ids = mod.c.ggml_view_2d(ctx, ids, self.n_used, self.m, ids.nb[1], 0);
        }

        const cur = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, self.k, if (self.b) 1 else self.n_used, self.m);
        mod.Graph.setName(cur, "cur");

        const ffn_up = self.buildLaneId(g, ups, cur, ids);
        const out0 = if (self.with_gate) self.buildGate(ctx, self.buildLaneId(g, gates, cur, ids), ffn_up) else ffn_up;

        const scale_ne = [4]i64{ 1, out0.ne[1], out0.ne[2], out0.ne[3] };
        const scale = g.newTensor(ctx, out0.type, 4, &scale_ne);
        const out: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, out0, scale);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills everything uniformly, or with expert ids for `use_id`.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        if (!self.use_id) {
            var it = mod.Helpers.tensors(ctx);
            while (it.next()) |t| ti.uniform(t, -1.0, 1.0);
        } else {
            mod.Helpers.initMulMatIdTensors(ti, ctx, self.n_mats, self.skip);
        }
    }

    /// The bound for fused mat-vec products.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-3;
    }

    /// The gated product of the gate and up lanes (`build_gate`).
    fn buildGate(self: *Self, ctx: *mod.c.ggml_context, ffn_gate: *mod.c.ggml_tensor, ffn_up: *mod.c.ggml_tensor) *mod.c.ggml_tensor {
        if (self.glu_op == mod.c.GGML_GLU_OP_SWIGLU_OAI) {
            const alpha: f32 = 1.702;
            const limit: f32 = 7.0;
            return mod.c.ggml_swiglu_oai(ctx, ffn_gate, ffn_up, alpha, limit);
        } else if (self.glu_op == mod.c.GGML_GLU_OP_SWIGLU_CLAMP) {
            const limit: f32 = 10.0;
            return mod.c.ggml_swiglu_clamp(ctx, ffn_gate, ffn_up, limit);
        }
        return mod.c.ggml_glu_split(ctx, ffn_gate, ffn_up, self.glu_op);
    }

    /// One dense lane: `mul_mat(w, cur)`, optionally scaled and biased
    /// (`build_lane_up` / `build_lane_gate`, `!use_id`).
    fn buildLaneDense(self: *Self, g: *mod.Graph, w: *mod.c.ggml_tensor, cur: *mod.c.ggml_tensor) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var lane: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, w, cur);
        if (self.with_lane_scale) {
            // build_lane_scale_dense
            const scale = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, 1);
            lane = mod.c.ggml_mul(ctx, lane, scale);
        }
        if (self.with_bias) {
            const bias_ne = [4]i64{ lane.ne[0], 1, self.batch_dims[0], self.batch_dims[1] };
            const bias = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &bias_ne);
            lane = mod.c.ggml_add(ctx, lane, bias);
        }
        return lane;
    }

    /// One expert lane: `mul_mat_id(w, cur, ids)`, optionally scaled per
    /// expert and biased (`build_lane_up` / `build_lane_gate`, `use_id`).
    fn buildLaneId(self: *Self, g: *mod.Graph, w: *mod.c.ggml_tensor, cur: *mod.c.ggml_tensor, ids: *mod.c.ggml_tensor) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var lane: *mod.c.ggml_tensor = mod.c.ggml_mul_mat_id(ctx, w, cur, ids);
        if (self.with_lane_scale) {
            // build_lane_scale_id
            const scale = g.newTensor1d(g.ctx_weights.?, mod.c.GGML_TYPE_F32, self.n_mats);
            var s: *mod.c.ggml_tensor = mod.c.ggml_reshape_3d(ctx, scale, 1, self.n_mats, 1);
            s = mod.c.ggml_repeat_4d(ctx, s, 1, self.n_mats, self.m, 1);
            s = mod.c.ggml_get_rows(ctx, s, ids);
            lane = mod.c.ggml_mul(ctx, lane, s);
        }
        if (self.with_bias) {
            const bias = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, lane.ne[0], self.n_mats);
            lane = mod.c.ggml_add_id(ctx, lane, bias, ids);
        }
        return lane;
    }
};
