//! Port of `test_topk_moe` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ARGSORT.

const std = @import("std");
const mod = @import("module.zig");

/// How expert probabilities are computed from the router logits
/// (upstream's `MoeGatingFunc`). `vars` prints its ordinal.
pub const MoeGatingFunc = enum(i32) {
    /// `GATING_FUNC_SOFTMAX`.
    softmax = 0,
    /// `GATING_FUNC_SIGMOID`.
    sigmoid = 1,
    /// `GATING_FUNC_SOFTMAX_WEIGHT`: softmax over the selected weights.
    softmaxWeight = 2,
    /// `GATING_FUNC_SQRT_SOFTPLUS`.
    sqrtSoftplus = 3,
};

/// The MoE router: gating, optional bias and expert groups, top-k expert
/// selection, then weights, optionally normalized and scaled.
pub const TestTopkMoe = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    ne: [4]i64 = .{ 10, 5, 1, 1 },
    n_expert_used: i32 = 1,
    with_norm: bool = false,
    bias_probs: bool = false,
    gating_func: MoeGatingFunc = .softmax,
    scale_w: f32 = 0.0,
    n_expert_groups: i32 = 1,

    // State.
    /// The selected experts and the weights, the nodes the fused comparison checks.
    nodes: [2]*mod.c.ggml_tensor = undefined,

    /// Checks the constructor's invariants.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        std.debug.assert(self.n_expert_used <= self.ne[0]);
        std.debug.assert(@mod(self.ne[0], self.n_expert_groups) == 0);
        std.debug.assert(self.n_expert_used <= @divTrunc(self.ne[0], self.n_expert_groups));
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "ne", "n_expert_used", "with_norm", "bias_probs", "gating_func", "scale_w", "n_expert_groups" });
    }

    /// Always `TOPK_MOE`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("TOPK_MOE");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds the router.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the weights.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const n_expert: i64 = self.ne[0];
        const n_tokens: i64 = self.ne[1];
        const n_exp_per_group: i64 = @divTrunc(n_expert, self.n_expert_groups);
        const n_group_used: i32 = @divTrunc(self.n_expert_groups, 2);
        const n_expert_used: i64 = self.n_expert_used;

        const logits = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &self.ne);
        const probs: *mod.c.ggml_tensor = switch (self.gating_func) {
            .softmax => mod.c.ggml_soft_max(ctx, logits),
            .sigmoid => mod.c.ggml_sigmoid(ctx, logits),
            .sqrtSoftplus => mod.c.ggml_sqrt(ctx, mod.c.ggml_softplus(ctx, logits)),
            .softmaxWeight => logits,
        };
        mod.Graph.setName(probs, "probs");

        var selection_probs = probs;
        if (self.bias_probs) {
            const exp_probs_b = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, self.ne[0]);
            mod.Graph.setName(exp_probs_b, "exp_probs_b");
            selection_probs = mod.c.ggml_add(ctx, probs, exp_probs_b);
            mod.Graph.setName(selection_probs, "selection_probs");
        }

        if (self.n_expert_groups > 1) {
            // [n_exp_per_group, n_expert_groups, n_tokens]
            const selection_groups: *mod.c.ggml_tensor = mod.c.ggml_reshape_3d(ctx, selection_probs, n_exp_per_group, self.n_expert_groups, n_tokens);

            // [2, n_expert_groups, n_tokens]
            var group_scores: *mod.c.ggml_tensor = mod.c.ggml_argsort_top_k(ctx, selection_groups, 2);
            // [1, 2, n_expert_groups, n_tokens]
            group_scores = mod.c.ggml_get_rows(ctx, mod.c.ggml_reshape_4d(ctx, selection_groups, 1, selection_groups.ne[0], selection_groups.ne[1], selection_groups.ne[2]), group_scores);
            // [1, n_expert_groups, n_tokens]
            group_scores = mod.c.ggml_sum_rows(ctx, mod.c.ggml_reshape_3d(ctx, group_scores, group_scores.ne[1], group_scores.ne[2], group_scores.ne[3]));
            // [n_expert_groups, n_tokens]
            group_scores = mod.c.ggml_reshape_2d(ctx, group_scores, group_scores.ne[1], group_scores.ne[2]);

            // [n_group_used, n_tokens]
            const expert_groups: *mod.c.ggml_tensor = mod.c.ggml_argsort_top_k(ctx, group_scores, n_group_used);
            mod.Graph.setName(expert_groups, "ffn_moe_group_topk");

            // [n_exp_per_group, n_group_used, n_tokens]
            selection_probs = mod.c.ggml_get_rows(ctx, selection_groups, expert_groups);
            // [n_exp_per_group, n_expert_groups, n_tokens]
            selection_probs = mod.c.ggml_set_rows(ctx, mod.c.ggml_fill(ctx, selection_groups, -std.math.inf(f32)), selection_probs, expert_groups);
            // [n_expert, n_tokens]
            selection_probs = mod.c.ggml_reshape_2d(ctx, selection_probs, n_expert, n_tokens);
            mod.Graph.setName(selection_probs, "ffn_moe_probs_masked");
        }

        // [n_expert_used, n_tokens]
        const selected_experts: *mod.c.ggml_tensor = mod.c.ggml_argsort_top_k(ctx, selection_probs, self.n_expert_used);
        mod.Graph.setName(selected_experts, "selected_experts");

        // [1, n_expert_used, n_tokens]
        var weights: *mod.c.ggml_tensor = mod.c.ggml_get_rows(ctx, mod.c.ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts);
        mod.Graph.setName(weights, "weights");

        if (self.gating_func == .softmaxWeight) {
            weights = mod.c.ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            weights = mod.c.ggml_soft_max(ctx, weights); // [n_expert_used, n_tokens]
            weights = mod.c.ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        }

        if (self.with_norm) {
            weights = mod.c.ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            var weights_sum: *mod.c.ggml_tensor = mod.c.ggml_sum_rows(ctx, weights); // [1, n_tokens]
            mod.Graph.setName(weights_sum, "weights_sum");
            weights_sum = mod.c.ggml_clamp(ctx, weights_sum, 6.103515625e-5, std.math.inf(f32));
            weights = mod.c.ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
            weights = mod.c.ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        }

        if (self.scale_w != 0.0) {
            weights = mod.c.ggml_scale(ctx, weights, self.scale_w);
        }

        mod.Graph.setName(weights, "weights");
        self.nodes = .{ selected_experts, weights };
        return weights;
    }

    /// Both outputs: the selected experts and the weights.
    ///
    /// - `self`: the case.
    /// - Returns: the nodes.
    pub fn fusionTestNodes(self: *Self) []const *mod.c.ggml_tensor {
        return &self.nodes;
    }

    /// NMSE of the sorted values, so the output may come in any order.
    ///
    /// - `self`: the case.
    /// - `allocator`: scratch allocator.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: *Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        _ = self;
        const a2 = allocator.dupe(f32, a) catch @panic("OOM");
        defer allocator.free(a2);
        const b2 = allocator.dupe(f32, b) catch @panic("OOM");
        defer allocator.free(b2);
        std.mem.sort(f32, a2, {}, std.sort.asc(f32));
        std.mem.sort(f32, b2, {}, std.sort.asc(f32));
        return mod.Compare.nmse(a2, b2);
    }
};
