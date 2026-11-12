//! Port of `test_moe_reduce` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ADD.

const std = @import("std");
const mod = @import("module.zig");

/// The MoE reduction: expert outputs weighted, optionally scaled, then
/// summed over experts through views and adds that backends may fuse.
pub const TestMoeReduce = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    n_embd: i64 = 0,
    n_expert_used: i64 = 0,
    n_tokens: i64 = 0,
    unaligned_experts: bool = false,
    with_expert_scale: bool = false,
    interleaved_views_adds: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "n_embd", "n_expert_used", "n_tokens", "unaligned_experts", "with_expert_scale", "interleaved_views_adds" });
    }

    /// Always `MOE_REDUCE`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MOE_REDUCE");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds the sum over experts of `experts * weights`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) !*mod.c.ggml_tensor {
        const ctx = g.ctx;
        var experts: *mod.c.ggml_tensor = undefined;
        if (self.unaligned_experts) {
            const storage = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, self.n_embd * self.n_expert_used * self.n_tokens + 1);
            mod.Graph.setName(storage, "experts_storage");
            const n_embd: usize = @intCast(self.n_embd);
            const n_expert_used: usize = @intCast(self.n_expert_used);
            experts = mod.c.ggml_view_3d(ctx, storage, self.n_embd, self.n_expert_used, self.n_tokens, n_embd * @sizeOf(f32), n_embd * n_expert_used * @sizeOf(f32), @sizeOf(f32));
        } else {
            experts = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, self.n_embd, self.n_expert_used, self.n_tokens);
        }
        mod.Graph.setName(experts, "experts");
        const weights = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, 1, self.n_expert_used, self.n_tokens);
        mod.Graph.setName(weights, "weights");

        var scaled = experts;
        if (self.with_expert_scale) {
            const expert_scale = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, 1, self.n_expert_used, self.n_tokens);
            mod.Graph.setName(expert_scale, "expert_scale");
            scaled = mod.c.ggml_mul(ctx, experts, expert_scale);
            mod.Graph.setName(scaled, "scaled_experts");
        }

        const weighted: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, scaled, weights);
        mod.Graph.setName(weighted, "weighted_experts");

        const expand = !self.interleaved_views_adds and g.mode == .check;
        const views = try g.allocator.alloc(*mod.c.ggml_tensor, @intCast(self.n_expert_used));
        for (views, 0..) |*view, expert| {
            view.* = mod.c.ggml_view_2d(ctx, weighted, self.n_embd, self.n_tokens, weighted.nb[2], expert * weighted.nb[1]);
            if (expand) mod.c.ggml_build_forward_expand(g.gf.?, view.*);
        }

        var out = views[0];
        for (views[1..]) |view| {
            out = mod.c.ggml_add(ctx, out, view);
            if (expand) mod.c.ggml_build_forward_expand(g.gf.?, out);
        }
        mod.Graph.setName(out, "moe_reduce");
        return out;
    }
};
