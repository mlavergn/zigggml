//! Port of `test_gated_delta_net_cache_fusion` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GATED_DELTA_NET + GGML_OP_CPY.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GATED_DELTA_NET whose snapshot states are copied into a
/// recurrent cache, which a backend may fuse. Compared as one graph.
pub const TestGatedDeltaNetCacheFusion = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    head_count: i64 = 4,
    head_size: i64 = 32,
    n_seq_tokens: i64 = 2,
    n_seqs: i64 = 1,
    /// Snapshot slot count (> 1).
    K: i64 = 2,

    // State.
    /// The cache copy, the node the fused comparison checks.
    fusion_nodes: [1]*mod.c.ggml_tensor = undefined,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "head_count", "head_size", "n_seq_tokens", "n_seqs", "K" });
    }

    /// Always `GATED_DELTA_NET_CACHE_FUSION`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("GATED_DELTA_NET_CACHE_FUSION");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The cache copy.
    ///
    /// - `self`: the case.
    /// - Returns: the nodes, valid after `buildGraph`.
    pub fn fusionTestNodes(self: *Self) []const *mod.c.ggml_tensor {
        return &self.fusion_nodes;
    }

    /// `(4·S_v + 2·S_v²)·H_v` per token.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        const s_v: u64 = @intCast(self.head_size);
        const h_v: u64 = @intCast(self.head_count);
        const n_t: u64 = @intCast(self.n_seq_tokens);
        const n_b: u64 = @intCast(self.n_seqs);
        return (4 * s_v + 2 * s_v * s_v) * h_v * n_t * n_b;
    }

    /// Builds `sum(cpy(snapshots(gated_delta_net(...)), cache))`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const s_v = self.head_size;
        const h_v = self.head_count;
        const h_k = self.head_count;
        const d = s_v * s_v * h_v;
        const n_written = @min(self.n_seq_tokens, self.K);

        var q = g.newTensor4d(ctx, self.type, self.head_size, h_k, self.n_seq_tokens, self.n_seqs);
        var k = g.newTensor4d(ctx, self.type, self.head_size, h_k, self.n_seq_tokens, self.n_seqs);
        const v = g.newTensor4d(ctx, self.type, self.head_size, h_v, self.n_seq_tokens, self.n_seqs);
        mod.Graph.setName(q, "q");
        mod.Graph.setName(k, "k");
        mod.Graph.setName(v, "v");
        const gate = g.newTensor4d(ctx, self.type, 1, h_v, self.n_seq_tokens, self.n_seqs);
        const beta = g.newTensor4d(ctx, self.type, 1, h_v, self.n_seq_tokens, self.n_seqs);
        const state = g.newTensor4d(ctx, self.type, self.head_size, self.head_size, h_v, self.n_seqs);
        mod.Graph.setName(gate, "g");
        mod.Graph.setName(beta, "beta");
        mod.Graph.setName(state, "state");

        q = mod.c.ggml_l2_norm(ctx, q, 1e-6);
        k = mod.c.ggml_l2_norm(ctx, k, 1e-6);

        const gdn_out: *mod.c.ggml_tensor = mod.c.ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, self.K);
        mod.Graph.setName(gdn_out, "gdn_out");

        // Attention scores view (the first part of the output).
        const ty = gdn_out.type;
        const attn: *mod.c.ggml_tensor = mod.c.ggml_view_4d(ctx, gdn_out, s_v, h_v, self.n_seq_tokens, self.n_seqs, mod.c.ggml_row_size(ty, s_v), mod.c.ggml_row_size(ty, s_v * h_v), mod.c.ggml_row_size(ty, s_v * h_v * self.n_seq_tokens), 0);
        mod.Graph.setName(attn, "attn");

        // Snapshot tail view [D, n_seqs, n_written].
        const attn_score_elems = s_v * h_v * self.n_seq_tokens * self.n_seqs;
        const src: *mod.c.ggml_tensor = mod.c.ggml_view_3d(ctx, gdn_out, d, self.n_seqs, n_written, mod.c.ggml_row_size(ty, d), mod.c.ggml_row_size(ty, d * self.n_seqs), mod.c.ggml_row_size(ty, attn_score_elems));

        // Recurrent cache view [D, n_seqs, n_written].
        const cache = g.newTensor3d(ctx, self.type, d, self.n_seqs, n_written);
        mod.Graph.setName(cache, "cache");
        const dst: *mod.c.ggml_tensor = mod.c.ggml_view_3d(ctx, cache, d, self.n_seqs, n_written, mod.c.ggml_row_size(cache.type, d), mod.c.ggml_row_size(cache.type, d * self.n_seqs), 0);

        const cpy: *mod.c.ggml_tensor = mod.c.ggml_cpy(ctx, src, dst);
        mod.Graph.setName(cpy, "gdn_cache_cpy");
        self.fusion_nodes = .{cpy};

        // Read the cpy output (not the plain dst view, which would not pull
        // the cpy into the graph), so neither the GDN nor the cpy is the
        // graph output.
        return mod.c.ggml_sum(ctx, cpy);
    }

    /// Fills the gate in [-20, -1e-4), beta in [0, 1), v in [-0.3, 5), the
    /// cache with zeros and the rest in [-1, 1).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (mod.Helpers.isViewOp(t.op)) continue;
            const name = std.mem.span(mod.c.ggml_get_name(t));
            if (std.mem.eql(u8, name, "g")) {
                ti.uniform(t, -20.0, -1e-4);
            } else if (std.mem.eql(u8, name, "beta")) {
                ti.uniform(t, 0.0, 1.0);
            } else if (std.mem.eql(u8, name, "v")) {
                ti.uniform(t, -0.3, 5.0);
            } else if (std.mem.eql(u8, name, "cache")) {
                ti.uniform(t, 0.0, 0.0);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
