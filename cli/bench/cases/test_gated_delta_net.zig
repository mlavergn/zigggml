//! Port of `test_gated_delta_net` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GATED_DELTA_NET.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GATED_DELTA_NET on L2-normalized q and k, optionally with
/// permuted (non-contiguous) inputs, a per-element (KDA) gate, repeated
/// value heads and the last `K` states kept.
pub const TestGatedDeltaNet = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    head_count: i64 = 4,
    head_size: i64 = 16,
    n_seq_tokens: i64 = 1,
    n_seqs: i64 = 1,
    /// Value heads per query/key head.
    v_repeat: i32 = 1,
    /// q, k and v are permuted views.
    permuted: bool = false,
    /// A per-element gate (Kimi Delta Attention) instead of a scalar one.
    kda: bool = false,
    /// Snapshot slot count: 1 = final-only, > 1 = the last K states.
    K: i64 = 1,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "head_count", "head_size", "n_seq_tokens", "n_seqs", "v_repeat", "permuted", "kda", "K" });
    }

    /// Builds `gated_delta_net(l2_norm(q), l2_norm(k), v, g, beta, state, K)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const hv = self.head_count * self.v_repeat;
        var q: *mod.c.ggml_tensor = undefined;
        var k: *mod.c.ggml_tensor = undefined;
        var v: *mod.c.ggml_tensor = undefined;
        if (self.permuted) {
            // Create with dims 1 and 2 swapped, then permute back for a non-contiguous layout.
            q = mod.c.ggml_permute(ctx, g.newTensor4d(ctx, self.type, self.head_size, self.n_seq_tokens, self.head_count, self.n_seqs), 0, 2, 1, 3);
            k = mod.c.ggml_permute(ctx, g.newTensor4d(ctx, self.type, self.head_size, self.n_seq_tokens, self.head_count, self.n_seqs), 0, 2, 1, 3);
            v = mod.c.ggml_permute(ctx, g.newTensor4d(ctx, self.type, self.head_size, self.n_seq_tokens, hv, self.n_seqs), 0, 2, 1, 3);
        } else {
            q = g.newTensor4d(ctx, self.type, self.head_size, self.head_count, self.n_seq_tokens, self.n_seqs);
            k = g.newTensor4d(ctx, self.type, self.head_size, self.head_count, self.n_seq_tokens, self.n_seqs);
            v = g.newTensor4d(ctx, self.type, self.head_size, hv, self.n_seq_tokens, self.n_seqs);
        }
        mod.Graph.setName(q, "q");
        mod.Graph.setName(k, "k");
        mod.Graph.setName(v, "v");
        const g_ne0 = if (self.kda) self.head_size else 1;
        const gate = g.newTensor4d(ctx, self.type, g_ne0, hv, self.n_seq_tokens, self.n_seqs);
        const beta = g.newTensor4d(ctx, self.type, 1, hv, self.n_seq_tokens, self.n_seqs);
        const state = g.newTensor4d(ctx, self.type, self.head_size, self.head_size, hv, self.n_seqs);
        mod.Graph.setName(gate, "g");
        mod.Graph.setName(beta, "beta");
        mod.Graph.setName(state, "state");
        // q and k are L2-normalized in qwen35/kimi-linear before the delta net.
        q = mod.c.ggml_l2_norm(ctx, q, 1e-6);
        k = mod.c.ggml_l2_norm(ctx, k, 1e-6);
        return mod.c.ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, self.K);
    }

    /// Fills the gate in [-20, -1e-4), beta in [0, 1), v in [-0.3, 5) and
    /// the rest in [-1, 1).
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
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
