//! Port of `test_rwkv_wkv7` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_RWKV_WKV7.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_RWKV_WKV7: the RWKV-7 recurrence, on L2-normalized `a` and `b`.
pub const TestRwkvWkv7 = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    head_count: i64 = 32,
    head_size: i64 = 64,
    n_seq_tokens: i64 = 32,
    n_seqs: i64 = 32,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "head_count", "head_size", "n_seq_tokens", "n_seqs" });
    }

    /// Builds `rwkv_wkv7(r, w, k, v, l2_norm(a), l2_norm(b), s)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const n_tokens = self.n_seq_tokens * self.n_seqs;
        const ne_tok = [3]i64{ self.head_size, self.head_count, n_tokens };
        const ne_s = [2]i64{ self.head_size * self.head_size * self.head_count, self.n_seqs };
        const r = g.newTensor(ctx, self.type, 3, &ne_tok);
        const w = g.newTensor(ctx, self.type, 3, &ne_tok);
        const k = g.newTensor(ctx, self.type, 3, &ne_tok);
        const v = g.newTensor(ctx, self.type, 3, &ne_tok);
        var a = g.newTensor(ctx, self.type, 3, &ne_tok);
        var b = g.newTensor(ctx, self.type, 3, &ne_tok);
        // Outputs may become NaN with long sequences without this normalization.
        a = mod.c.ggml_l2_norm(ctx, a, 1e-7);
        b = mod.c.ggml_l2_norm(ctx, b, 1e-7);
        const s = g.newTensor(ctx, self.type, 2, &ne_s);
        return mod.c.ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, s);
    }
};
