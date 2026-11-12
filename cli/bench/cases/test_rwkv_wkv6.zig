//! Port of `test_rwkv_wkv6` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_RWKV_WKV6.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_RWKV_WKV6: the RWKV-6 weighted key-value recurrence.
pub const TestRwkvWkv6 = struct {
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

    /// Builds `rwkv_wkv6(k, v, r, tf, td, s)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const n_tokens = self.n_seq_tokens * self.n_seqs;
        const ne_tok = [3]i64{ self.head_size, self.head_count, n_tokens };
        const ne_tf = [2]i64{ self.head_size, self.head_count };
        const ne_s = [2]i64{ self.head_size * self.head_size * self.head_count, self.n_seqs };
        const r = g.newTensor(ctx, self.type, 3, &ne_tok);
        const k = g.newTensor(ctx, self.type, 3, &ne_tok);
        const v = g.newTensor(ctx, self.type, 3, &ne_tok);
        const tf = g.newTensor(ctx, self.type, 2, &ne_tf);
        const td = g.newTensor(ctx, self.type, 3, &ne_tok);
        const s = g.newTensor(ctx, self.type, 2, &ne_s);
        return mod.c.ggml_rwkv_wkv6(ctx, k, v, r, tf, td, s);
    }
};
