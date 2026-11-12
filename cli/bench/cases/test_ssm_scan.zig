//! Port of `test_ssm_scan` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SSM_SCAN.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SSM_SCAN: the Mamba-1 (`head_dim == 1`) or Mamba-2 selective
/// scan, optionally with x, B and C overlapping views of one tensor and with
/// the last `K` states kept.
pub const TestSsmScan = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    d_state: i64 = 32,
    /// 1 for Mamba-1; > 1 for Mamba-2 (scalar A per head).
    head_dim: i64 = 1,
    n_head: i64 = 32,
    n_group: i64 = 1,
    n_seq_tokens: i64 = 32,
    n_seqs: i64 = 32,
    /// x, B and C are views of one tensor.
    xbc_overlap: bool = false,
    /// Snapshot slot count.
    K: i64 = 1,
    /// A weak (near-zero) decay.
    weak_decay: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "d_state", "head_dim", "n_head", "n_group", "n_seq_tokens", "n_seqs", "xbc_overlap", "K", "weak_decay" });
    }

    /// The SSD path (`head_dim > 1`) uses FP16 intermediates; Mamba-1 is
    /// pure FP32.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return if (self.head_dim > 1) 2e-7 else 1e-7;
    }

    /// Builds `ssm_scan(s, x, dt, A, B, C, ids, K)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const s = g.newTensor4d(ctx, self.type, self.d_state, self.head_dim, self.n_head, self.n_seqs);
        const dt = g.newTensor3d(ctx, self.type, self.n_head, self.n_seq_tokens, self.n_seqs);
        const A = g.newTensor2d(ctx, self.type, if (self.head_dim > 1) 1 else self.d_state, self.n_head);
        var x: *mod.c.ggml_tensor = undefined;
        var B: *mod.c.ggml_tensor = undefined;
        var C: *mod.c.ggml_tensor = undefined;

        if (self.xbc_overlap) {
            const xbc = g.newTensor4d(ctx, self.type, self.d_state, self.n_head, self.n_seq_tokens, 2 * self.n_seqs);
            x = mod.c.ggml_view_4d(ctx, xbc, self.head_dim, self.n_head, self.n_seq_tokens, self.n_seqs, xbc.nb[1], xbc.nb[2], xbc.nb[3], xbc.nb[3]);
            B = mod.c.ggml_view_4d(ctx, xbc, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs, xbc.nb[1], xbc.nb[2], xbc.nb[3], 0);
            C = mod.c.ggml_view_4d(ctx, xbc, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs, xbc.nb[1], xbc.nb[2], xbc.nb[3], 2 * xbc.nb[3]);
        } else {
            x = g.newTensor4d(ctx, self.type, self.head_dim, self.n_head, self.n_seq_tokens, self.n_seqs);
            B = g.newTensor4d(ctx, self.type, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs);
            C = g.newTensor4d(ctx, self.type, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs);
        }
        const ids = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, self.n_seqs);
        return mod.c.ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, self.K);
    }

    /// Fills `ids` with permutations of `0..n_seqs`, `A` with a negative
    /// decay, and the rest uniformly in [-1, 1).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                if (mod.Helpers.isViewOp(t.op)) continue;
                initPermutedIds(ti, t);
            } else if (mod.Helpers.isViewOp(t.op)) {
                continue;
            } else if (t.ne[1] == self.n_head and t.ne[2] == 1) {
                // A {1 or d_state, n_head}: ne[2] == 1 tells it from the 3-D/4-D tensors.
                if (self.weak_decay) ti.uniform(t, -0.02, -0.005) else ti.uniform(t, -1.0, -0.5);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};

/// Fills each row of I32 tensor `t` with a shuffled `0..ne[0]`.
fn initPermutedIds(ti: *mod.TensorInit, t: *mod.c.ggml_tensor) void {
    const n: usize = @intCast(t.ne[0]);
    const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
    defer ti.allocator.free(data);
    var r: i64 = 0;
    while (r < mod.c.ggml_nrows(t)) : (r += 1) {
        for (data, 0..) |*d, i| d.* = @intCast(i);
        ti.random.shuffle(i32, data);
        mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], n * @sizeOf(i32));
    }
}
