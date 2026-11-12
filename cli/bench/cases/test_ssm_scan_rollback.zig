//! Port of `test_ssm_scan_rollback` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SSM_SCAN.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SSM_SCAN snapshots: each of the last `K` states of a full scan
/// must equal the final state of a scan over the matching token prefix. The
/// output is the summed squared difference, which should be zero.
pub const TestSsmScanRollback = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    d_state: i64 = 32,
    head_dim: i64 = 64,
    n_head: i64 = 16,
    n_group: i64 = 2,
    n_seq_tokens: i64 = 8,
    n_seqs: i64 = 2,
    /// Snapshot slot count.
    K: i64 = 3,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "d_state", "head_dim", "n_head", "n_group", "n_seq_tokens", "n_seqs", "K" });
    }

    /// Always `SSM_SCAN_ROLLBACK`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("SSM_SCAN_ROLLBACK");
    }

    /// The comparison is between the whole graph's outputs.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The largest accepted difference.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxErr(self: *Self) f64 {
        _ = self;
        return 1e-6;
    }

    /// The largest magnitude in either result: both should be zero.
    ///
    /// - `self`: the case.
    /// - `allocator`: unused.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: *Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        _ = self;
        _ = allocator;
        var result: f64 = 0.0;
        for (a, b) |a_i, b_i| {
            result = @max(result, @as(f64, @abs(a_i)));
            result = @max(result, @as(f64, @abs(b_i)));
        }
        return result;
    }

    /// Builds the summed squared difference between each snapshot and the
    /// matching prefix scan's final state.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const s = g.newTensor4d(ctx, self.type, self.d_state, self.head_dim, self.n_head, self.n_seqs);
        const x = g.newTensor4d(ctx, self.type, self.head_dim, self.n_head, self.n_seq_tokens, self.n_seqs);
        const dt = g.newTensor3d(ctx, self.type, self.n_head, self.n_seq_tokens, self.n_seqs);
        const A = g.newTensor2d(ctx, self.type, 1, self.n_head);
        const B = g.newTensor4d(ctx, self.type, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs);
        const C = g.newTensor4d(ctx, self.type, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs);
        const ids = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, self.n_seqs);

        const full: *mod.c.ggml_tensor = mod.c.ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, self.K);

        const y_elems = self.head_dim * self.n_head * self.n_seq_tokens * self.n_seqs;
        const state_elems = self.d_state * self.head_dim * self.n_head * self.n_seqs;

        var out: ?*mod.c.ggml_tensor = null;
        var slot: i64 = 0;
        while (slot < self.K) : (slot += 1) {
            const prefix_tokens = self.n_seq_tokens - slot;

            const x_prefix: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_4d(ctx, x, self.head_dim, self.n_head, prefix_tokens, self.n_seqs, x.nb[1], x.nb[2], x.nb[3], 0));
            const dt_prefix: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_3d(ctx, dt, self.n_head, prefix_tokens, self.n_seqs, dt.nb[1], dt.nb[2], 0));
            const B_prefix: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_4d(ctx, B, self.d_state, self.n_group, prefix_tokens, self.n_seqs, B.nb[1], B.nb[2], B.nb[3], 0));
            const C_prefix: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_4d(ctx, C, self.d_state, self.n_group, prefix_tokens, self.n_seqs, C.nb[1], C.nb[2], C.nb[3], 0));

            const prefix: *mod.c.ggml_tensor = mod.c.ggml_ssm_scan(ctx, s, x_prefix, dt_prefix, A, B_prefix, C_prefix, ids, 1);

            const full_offset: usize = @as(usize, @intCast(y_elems + slot * state_elems)) * mod.c.ggml_element_size(full);
            const prefix_offset: usize = @as(usize, @intCast(self.head_dim * self.n_head * prefix_tokens * self.n_seqs)) * mod.c.ggml_element_size(prefix);
            const full_state: *mod.c.ggml_tensor = mod.c.ggml_view_1d(ctx, full, state_elems, full_offset);
            const prefix_state: *mod.c.ggml_tensor = mod.c.ggml_view_1d(ctx, prefix, state_elems, prefix_offset);
            const diff: *mod.c.ggml_tensor = mod.c.ggml_sum(ctx, mod.c.ggml_sqr(ctx, mod.c.ggml_sub(ctx, full_state, prefix_state)));

            out = if (out) |o| mod.c.ggml_add(ctx, o, diff) else diff;
        }
        return out.?;
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
                ti.uniform(t, -1.0, -0.5);
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
