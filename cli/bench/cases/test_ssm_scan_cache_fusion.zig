//! Port of `test_ssm_scan_cache_fusion` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SSM_SCAN + GGML_OP_CPY.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SSM_SCAN whose snapshot states are copied into a recurrent
/// cache, which a backend may fuse into the scan. Compared as one graph.
pub const TestSsmScanCacheFusion = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    d_state: i64 = 128,
    head_dim: i64 = 64,
    n_head: i64 = 16,
    n_group: i64 = 2,
    n_seq_tokens: i64 = 4,
    n_seqs: i64 = 1,
    /// Snapshot slot count (1 = final state only).
    K: i64 = 4,

    // State.
    /// The cache copy, the node the fused comparison checks.
    fusion_nodes: [1]*mod.c.ggml_tensor = undefined,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "d_state", "head_dim", "n_head", "n_group", "n_seq_tokens", "n_seqs", "K" });
    }

    /// Always `SSM_SCAN_CACHE_FUSION`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("SSM_SCAN_CACHE_FUSION");
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

    /// Builds `sum(cont(cpy(snapshots(ssm_scan(...)), cache)))`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const D = self.d_state * self.head_dim * self.n_head;
        const n_written = @min(self.n_seq_tokens, self.K);

        // More cache rows per slot than seqs and a non-zero first row, so a
        // wrong slot stride or offset shows up.
        const mem_size = self.n_seqs + 2;
        const kv_head: usize = 1;

        const s = g.newTensor4d(ctx, self.type, self.d_state, self.head_dim, self.n_head, self.n_seqs);
        const x = g.newTensor4d(ctx, self.type, self.head_dim, self.n_head, self.n_seq_tokens, self.n_seqs);
        const dt = g.newTensor3d(ctx, self.type, self.n_head, self.n_seq_tokens, self.n_seqs);
        const A = g.newTensor2d(ctx, self.type, 1, self.n_head);
        const B = g.newTensor4d(ctx, self.type, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs);
        const C = g.newTensor4d(ctx, self.type, self.d_state, self.n_group, self.n_seq_tokens, self.n_seqs);
        const ids = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, self.n_seqs);
        mod.Graph.setName(A, "A");
        mod.Graph.setName(ids, "ids");

        const out: *mod.c.ggml_tensor = mod.c.ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, self.K);
        mod.Graph.setName(out, "ssm_out");

        // Snapshot tail view [D, n_seqs, n_written].
        const src: *mod.c.ggml_tensor = mod.c.ggml_view_3d(ctx, out, D, self.n_seqs, n_written, mod.c.ggml_row_size(out.type, D), mod.c.ggml_row_size(out.type, D * self.n_seqs), mod.c.ggml_row_size(out.type, mod.c.ggml_nelements(x)));

        // Recurrent cache view [D, n_seqs, n_written].
        const cache = g.newTensor2d(ctx, self.type, D, mem_size * n_written);
        mod.Graph.setName(cache, "cache");
        const dst: *mod.c.ggml_tensor = mod.c.ggml_view_3d(ctx, cache, D, self.n_seqs, n_written, cache.nb[1], @as(usize, @intCast(mem_size)) * cache.nb[1], kv_head * cache.nb[1]);

        const cpy: *mod.c.ggml_tensor = mod.c.ggml_cpy(ctx, src, dst);
        mod.Graph.setName(cpy, "ssm_cache_cpy");
        self.fusion_nodes = .{cpy};

        // Read the cpy output so neither the scan nor the cpy is the graph
        // output (cont, since the cache view is strided).
        return mod.c.ggml_sum(ctx, mod.c.ggml_cont(ctx, cpy));
    }

    /// Fills `ids` with `0..n_seqs`, `A` with a negative decay, the cache
    /// with zeros, and the rest uniformly in [-1, 1).
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
            if (std.mem.eql(u8, name, "ids")) {
                const n: usize = @intCast(t.ne[0]);
                const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data, 0..) |*d, i| d.* = @intCast(i);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, n * @sizeOf(i32));
            } else if (std.mem.eql(u8, name, "A")) {
                ti.uniform(t, -1.0, -0.5);
            } else if (std.mem.eql(u8, name, "cache")) {
                ti.uniform(t, 0.0, 0.0);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
