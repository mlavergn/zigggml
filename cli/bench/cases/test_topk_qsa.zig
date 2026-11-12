//! Port of `test_topk_qsa` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_TOP_K.

const std = @import("std");
const mod = @import("module.zig");

/// The qwen4exp QSA indexer top-k fusion: expands per-block scores to
/// cells, adds the f16 mask, then takes the top k.
pub const TestTopkQsa = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    n_blocks: i64 = 512,
    n_kv: i64 = 2048,
    n_tps: i64 = 2,
    n_stream: i64 = 1,
    width: i32 = 1500,

    // State.
    /// The output, the one node the fused comparison checks.
    out: [1]*mod.c.ggml_tensor = undefined,

    /// Always `TOPK_QSA`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("TOPK_QSA");
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "n_blocks", "n_kv", "n_tps", "n_stream", "width" });
    }

    /// The results must match exactly.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxErr(self: *Self) f64 {
        _ = self;
        return 0.0;
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `top_k(expand(score) + mask, width)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const score = g.newTensor3d(ctx, mod.c.GGML_TYPE_F32, self.n_blocks, self.n_tps, self.n_stream);
        mod.Graph.setName(score, "score");
        const cell_blk = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.n_kv, self.n_stream);
        mod.Graph.setName(cell_blk, "cell_blk");
        const kq_mask = g.newTensor3d(ctx, mod.c.GGML_TYPE_F16, self.n_kv, self.n_tps, self.n_stream);
        mod.Graph.setName(kq_mask, "kq_mask");

        const a: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_permute(ctx, score, 1, 0, 2, 3));
        var e: *mod.c.ggml_tensor = mod.c.ggml_get_rows(ctx, a, cell_blk);
        e = mod.c.ggml_cont(ctx, mod.c.ggml_permute(ctx, e, 1, 0, 2, 3));
        const m: *mod.c.ggml_tensor = mod.c.ggml_cast(ctx, kq_mask, mod.c.GGML_TYPE_F32);
        e = mod.c.ggml_add(ctx, e, mod.c.ggml_reshape_3d(ctx, m, self.n_kv, self.n_tps, self.n_stream));
        const out: *mod.c.ggml_tensor = mod.c.ggml_top_k(ctx, e, self.width);
        mod.Graph.setName(out, "out");
        self.out[0] = out;
        return out;
    }

    /// The output.
    ///
    /// - `self`: the case.
    /// - Returns: the nodes.
    pub fn fusionTestNodes(self: *Self) []const *mod.c.ggml_tensor {
        return &self.out;
    }

    /// A distinct mask ramp and small scores keep every cell value unique,
    /// so top-k has no ties.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.op != mod.c.GGML_OP_NONE) continue;
            const n: usize = @intCast(mod.c.ggml_nelements(t));
            if (t.type == mod.c.GGML_TYPE_I32) {
                const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*v| v.* = @intCast(ti.random.uintLessThan(u64, @intCast(self.n_blocks)));
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, n * @sizeOf(i32));
            } else if (t.type == mod.c.GGML_TYPE_F16) {
                const data = ti.allocator.alloc(mod.c.ggml_fp16_t, n) catch @panic("OOM");
                defer ti.allocator.free(data);
                const n_kv: usize = @intCast(self.n_kv);
                const nrows: usize = @intCast(mod.c.ggml_nrows(t));
                for (0..nrows) |r| {
                    for (0..n_kv) |i| data[r * n_kv + i] = mod.c.ggml_fp32_to_fp16(@floatFromInt(i));
                }
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, n * @sizeOf(mod.c.ggml_fp16_t));
            } else {
                ti.uniform(t, 0.0, 0.5);
            }
        }
    }

    /// The top-k output order is unspecified: compares as a set of indices.
    ///
    /// - `self`: the case.
    /// - `allocator`: scratch allocator.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: *Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        _ = self;
        const n = a.len;
        const fa = allocator.alloc(f32, n) catch @panic("OOM");
        defer allocator.free(fa);
        const fb = allocator.alloc(f32, n) catch @panic("OOM");
        defer allocator.free(fb);
        var diff: f64 = 0.0;
        for (0..n) |i| {
            fa[i] = @floatFromInt(Self.truncI32(a[i]));
            fb[i] = @floatFromInt(Self.truncI32(b[i]));
            diff += @abs(a[i] - fa[i]) + @abs(b[i] - fb[i]);
        }
        return diff + mod.Compare.jdst(allocator, fa, fb);
    }

    /// `(int32_t) x`, saturating where the C++ cast is undefined.
    fn truncI32(x: f32) i32 {
        if (std.math.isNan(x)) return 0;
        const lo: f32 = @floatFromInt(std.math.minInt(i32));
        const hi: f32 = 2147483520.0; // the largest f32 below 2^31
        return @intFromFloat(std.math.clamp(@trunc(x), lo, hi));
    }
};
