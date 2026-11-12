//! Port of `test_top_k_mask` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_TOP_K_MASK.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_TOP_K_MASK: keeps the mask at the cells of the top-k picks (cells
/// or blocks of cells, plus extra cells) and sets the rest to -inf.
pub const TestTopKMask = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    n_kv: i64 = 256,
    /// Cells per block; 0 for no blocks, where a pick is a cell.
    n_blk: i64 = 4,
    /// Number of blocks; replaced by `n_kv` in `setup` without blocks.
    n_pick: i64 = 64,
    n_top: i64 = 16,
    /// Extra cells per row; 0 for none.
    n_extra: i64 = 3,
    /// Whether picks have scores; without, every pick is live.
    score: bool = true,
    /// Rows per stream.
    nb: i64 = 32,
    /// Rows of top_k per mask row, broadcast over ne2.
    nh: i64 = 1,
    /// Streams.
    ns: i64 = 1,
    /// Blocks of consecutive cells and live picks, as in a real cache.
    real: bool = false,
    /// One mask row for all rows and streams.
    bcast: bool = false,

    /// Without blocks, a pick is a cell: `n_pick` becomes `n_kv`.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        if (self.n_blk == 0) self.n_pick = self.n_kv;
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "n_kv", "n_blk", "n_pick", "n_top", "n_extra", "score", "nb", "nh", "ns", "real", "bcast" });
    }

    /// The result is a copy of the mask or -inf; the finite values must
    /// match exactly.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxErr(self: *Self) f64 {
        _ = self;
        return 0.0;
    }

    /// The largest absolute difference where both values are finite.
    ///
    /// - `self`: the case.
    /// - `allocator`: scratch allocator, unused.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: *Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        _ = self;
        _ = allocator;
        var result: f64 = 0.0;
        for (a, b) |x, y| {
            if (std.math.isFinite(x) and std.math.isFinite(y)) result = @max(result, @as(f64, @abs(x - y)));
        }
        return result;
    }

    /// The output plus, per row, the picks, scores, block cells and extra
    /// cells read, with the mask and output values each touches.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the size.
    pub fn opSize(self: *Self, t: *mod.c.ggml_tensor) usize {
        const ts = mod.c.ggml_type_size(self.type.raw());
        const nc: usize = @intCast(@max(self.n_blk, 1));
        const rows: usize = @intCast(self.nb * self.nh * self.ns);
        const n_top: usize = @intCast(self.n_top);
        const n_extra: usize = @intCast(self.n_extra);
        const score_size: usize = if (self.score) @sizeOf(f32) else 0;
        return mod.c.ggml_nbytes(t) + rows * (n_top * (@sizeOf(i32) + score_size + nc * (@sizeOf(i32) + 2 * ts)) + n_extra * (@sizeOf(i32) + 2 * ts));
    }

    /// Builds `top_k_mask(m, top_k, score, blocks, extra)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const m = if (self.bcast) g.newTensor1d(ctx, self.type, self.n_kv) else g.newTensor4d(ctx, self.type, self.n_kv, self.nb, 1, self.ns);
        mod.Graph.setName(m, "m");

        const top_k = g.newTensor4d(ctx, mod.c.GGML_TYPE_I32, self.n_top, self.nb, self.nh, self.ns);
        mod.Graph.setName(top_k, "top_k");

        var s: ?*mod.c.ggml_tensor = null;
        if (self.score) {
            const st = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, self.n_pick, self.nb, self.nh, self.ns);
            mod.Graph.setName(st, "score");
            s = st;
        }

        var blocks: ?*mod.c.ggml_tensor = null;
        if (self.n_blk > 0) {
            const bt = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.n_blk, self.n_pick);
            mod.Graph.setName(bt, "blocks");
            blocks = bt;
        }

        var extra: ?*mod.c.ggml_tensor = null;
        if (self.n_extra > 0) {
            const et = g.newTensor4d(ctx, mod.c.GGML_TYPE_I32, self.n_extra, self.nb, self.nh, self.ns);
            mod.Graph.setName(et, "extra");
            extra = et;
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_top_k_mask(ctx, m, top_k, s, blocks, extra);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the mask, scores (some -inf), top-k picks (some out of range),
    /// and block and extra cells (some out of range, which the op skips).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var prng: std.Random.DefaultPrng = .init(0x4B50);
        const random = prng.random();

        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            const name = std.mem.sliceTo(&t.name, 0);
            const nels: usize = @intCast(mod.c.ggml_nelements(t));
            if (std.mem.eql(u8, name, "m")) {
                if (self.type.raw() == mod.c.GGML_TYPE_F16) {
                    ti.kqMask(t, -1.0, 1.0);
                } else {
                    const data = ti.allocator.alloc(f32, nels) catch @panic("OOM");
                    defer ti.allocator.free(data);
                    for (data) |*v| v.* = if (random.int(u32) % 4 == 0) -std.math.inf(f32) else -1.0 + 2.0 * random.float(f32);
                    mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(f32));
                }
            } else if (std.mem.eql(u8, name, "score")) {
                // -inf marks the picks a row cannot see.
                const data = ti.allocator.alloc(f32, nels) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*v| v.* = if (!self.real and random.int(u32) % 4 == 0) -std.math.inf(f32) else -1.0 + 2.0 * random.float(f32);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(f32));
            } else if (std.mem.eql(u8, name, "top_k")) {
                const n_pick: usize = @intCast(self.n_pick);
                const n_top: usize = @intCast(self.n_top);
                const data = ti.allocator.alloc(i32, nels) catch @panic("OOM");
                defer ti.allocator.free(data);
                const order = ti.allocator.alloc(i32, n_pick) catch @panic("OOM");
                defer ti.allocator.free(order);
                const nrows: usize = @intCast(mod.c.ggml_nrows(t));
                for (0..nrows) |r| {
                    for (order, 0..) |*o, i| o.* = @intCast(i);
                    random.shuffle(i32, order);
                    for (0..n_top) |j| {
                        const is_out = !self.real and j == 0 and r % 5 == 0;
                        data[r * n_top + j] = if (is_out) (if (r % 2 != 0) -1 else @as(i32, @intCast(n_pick))) else order[j % n_pick];
                    }
                }
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(i32));
            } else if (std.mem.eql(u8, name, "blocks") or std.mem.eql(u8, name, "extra")) {
                self.initCells(ti, random, t);
            } else {
                // The sentinels and the output; the callback checks the sentinels for NaN.
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }

    /// Fills a block or extra cell tensor: consecutive cells (and the last
    /// cells for extras) when `real`, else random cells with some -1 or
    /// `n_kv`, out of range.
    fn initCells(self: *Self, ti: *mod.TensorInit, random: std.Random, t: *mod.c.ggml_tensor) void {
        const data = ti.allocator.alloc(i32, @intCast(mod.c.ggml_nelements(t))) catch @panic("OOM");
        defer ti.allocator.free(data);
        const n_kv: i32 = @intCast(self.n_kv);
        if (self.real) {
            // Block p holds cells p*n_blk..p*n_blk + n_blk - 1; the extra cells are the last cells.
            const is_blocks = std.mem.eql(u8, std.mem.sliceTo(&t.name, 0), "blocks");
            for (data, 0..) |*d, i| {
                const idx: i64 = @intCast(i);
                const cell: i64 = if (is_blocks) idx else self.n_kv - 1 - @mod(idx, t.ne[0]);
                d.* = if (cell < self.n_kv) @intCast(cell) else n_kv;
            }
        } else {
            for (data) |*d| {
                const r = random.int(u32) % 16;
                d.* = if (r == 0) n_kv else if (r == 1) -1 else random.intRangeAtMost(i32, 0, n_kv - 1);
            }
        }
        mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(i32));
    }
};
