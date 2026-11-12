//! Port of `test_flash_attn_ext_kv_rows` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_FLASH_ATTN_EXT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_FLASH_ATTN_EXT with kv_rows: each q slice attends to its own list
/// of K/V cache rows.
pub const TestFlashAttnExtKvRows = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    hs: i64 = 256,
    /// K/V heads.
    nh: i64 = 2,
    gqa: i64 = 8,
    /// Rows of the K/V cache.
    n_cells: i64 = 2048,
    /// List length per slice.
    kv: i64 = 512,
    /// Queries per slice.
    nb: i64 = 8,
    n_slices: i64 = 4,
    type_kv: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("hs", self.hs);
        try v.add("nh", self.nh);
        try v.add("gqa", self.gqa);
        try v.add("n_cells", self.n_cells);
        try v.add("kv", self.kv);
        try v.add("nb", self.nb);
        try v.add("n_slices", self.n_slices);
        try v.add("type_KV", self.type_kv);
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `flash_attn_ext_rows(q, k, v, m, r)` over a shared K/V cache.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const q = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, self.hs, self.nb, self.nh * self.gqa, self.n_slices);
        mod.Graph.setName(q, "q");

        // Cache layout [hs, nh, n_cells], shared by all slices.
        const k0 = g.newTensor3d(ctx, self.type_kv, self.hs, self.nh, self.n_cells);
        const v0 = g.newTensor3d(ctx, self.type_kv, self.hs, self.nh, self.n_cells);
        mod.Graph.setName(k0, "k0");
        mod.Graph.setName(v0, "v0");

        const k: *mod.c.ggml_tensor = mod.c.ggml_permute(ctx, k0, 0, 2, 1, 3);
        const v: *mod.c.ggml_tensor = mod.c.ggml_permute(ctx, v0, 0, 2, 1, 3);

        const m = g.newTensor4d(ctx, mod.c.GGML_TYPE_F16, self.kv, self.nb, 1, self.n_slices);
        mod.Graph.setName(m, "m");

        const r = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.kv, self.n_slices);
        mod.Graph.setName(r, "r");

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.hs)));
        const out: *mod.c.ggml_tensor = mod.c.ggml_flash_attn_ext_rows(ctx, q, k, v, m, r, scale, 0.0, 0.0);
        _ = mod.c.ggml_prec_set_acc(out, mod.c.GGML_PREC_F32);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the row lists with distinct random cells per slice (padded with
    /// -1), the mask with values that hide the padding and a fixed pattern of
    /// dropped positions, and the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var prng: std.Random.DefaultPrng = .init(0x6B76);
        const random = prng.random();
        const kv: usize = @intCast(self.kv);
        const nb: usize = @intCast(self.nb);
        const n_slices: usize = @intCast(self.n_slices);
        const n_cells: usize = @intCast(self.n_cells);

        // Per slice: distinct random cells; the tail of the list is padding.
        const rows = ti.allocator.alloc(i32, kv * n_slices) catch @panic("OOM");
        defer ti.allocator.free(rows);
        @memset(rows, -1);
        const order = ti.allocator.alloc(i32, n_cells) catch @panic("OOM");
        defer ti.allocator.free(order);
        for (0..n_slices) |s| {
            for (order, 0..) |*o, i| o.* = @intCast(i);
            random.shuffle(i32, order);
            const n_used: usize = @min(n_cells, kv - (s * 37) % @max(1, kv / 2));
            for (0..n_used) |i| rows[s * kv + i] = order[i];
        }

        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.view_src != null) continue;
            const name = std.mem.sliceTo(&t.name, 0);
            if (std.mem.eql(u8, name, "r")) {
                mod.c.ggml_backend_tensor_set(t, rows.ptr, 0, rows.len * @sizeOf(i32));
            } else if (std.mem.eql(u8, name, "m")) {
                const f = ti.allocator.alloc(f32, @intCast(mod.c.ggml_nelements(t))) catch @panic("OOM");
                defer ti.allocator.free(f);
                for (0..n_slices) |s| {
                    for (0..nb) |j| {
                        for (0..kv) |i| {
                            const is_pad = rows[s * kv + i] < 0;
                            const drop = (i + 3 * j + 5 * s) % 11 == 0;
                            f[(s * nb + j) * kv + i] = if (is_pad or drop) -std.math.inf(f32) else -1.0 + 2.0 * random.float(f32);
                        }
                    }
                }
                const h = ti.allocator.alloc(mod.c.ggml_fp16_t, f.len) catch @panic("OOM");
                defer ti.allocator.free(h);
                mod.c.ggml_fp32_to_fp16_row(f.ptr, h.ptr, @intCast(f.len));
                mod.c.ggml_backend_tensor_set(t, h.ptr, 0, h.len * @sizeOf(mod.c.ggml_fp16_t));
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
