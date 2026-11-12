//! Port of `test_top_k` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_TOP_K.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_TOP_K of each row, optionally over rows with tied values.
pub const TestTopK = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 16, 10, 10, 10 },
    k: i32 = 4,
    ties: bool = false,

    // State.
    /// The input, saved for `err`.
    input: ?*mod.c.ggml_tensor = null,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "k", "ties" });
    }

    /// The results must match exactly.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxErr(self: *Self) f64 {
        _ = self;
        return 0.0;
    }

    /// With ties, only the final result is validated: `err` cannot handle
    /// the sentinels.
    ///
    /// - `self`: the case.
    /// - Returns: `ties`.
    pub fn runWholeGraph(self: *Self) bool {
        return self.ties;
    }

    /// Without ties, the same set of indices in any order; with ties, index
    /// sets whose input values match, without duplicate indices.
    ///
    /// - `self`: the case.
    /// - `allocator`: scratch allocator.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: *Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        const n = a.len;
        if (self.ties) {
            const input = self.input.?;
            const nel: usize = @intCast(mod.c.ggml_nelements(input));
            const src = allocator.alloc(f32, nel) catch @panic("OOM");
            defer allocator.free(src);
            mod.c.ggml_backend_tensor_get(input, src.ptr, 0, nel * mod.c.ggml_type_size(self.type.raw()));

            var diff: f64 = 0.0;
            const k: usize = @intCast(self.k);
            const nrows: usize = @intCast(mod.c.ggml_nrows(input));
            std.debug.assert(n == nrows * k);
            const cols: usize = @intCast(input.ne[0]);
            const ia = allocator.alloc(i32, k) catch @panic("OOM");
            defer allocator.free(ia);
            const ib = allocator.alloc(i32, k) catch @panic("OOM");
            defer allocator.free(ib);
            const asrc = allocator.alloc(f32, k) catch @panic("OOM");
            defer allocator.free(asrc);
            const bsrc = allocator.alloc(f32, k) catch @panic("OOM");
            defer allocator.free(bsrc);
            for (0..nrows) |r| {
                // Convert the row's indices back to integers.
                for (0..k) |col| {
                    ia[col] = Self.truncI32(a[r * k + col]);
                    ib[col] = Self.truncI32(b[r * k + col]);
                }
                // The source values of each row must match.
                for (0..k) |col| {
                    asrc[col] = src[r * cols + @as(usize, @intCast(ia[col]))];
                    bsrc[col] = src[r * cols + @as(usize, @intCast(ib[col]))];
                }
                diff += mod.Compare.jdst(allocator, asrc, bsrc);
                // No duplicate indices.
                std.mem.sort(i32, ia, {}, std.sort.asc(i32));
                std.mem.sort(i32, ib, {}, std.sort.asc(i32));
                if (Self.hasAdjacentDuplicate(ia)) diff += 1;
                if (Self.hasAdjacentDuplicate(ib)) diff += 1;
            }
            return diff;
        }

        const fa = allocator.alloc(f32, n) catch @panic("OOM");
        defer allocator.free(fa);
        const fb = allocator.alloc(f32, n) catch @panic("OOM");
        defer allocator.free(fb);
        var diff: f64 = 0.0;
        for (0..n) |i| {
            const ia = Self.truncI32(a[i]);
            const ib = Self.truncI32(b[i]);
            // Penalize data that is not integer valued.
            diff += @abs(a[i] - @as(f32, @floatFromInt(ia)));
            diff += @abs(b[i] - @as(f32, @floatFromInt(ib)));
            fa[i] = @floatFromInt(ia);
            fb[i] = @floatFromInt(ib);
        }
        return diff + mod.Compare.jdst(allocator, fa, fb);
    }

    /// Builds `top_k(a, k)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        // Saved for err().
        self.input = a;
        const out: *mod.c.ggml_tensor = mod.c.ggml_top_k(ctx, a, self.k);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills each row with shuffled distinct values, or with duplicates
    /// when `ties`.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        const tie_denom: i64 = @max(1, @min(10, @divTrunc(self.k, 2)));
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            const n: usize = @intCast(t.ne[0]);
            const data = ti.allocator.alloc(f32, n) catch @panic("OOM");
            defer ti.allocator.free(data);
            var r: i64 = 0;
            while (r < mod.c.ggml_nrows(t)) : (r += 1) {
                for (data, 0..) |*d, i| {
                    const iv: i64 = @intCast(i);
                    // Integer division introduces duplicates.
                    d.* = @floatFromInt(if (self.ties) @divTrunc(iv, tie_denom) else iv);
                }
                ti.random.shuffle(f32, data);
                mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], n * @sizeOf(f32));
            }
        }
    }

    /// `(int32_t) x`, saturating where the C++ cast is undefined.
    fn truncI32(x: f32) i32 {
        if (std.math.isNan(x)) return 0;
        const lo: f32 = @floatFromInt(std.math.minInt(i32));
        const hi: f32 = 2147483520.0; // the largest f32 below 2^31
        return @intFromFloat(std.math.clamp(@trunc(x), lo, hi));
    }

    /// Whether sorted `v` holds two equal neighbors (`std::adjacent_find`).
    fn hasAdjacentDuplicate(v: []const i32) bool {
        if (v.len < 2) return false;
        for (v[0 .. v.len - 1], v[1..]) |x, y| {
            if (x == y) return true;
        }
        return false;
    }
};
