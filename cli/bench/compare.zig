const std = @import("std");
const mod = @import("module.zig");

/// Reads tensors back as floats and measures how far two results are apart:
/// the ports of `test-backend-ops`' `tensor_to_float`, `nmse`, `jdst` and
/// `isinf_or_max`.
pub const Compare = struct {
    const Self = @This();

    /// The default error bound (`test_case::max_nmse_err`).
    pub const default_max_nmse_err: f64 = 1e-7;

    /// Reads `t` from its backend and converts every element to `f32`,
    /// walking by index so views with gaps read correctly (`tensor_to_float`).
    ///
    /// - `allocator`: allocator for the result and a staging buffer.
    /// - `t`: the tensor.
    /// - Returns: the elements, owned by the caller.
    pub fn tensorToFloat(allocator: std.mem.Allocator, t: *const mod.c.ggml_tensor) ![]f32 {
        const nbytes = mod.c.ggml_nbytes(t);
        const buf = try allocator.alloc(u8, nbytes);
        defer allocator.free(buf);
        mod.c.ggml_backend_tensor_get(t, buf.ptr, 0, nbytes);

        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(allocator);
        try out.ensureTotalCapacity(allocator, @intCast(mod.c.ggml_nelements(t)));

        const traits = mod.c.ggml_get_type_traits(t.type);
        const bs: usize = @intCast(mod.c.ggml_blck_size(t.type));
        const vq = try allocator.alloc(f32, bs);
        defer allocator.free(vq);
        const converts = mod.c.ggml_is_quantized(t.type) or t.type == mod.c.GGML_TYPE_F8_E4M3;

        var k3: usize = 0;
        while (k3 < t.ne[3]) : (k3 += 1) {
            var k2: usize = 0;
            while (k2 < t.ne[2]) : (k2 += 1) {
                var k1: usize = 0;
                while (k1 < t.ne[1]) : (k1 += 1) {
                    var k0: usize = 0;
                    while (k0 < t.ne[0]) : (k0 += bs) {
                        const i = k3 * t.nb[3] + k2 * t.nb[2] + k1 * t.nb[1] + k0 / bs * t.nb[0];
                        const p = buf[i..];
                        switch (t.type) {
                            mod.c.GGML_TYPE_F16 => out.appendAssumeCapacity(mod.c.ggml_fp16_to_fp32(std.mem.readInt(u16, p[0..2], .little))),
                            mod.c.GGML_TYPE_BF16 => out.appendAssumeCapacity(mod.c.ggml_bf16_to_fp32(.{ .bits = std.mem.readInt(u16, p[0..2], .little) })),
                            mod.c.GGML_TYPE_F32 => out.appendAssumeCapacity(@bitCast(std.mem.readInt(u32, p[0..4], .little))),
                            mod.c.GGML_TYPE_I64 => out.appendAssumeCapacity(@floatFromInt(std.mem.readInt(i64, p[0..8], .little))),
                            mod.c.GGML_TYPE_I32 => out.appendAssumeCapacity(@floatFromInt(std.mem.readInt(i32, p[0..4], .little))),
                            mod.c.GGML_TYPE_I16 => out.appendAssumeCapacity(@floatFromInt(std.mem.readInt(i16, p[0..2], .little))),
                            mod.c.GGML_TYPE_I8 => out.appendAssumeCapacity(@floatFromInt(@as(i8, @bitCast(p[0])))),
                            else => {
                                if (!converts) return error.UnsupportedType;
                                traits.*.to_float.?(p.ptr, vq.ptr, @intCast(bs));
                                try out.appendSlice(allocator, vq);
                            },
                        }
                    }
                }
            }
        }
        return out.toOwnedSlice(allocator);
    }

    /// Normalized mean squared error, mse(a, b) / mse(a, 0) (`nmse`).
    ///
    /// - `a`: the reference values.
    /// - `b`: the values under test, as many as `a`.
    /// - Returns: the error.
    pub fn nmse(a: []const f32, b: []const f32) f64 {
        var mse_a_b: f64 = 0.0;
        var mse_a_0: f64 = 0.0;
        for (a, b) |a_i, b_i| {
            mse_a_b += (a_i - b_i) * (a_i - b_i);
            mse_a_0 += a_i * a_i;
        }
        return mse_a_b / mse_a_0;
    }

    /// The Jaccard distance between `a` and `b` taken as multisets: 0 for the
    /// same values, 1 for none in common (`jdst`).
    ///
    /// - `allocator`: allocator for the counts, freed before returning.
    /// - `a`: one set of values.
    /// - `b`: the other, as many as `a`.
    /// - Returns: the distance.
    pub fn jdst(allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        var counts: std.AutoHashMapUnmanaged(u32, i64) = .empty;
        defer counts.deinit(allocator);
        for (a) |x| {
            const e = counts.getOrPutValue(allocator, @bitCast(x), 0) catch @panic("OOM");
            e.value_ptr.* += 1;
        }
        for (b) |x| {
            const e = counts.getOrPutValue(allocator, @bitCast(x), 0) catch @panic("OOM");
            e.value_ptr.* -= 1;
        }
        var diff: u64 = 0;
        var it = counts.valueIterator();
        while (it.next()) |v| diff += @abs(v.*);
        return @as(f64, @floatFromInt(diff)) / @as(f64, @floatFromInt(2 * a.len));
    }

    /// Whether `f` is infinite or ±`FLT_MAX`, which upstream counts as
    /// infinity (`isinf_or_max`).
    ///
    /// - `f`: the value.
    /// - Returns: true for ±inf or ±FLT_MAX.
    pub fn isInfOrMax(f: f32) bool {
        return std.math.isInf(f) or f == std.math.floatMax(f32) or f == -std.math.floatMax(f32);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "nmse is zero for equal inputs and grows with the difference" {
    const a = [_]f32{ 1, 2, 3 };
    try std.testing.expectEqual(@as(f64, 0), Compare.nmse(&a, &a));
    const b = [_]f32{ 1, 2, 4 };
    try std.testing.expect(Compare.nmse(&a, &b) > 0);
}

test "jdst counts values, not positions" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 4, 3, 2, 1 };
    try std.testing.expectEqual(@as(f64, 0), Compare.jdst(std.testing.allocator, &a, &b));
    const c = [_]f32{ 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(f64, 1), Compare.jdst(std.testing.allocator, &a, &c));
}
