const std = @import("std");
const mod = @import("module.zig");

/// Fills a case's input tensors with data: the ports of `test-backend-ops`'
/// `init_tensor_uniform`, `init_tensor_kq_mask`, `init_tensor_kq_mask_sparse`
/// and `init_tensor_tril`.
///
/// Upstream draws from `std::random_device`; this draws from one PRNG seeded
/// per run, so values differ from upstream's but have the same distribution.
/// Quantization runs on one thread, where upstream spreads it over several.
pub const TensorInit = struct {
    const Self = @This();

    /// Allocator for the staging buffers.
    allocator: std.mem.Allocator,
    /// The value source.
    random: std.Random,

    /// Binds a value source.
    ///
    /// - `allocator`: allocator for staging buffers, freed before each call returns.
    /// - `random`: the PRNG to draw from.
    /// - Returns: the initializer.
    pub fn init(allocator: std.mem.Allocator, random: std.Random) Self {
        return Self{ .allocator = allocator, .random = random };
    }

    /// Fills every tensor in `ctx` uniformly in [-1, 1): the default
    /// `test_case::initialize_tensors`.
    ///
    /// - `self`: the initializer.
    /// - `ctx`: the context whose tensors are filled.
    pub fn all(self: *Self, ctx: *mod.c.ggml_context) void {
        self.allRange(ctx, -1.0, 1.0);
    }

    /// Fills every tensor in `ctx` uniformly in [`min`, `max`).
    ///
    /// - `self`: the initializer.
    /// - `ctx`: the context whose tensors are filled.
    /// - `min`: the lower bound.
    /// - `max`: the upper bound.
    pub fn allRange(self: *Self, ctx: *mod.c.ggml_context, min: f32, max: f32) void {
        var t: ?*mod.c.ggml_tensor = mod.c.ggml_get_first_tensor(ctx);
        while (t) |tensor| : (t = mod.c.ggml_get_next_tensor(ctx, tensor)) {
            self.uniform(tensor, min, max);
        }
    }

    /// A value uniform in [`min`, `max`).
    ///
    /// - `self`: the initializer.
    /// - `min`: the lower bound.
    /// - `max`: the upper bound.
    /// - Returns: the value.
    pub fn float(self: *Self, min: f32, max: f32) f32 {
        return min + (max - min) * self.random.float(f32);
    }

    /// Fills `tensor` uniformly in [`min`, `max`), converting or quantizing to
    /// its type (`init_tensor_uniform`).
    ///
    /// Integer tensors receive the float bit patterns, as upstream's do. An
    /// optional importance matrix is used or not at random, so both
    /// quantization paths are exercised.
    ///
    /// - `self`: the initializer.
    /// - `tensor`: the tensor, already allocated in a backend buffer.
    /// - `min`: the lower bound.
    /// - `max`: the upper bound.
    pub fn uniform(self: *Self, tensor: *mod.c.ggml_tensor, min: f32, max: f32) void {
        if (mod.c.ggml_is_empty(tensor)) return;
        const nels: usize = @intCast(mod.c.ggml_nelements(tensor));
        const data = self.allocator.alloc(f32, nels) catch @panic("OOM");
        defer self.allocator.free(data);
        for (data) |*x| x.* = self.float(min, max);

        const ty = tensor.type;
        if (ty == mod.c.GGML_TYPE_F32 or ty == mod.c.GGML_TYPE_I32) {
            mod.c.ggml_backend_tensor_set(tensor, data.ptr, 0, nels * @sizeOf(f32));
        } else if (mod.c.ggml_is_quantized(ty) or ty == mod.c.GGML_TYPE_F8_E4M3 or ty == mod.c.GGML_TYPE_F16 or ty == mod.c.GGML_TYPE_BF16) {
            const blck: usize = @intCast(mod.c.ggml_blck_size(ty));
            std.debug.assert(nels % blck == 0);
            // A dummy importance matrix; optional ones are used half the time.
            const imatrix = self.allocator.alloc(f32, @intCast(tensor.ne[0])) catch @panic("OOM");
            defer self.allocator.free(imatrix);
            @memset(imatrix, 1.0);
            var im: ?[*]const f32 = imatrix.ptr;
            if (!mod.c.ggml_quantize_requires_imatrix(ty) and data[0] > 0.5 * (min + max)) im = null;

            const dataq = self.allocator.alloc(u8, mod.c.ggml_row_size(ty, @intCast(nels))) catch @panic("OOM");
            defer self.allocator.free(dataq);
            _ = mod.c.ggml_quantize_chunk(ty, data.ptr, dataq.ptr, 0, @intCast(nels / blck), @intCast(blck), im);
            mod.c.ggml_backend_tensor_set(tensor, dataq.ptr, 0, dataq.len);
        } else if (ty == mod.c.GGML_TYPE_I8 or ty == mod.c.GGML_TYPE_I16) {
            mod.c.ggml_backend_tensor_set(tensor, data.ptr, 0, nels * mod.c.ggml_type_size(ty));
        } else if (ty == mod.c.GGML_TYPE_I64) {
            // Mirrors the float data into both halves, as upstream does.
            const half = nels * @sizeOf(f32);
            mod.c.ggml_backend_tensor_set(tensor, data.ptr, 0, half);
            mod.c.ggml_backend_tensor_set(tensor, data.ptr, half, half);
        } else {
            std.debug.panic("TensorInit.uniform: unsupported type {s}", .{std.mem.span(mod.c.ggml_type_name(ty))});
        }
    }

    /// Fills an F16 mask uniformly, with about a fifth of it in random
    /// 128x64 blocks of `-inf` or `0` (`init_tensor_kq_mask`).
    ///
    /// - `self`: the initializer.
    /// - `tensor`: an F16 tensor.
    /// - `min`: the lower bound.
    /// - `max`: the upper bound.
    pub fn kqMask(self: *Self, tensor: *mod.c.ggml_tensor, min: f32, max: f32) void {
        std.debug.assert(tensor.type == mod.c.GGML_TYPE_F16);
        const ne0: usize = @intCast(tensor.ne[0]);
        const ne1: usize = @intCast(tensor.ne[1]);
        const ne2: usize = @intCast(tensor.ne[2]);
        const ne3: usize = @intCast(tensor.ne[3]);
        const n = ne0 * ne1 * ne2 * ne3;
        const data = self.allocator.alloc(f32, n) catch @panic("OOM");
        defer self.allocator.free(data);
        for (data) |*x| x.* = self.float(min, max);

        const blck0 = 128;
        const blck1 = 64;
        const n_blocks: usize = @intFromFloat(0.2 * @as(f64, @floatFromInt(n)) / (blck0 * blck1));
        for (0..n_blocks) |_| {
            const p3 = self.random.uintLessThan(usize, ne3);
            const p2 = self.random.uintLessThan(usize, ne2);
            const p1 = self.random.uintLessThan(usize, ne1);
            const p0 = self.random.uintLessThan(usize, ne0);
            const inf = self.random.boolean();
            var k1: usize = 0;
            while (k1 < blck1 and p1 + k1 < ne1) : (k1 += 1) {
                const idx = p3 * ne2 * ne1 * ne0 + p2 * ne1 * ne0 + (p1 + k1) * ne0 + p0;
                var k0: usize = 0;
                while (k0 < blck0 and p0 + k0 < ne0) : (k0 += 1) {
                    data[idx + k0] = if (inf) -std.math.inf(f32) else 0.0;
                }
            }
        }
        self.setF16(tensor, data);
    }

    /// Fills an F16 mask that admits at most `n_kv_max` positions per row,
    /// the rest `-inf` (`init_tensor_kq_mask_sparse`). Deterministic, as
    /// upstream's is.
    ///
    /// - `self`: the initializer.
    /// - `tensor`: an F16 tensor.
    /// - `n_kv_max`: the most admitted positions per row, in 1..ne[0].
    pub fn kqMaskSparse(self: *Self, tensor: *mod.c.ggml_tensor, n_kv_max: i64) void {
        std.debug.assert(tensor.type == mod.c.GGML_TYPE_F16);
        std.debug.assert(n_kv_max > 0 and n_kv_max <= tensor.ne[0]);
        const ne0: usize = @intCast(tensor.ne[0]);
        const nrows: usize = @intCast(mod.c.ggml_nrows(tensor));
        const data = self.allocator.alloc(f32, ne0 * nrows) catch @panic("OOM");
        defer self.allocator.free(data);
        @memset(data, -std.math.inf(f32));
        const order = self.allocator.alloc(i32, ne0) catch @panic("OOM");
        defer self.allocator.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);

        var prng: std.Random.DefaultPrng = .init(0x5A17);
        const random = prng.random();
        const kv_max: usize = @intCast(n_kv_max);
        for (0..nrows) |row| {
            random.shuffle(i32, order);
            const count = kv_max - row % @min(kv_max, 17);
            std.mem.sort(i32, order[0..count], {}, std.sort.asc(i32));
            for (0..count) |i| {
                const k: f32 = @floatFromInt(1 + (i + row) % 7);
                data[row * ne0 + @as(usize, @intCast(order[i]))] = -0.03125 * k;
            }
        }
        self.setF16(tensor, data);
    }

    /// Fills a square F32 tensor's lower triangle (diagonal included)
    /// uniformly and the rest with zero (`init_tensor_tril`).
    ///
    /// - `self`: the initializer.
    /// - `tensor`: an F32 tensor with ne[0] == ne[1].
    /// - `min`: the lower bound.
    /// - `max`: the upper bound.
    pub fn tril(self: *Self, tensor: *mod.c.ggml_tensor, min: f32, max: f32) void {
        std.debug.assert(tensor.type == mod.c.GGML_TYPE_F32);
        std.debug.assert(tensor.ne[0] == tensor.ne[1]);
        const nbytes = mod.c.ggml_nbytes(tensor);
        const data = self.allocator.alloc(f32, nbytes / @sizeOf(f32)) catch @panic("OOM");
        defer self.allocator.free(data);
        @memset(data, 0.0);
        var k3: i64 = 0;
        while (k3 < tensor.ne[3]) : (k3 += 1) {
            var k2: i64 = 0;
            while (k2 < tensor.ne[2]) : (k2 += 1) {
                var k1: i64 = 0;
                while (k1 < tensor.ne[1]) : (k1 += 1) {
                    var k0: i64 = 0;
                    while (k0 < tensor.ne[0]) : (k0 += 1) {
                        const offset = @as(usize, @intCast(k0)) * tensor.nb[0] + @as(usize, @intCast(k1)) * tensor.nb[1] +
                            @as(usize, @intCast(k2)) * tensor.nb[2] + @as(usize, @intCast(k3)) * tensor.nb[3];
                        data[offset / @sizeOf(f32)] = if (k0 <= k1) self.float(min, max) else 0.0;
                    }
                }
            }
        }
        mod.c.ggml_backend_tensor_set(tensor, data.ptr, 0, nbytes);
    }

    /// Converts `data` to F16 and uploads it to `tensor`.
    fn setF16(self: *Self, tensor: *mod.c.ggml_tensor, data: []const f32) void {
        const half = self.allocator.alloc(mod.c.ggml_fp16_t, data.len) catch @panic("OOM");
        defer self.allocator.free(half);
        mod.c.ggml_fp32_to_fp16_row(data.ptr, half.ptr, @intCast(data.len));
        mod.c.ggml_backend_tensor_set(tensor, half.ptr, 0, half.len * @sizeOf(mod.c.ggml_fp16_t));
    }
};
