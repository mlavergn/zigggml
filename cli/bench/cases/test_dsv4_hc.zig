//! Port of `test_dsv4_hc` (tests/test-backend-ops.cpp @cd959bae9): the shared base of the GGML_OP_DSV4_HC_* cases.

const std = @import("std");
const mod = @import("module.zig");

/// The 32-bit Mersenne Twister (`std::mt19937`), so the seeded inputs match
/// upstream's bit for bit.
const Mt19937 = struct {
    const Self = @This();

    const n = 624;
    const m = 397;

    state: [n]u32 = undefined,
    index: usize = n,

    /// Seeds the generator (`std::mt19937(seed)`).
    fn init(seed: u32) Self {
        var self = Self{};
        self.state[0] = seed;
        for (1..n) |i| {
            const prev = self.state[i - 1];
            self.state[i] = 1812433253 *% (prev ^ (prev >> 30)) +% @as(u32, @intCast(i));
        }
        return self;
    }

    /// The next 32-bit output.
    fn next(self: *Self) u32 {
        if (self.index >= n) {
            for (0..n) |i| {
                const y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % n] & 0x7fffffff);
                var v = self.state[(i + m) % n] ^ (y >> 1);
                if (y & 1 != 0) v ^= 0x9908b0df;
                self.state[i] = v;
            }
            self.index = 0;
        }
        var y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// A float uniform in [`lo`, `hi`), as libstdc++'s
    /// `std::uniform_real_distribution<float>` draws it from one output.
    fn uniform(self: *Self, lo: f32, hi: f32) f32 {
        var r: f32 = @as(f32, @floatFromInt(self.next())) / 4294967296.0;
        if (r >= 1.0) r = std.math.nextAfter(f32, 1.0, 0.0);
        return r * (hi - lo) + lo;
    }
};

/// The shared state and input initialization of the DSV4 hyper-connection
/// cases (`TestDsv4HcComb`, `TestDsv4HcPre`, `TestDsv4HcPost`), which hold
/// one of these. Never a case on its own: it has no graph.
pub const TestDsv4Hc = struct {
    const Self = @This();

    /// The hyper-connection count.
    pub const hc: i64 = 4;

    // State.
    /// The output of the last graph built.
    out: ?*mod.c.ggml_tensor = null,

    /// An FNV-1a hash of `t`'s name and extents, seeding its inputs so they
    /// are the same on every run (`tensor_seed`).
    ///
    /// - `t`: the tensor.
    /// - Returns: the seed.
    pub fn tensorSeed(t: *const mod.c.ggml_tensor) u32 {
        var seed: u32 = 2166136261;
        for (std.mem.span(mod.c.ggml_get_name(t))) |ch| {
            seed ^= ch;
            seed *%= 16777619;
        }
        for (t.ne) |extent| {
            seed ^= @truncate(@as(u64, @bitCast(extent)));
            seed *%= 16777619;
        }
        return seed;
    }

    /// The value range for a named input (`tensor_range`).
    ///
    /// - `name`: the tensor's name.
    /// - Returns: the bounds `[lo, hi]`, or null for an unnamed input.
    pub fn tensorRange(name: []const u8) ?[2]f32 {
        if (std.mem.eql(u8, name, "mixes")) return .{ -2.0, 2.0 };
        if (std.mem.eql(u8, name, "scale")) return .{ -0.5, 0.5 };
        if (std.mem.eql(u8, name, "base")) return .{ -0.25, 0.25 };
        if (std.mem.eql(u8, name, "weights") or std.mem.eql(u8, name, "comb")) return .{ 0.0, 1.0 };
        if (std.mem.eql(u8, name, "post")) return .{ 0.0, 2.0 };
        if (std.mem.eql(u8, name, "gate")) return .{ -4.0, 4.0 };
        if (std.mem.eql(u8, name, "x") or std.mem.eql(u8, name, "residual")) return .{ -1.0, 1.0 };
        return null;
    }

    /// Fills each named input from a generator seeded by its name and shape,
    /// within its range, and anything else uniformly in [-1, 1).
    ///
    /// - `self`: the shared state.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            const range = Self.tensorRange(std.mem.span(mod.c.ggml_get_name(t))) orelse {
                ti.uniform(t, -1.0, 1.0);
                continue;
            };
            std.debug.assert(t.type == mod.c.GGML_TYPE_F32);
            var rng = Mt19937.init(Self.tensorSeed(t));
            const data = ti.allocator.alloc(f32, @intCast(mod.c.ggml_nelements(t))) catch @panic("OOM");
            defer ti.allocator.free(data);
            for (data) |*v| v.* = rng.uniform(range[0], range[1]);
            mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(f32));
        }
    }
};
