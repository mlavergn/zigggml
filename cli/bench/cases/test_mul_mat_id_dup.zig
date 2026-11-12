//! Port of `test_mul_mat_id_dup` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT_ID.

const std = @import("std");
const mod = @import("module.zig");

/// `std::default_random_engine` (libstdc++'s `minstd_rand0`), so the ids
/// match upstream's fixed-seed draw.
const MinstdRand0 = struct {
    state: u64,

    /// Seeds the engine.
    ///
    /// - `seed`: the seed.
    /// - Returns: the engine.
    fn init(seed: u64) MinstdRand0 {
        const s = seed % 2147483647;
        return .{ .state = if (s == 0) 1 else s };
    }

    /// The next value, in [1, 2^31 - 2].
    ///
    /// - `self`: the engine.
    /// - Returns: the value.
    fn next(self: *MinstdRand0) u64 {
        self.state = self.state * 16807 % 2147483647;
        return self.state;
    }
};

/// GGML_OP_MUL_MAT_ID with expert ids repeated within a token's row:
/// `test_mul_mat_id` by composition.
pub const TestMulMatIdDup = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults (the base's).
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    n_mats: i32 = 8,
    n_used: i32 = 2,
    /// Broadcast the b matrix.
    b: bool = false,
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 32,
    /// Rows of `as` in memory: the experts of `as` are strided for m_v > m, no view for 0.
    m_v: i64 = 0,
    ids_offset: bool = false,
    /// Set some ids to a negative value.
    skip: bool = false,

    // State.
    /// The base case, built from the parameters in `setup`.
    base: mod.TestMulMatId = .{},

    /// Builds the base case.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.base = .{
            .type_a = self.type_a,
            .type_b = self.type_b,
            .n_mats = self.n_mats,
            .n_used = self.n_used,
            .b = self.b,
            .m = self.m,
            .n = self.n,
            .k = self.k,
            .m_v = self.m_v,
            .ids_offset = self.ids_offset,
            .skip = self.skip,
        };
        self.base.setup();
    }

    /// Writes the base's parameters, then `dup=1`.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try self.base.vars(w);
        try w.writeAll(",dup=1");
    }

    /// The base's bound.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        return self.base.maxNmseErrBackend(backend);
    }

    /// The base's FLOP count.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        return self.base.opFlops(t);
    }

    /// Builds the base graph.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        return self.base.buildGraph(g);
    }

    /// Fills the ids with repeated experts, mostly expert 0, from a fixed
    /// seed (the last row always uses the last expert), and the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var rng = MinstdRand0.init(1234);
        const n_mats: u64 = @intCast(self.n_mats);
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type != mod.c.GGML_TYPE_I32) {
                ti.uniform(t, -1.0, 1.0);
                continue;
            }
            if (mod.Helpers.isViewOp(t.op)) continue;
            const ne0: usize = @intCast(t.ne[0]);
            const data = ti.allocator.alloc(i32, ne0) catch @panic("OOM");
            defer ti.allocator.free(data);
            const nrows = mod.c.ggml_nrows(t);
            var r: i64 = 0;
            while (r < nrows) : (r += 1) {
                for (data) |*d| {
                    d.* = if (rng.next() % 4 == 0) @intCast(rng.next() % n_mats) else 0;
                }
                if (r == nrows - 1) data[if (self.ids_offset) 1 else 0] = self.n_mats - 1;
                mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], ne0 * @sizeOf(i32));
            }
        }
    }

    /// Re-draws the ids between timed runs, as the base does.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn reinitPerfIter(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.base.reinitPerfIter(ti, ctx);
    }
};
