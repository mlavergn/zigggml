//! Port of `test_mul_mat_hadamard` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_MUL_MAT with `a` a Walsh-Hadamard matrix and the
/// `GGML_HINT_SRC0_IS_HADAMARD` hint set: `test_mul_mat` by composition.
pub const TestMulMatHadamard = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 32,
    /// Dims 3 and 4.
    bs: [2]i64 = .{ 1, 1 },
    /// Repeat in dims 3 and 4.
    nr: [2]i64 = .{ 1, 1 },

    // State.
    /// The base case, built from the parameters in `setup`.
    base: mod.TestMulMat = .{},

    /// Builds the base case.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        std.debug.assert(self.type_a.raw() == mod.c.GGML_TYPE_F32);
        self.base = .{ .type_a = self.type_a, .type_b = self.type_b, .m = self.m, .n = self.n, .k = self.k, .bs = self.bs, .nr = self.nr };
    }

    /// Writes the case's parameters, as the base does.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try self.base.vars(w);
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

    /// Whether the base compares the whole graph.
    ///
    /// - `self`: the case.
    /// - Returns: the flag.
    pub fn runWholeGraph(self: *Self) bool {
        return self.base.runWholeGraph();
    }

    /// Builds the base graph and sets the Hadamard hint on every `MUL_MAT`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const out = self.base.buildGraph(g);
        var it = mod.Helpers.tensors(g.ctx);
        while (it.next()) |t| {
            if (t.op == mod.c.GGML_OP_MUL_MAT) mod.c.ggml_mul_mat_set_hint(t, mod.c.GGML_HINT_SRC0_IS_HADAMARD);
        }
        return out;
    }

    /// Fills `a` with a normalized Walsh-Hadamard matrix and the other float
    /// tensors uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (std.mem.eql(u8, std.mem.sliceTo(&t.name, 0), "a")) {
                const n_cols: usize = @intCast(t.ne[0]);
                const n_rows: usize = @intCast(mod.c.ggml_nrows(t));
                const data = ti.allocator.alloc(f32, n_cols * n_rows) catch @panic("OOM");
                defer ti.allocator.free(data);
                const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(n_cols)));
                for (0..n_rows) |r| {
                    for (0..n_cols) |i| {
                        data[r * n_cols + i] = if (@popCount(r & i) % 2 == 0) scale else -scale;
                    }
                }
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(f32));
            } else if (t.type == mod.c.GGML_TYPE_F32 or t.type == mod.c.GGML_TYPE_F16) {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }

    /// Always `MUL_MAT_HADAMARD`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_HADAMARD");
    }
};
