//! Port of `test_mul_mat_pos` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT.

const std = @import("std");
const mod = @import("module.zig");

/// `test_mul_mat` with src1 in [0, 1]: a zero-mean src1 hides errors in the
/// zero point or the min of a quantized src0.
pub const TestMulMatPos = struct {
    const Self = @This();

    // Constructor parameters (inherited from `test_mul_mat`), in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    m: i64 = 32,
    n: i64 = 32,
    k: i64 = 32,
    /// Dims 3 and 4.
    bs: [2]i64 = .{ 10, 10 },
    /// Repeat in dims 3 and 4.
    nr: [2]i64 = .{ 2, 2 },
    /// Permutation of dimensions.
    per: [4]i64 = .{ 0, 1, 2, 3 },
    /// Size of k in memory: a non-contiguous view for k_v > k, none for 0.
    k_v: i64 = 0,
    /// Number of outputs.
    o: u32 = 1,
    /// a and b are overlapping views of the same tensor.
    src_overlap: bool = false,
    /// Rows of a in memory: the batches of a are strided for m_v > m, none for 0.
    m_v: i64 = 0,
    /// Bytes after the m_v rows of each batch of a.
    pad: i64 = 0,

    // State.
    /// The `test_mul_mat` this case extends.
    base: mod.TestMulMat = .{},

    /// Builds the base case from the parameters.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.base = .{
            .type_a = self.type_a,
            .type_b = self.type_b,
            .m = self.m,
            .n = self.n,
            .k = self.k,
            .bs = self.bs,
            .nr = self.nr,
            .per = self.per,
            .k_v = self.k_v,
            .o = self.o,
            .src_overlap = self.src_overlap,
            .m_v = self.m_v,
            .pad = self.pad,
        };
    }

    /// Writes the base case's parameters, then `src1=[0,1]`.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try self.base.vars(w);
        try w.writeAll(",src1=[0,1]");
    }

    /// The base case's default bound.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return self.base.maxNmseErr();
    }

    /// The base case's per-backend bound.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        return self.base.maxNmseErrBackend(backend);
    }

    /// The base case's FLOP count.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        return self.base.opFlops(t);
    }

    /// Builds the base case's graph.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        return self.base.buildGraph(g);
    }

    /// Whether the base case compares the whole graph.
    ///
    /// - `self`: the case.
    /// - Returns: the flag.
    pub fn runWholeGraph(self: *Self) bool {
        return self.base.runWholeGraph();
    }

    /// Always `MUL_MAT`, as in the base case.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        try self.base.opDesc(t, w);
    }

    /// Fills F32 tensors uniformly in [0, 1] and the rest in [-1, 1].
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_F32) {
                ti.uniform(t, 0.0, 1.0);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
