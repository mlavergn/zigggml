//! Port of `test_get_rows_after_mul_mat` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GET_ROWS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GET_ROWS on a weight also used by a preceding GGML_OP_MUL_MAT
/// (tied embeddings), which a backend may have put in a reordered layout.
pub const TestGetRowsAfterMulMat = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_Q8_0),
    /// Columns.
    n: i32 = 256,
    /// Rows.
    m: i32 = 16,
    /// Rows to get.
    r: i32 = 4,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "n", "m", "r" });
    }

    /// The bound of `test_mul_mat`.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `repeat(mul_mat(w, x), gr) + gr` with `gr = get_rows(w, rows)`,
    /// so the mul_mat runs first.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const w = g.newTensor2d(ctx, self.type, self.n, self.m);
        mod.Graph.setName(w, "w");
        const x = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, self.n, 1);
        mod.Graph.setName(x, "x");
        const rows = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, self.r);
        mod.Graph.setName(rows, "rows");

        const mm: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, w, x);
        mod.Graph.setName(mm, "mm");
        const gr: *mod.c.ggml_tensor = mod.c.ggml_get_rows(ctx, w, rows);
        mod.Graph.setName(gr, "gr");

        // repeat(mm) is the first source, so mul_mat is computed before get_rows.
        const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, mod.c.ggml_repeat(ctx, mm, gr), gr);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the row ids with random rows of `w`, the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                const data = ti.allocator.alloc(i32, @intCast(mod.c.ggml_nelements(t))) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = ti.random.intRangeLessThan(i32, 0, self.m);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(i32));
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
