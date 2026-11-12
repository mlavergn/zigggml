//! Port of `test_get_rows` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GET_ROWS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GET_ROWS, with optionally non-contiguous row ids and source.
pub const TestGetRows = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Columns.
    n: i32 = 10,
    /// Rows.
    m: i32 = 5,
    /// Rows to get.
    r: i32 = 3,
    /// Batch size, dim 2.
    be1: i32 = 1,
    /// Batch size, dim 3.
    be2: i32 = 1,
    /// View of src1 (the row ids).
    v: bool = false,
    /// View of src0.
    vs0: bool = false,
    /// Column offset of the src0 view.
    offset_cols: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "n", "m", "r", "be1", "be2", "v", "vs0", "offset_cols" });
    }

    /// Builds `get_rows(in, rows)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var in: *mod.c.ggml_tensor = undefined;
        if (self.vs0) {
            const offset_rows: i32 = 3;
            const padded_m = self.m + offset_rows;
            const padded_n = self.n + self.offset_cols;
            const in_padded = g.newTensor4d(ctx, self.type, padded_n, padded_m, self.be1, self.be2);
            mod.Graph.setName(in_padded, "in_padded");
            const offset = @as(usize, @intCast(self.offset_cols)) * in_padded.nb[0] + @as(usize, @intCast(offset_rows)) * in_padded.nb[1];
            in = mod.c.ggml_view_4d(ctx, in_padded, self.n, self.m, self.be1, self.be2, in_padded.nb[1], in_padded.nb[2], in_padded.nb[3], offset);
            mod.Graph.setName(in, "in_view");
        } else {
            in = g.newTensor4d(ctx, self.type, self.n, self.m, self.be1, self.be2);
            mod.Graph.setName(in, "in");
        }

        var rows = g.newTensor3d(ctx, mod.c.GGML_TYPE_I32, if (self.v) self.r + 1 else self.r, self.be1, self.be2);
        mod.Graph.setName(rows, "rows");
        if (self.v) {
            rows = mod.c.ggml_view_3d(ctx, rows, @divTrunc(self.r, 2), self.be1, self.be2, rows.nb[1], rows.nb[2], rows.nb[0]);
            mod.Graph.setName(rows, "view_of_rows");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_get_rows(ctx, in, rows);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the row ids with random rows of `in`, the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (mod.Helpers.isViewOp(t.op)) continue;
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
