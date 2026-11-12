//! Port of `test_get_rows_back` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GET_ROWS_BACK.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GET_ROWS_BACK, with optionally non-contiguous row ids.
pub const TestGetRowsBack = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    /// Columns.
    n: i32 = 10,
    /// Rows.
    m: i32 = 5,
    /// Rows to get.
    r: i32 = 3,
    /// Batch size.
    b: i32 = 1,
    /// View: non-contiguous src1.
    v: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "n", "m", "r", "b", "v" });
    }

    /// Builds `get_rows_back(grad, rows, in_forward)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const in_forward = g.newTensor3d(ctx, self.type, self.n, self.m, self.b);
        mod.Graph.setName(in_forward, "in_forward");

        var rows = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.r, self.b);
        mod.Graph.setName(rows, "rows");
        if (self.v) {
            rows = mod.c.ggml_view_2d(ctx, rows, @divTrunc(self.r, 2), self.b, rows.nb[1], 0);
            mod.Graph.setName(rows, "view_of_rows");
        }

        const grad = g.newTensor3d(ctx, self.type, self.n, self.r, self.b);
        mod.Graph.setName(grad, "grad");

        const out: *mod.c.ggml_tensor = mod.c.ggml_get_rows_back(ctx, grad, rows, in_forward);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the row ids with random rows, the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                if (mod.Helpers.isViewOp(t.op)) continue;
                const data = ti.allocator.alloc(i32, @intCast(self.r * self.b)) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = ti.random.intRangeLessThan(i32, 0, self.m);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, data.len * @sizeOf(i32));
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
