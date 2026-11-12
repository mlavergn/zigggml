//! Port of `test_argsort` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ARGSORT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ARGSORT of each row, ascending or descending.
pub const TestArgsort = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 16, 10, 10, 10 },
    order: mod.c.enum_ggml_sort_order = mod.c.GGML_SORT_ORDER_ASC,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "order" });
    }

    /// Builds `argsort(a, order)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_argsort(ctx, a, self.order);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills I32 tensors with shuffled random values and F32 rows with
    /// shuffled distinct values, so the sort has no ties.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                // Indices.
                const n: usize = @intCast(mod.c.ggml_nelements(t));
                const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = ti.random.intRangeAtMost(i32, 0, std.math.maxInt(i32));
                ti.random.shuffle(i32, data);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, n * @sizeOf(i32));
            } else if (t.type == mod.c.GGML_TYPE_F32) {
                // Unique values avoid ties.
                const n: usize = @intCast(t.ne[0]);
                const data = ti.allocator.alloc(f32, n) catch @panic("OOM");
                defer ti.allocator.free(data);
                var r: i64 = 0;
                while (r < mod.c.ggml_nrows(t)) : (r += 1) {
                    for (data, 0..) |*d, i| d.* = @floatFromInt(i);
                    ti.random.shuffle(f32, data);
                    mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], n * @sizeOf(f32));
                }
            } else {
                @panic("fatal error");
            }
        }
    }
};
