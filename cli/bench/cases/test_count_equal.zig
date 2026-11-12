//! Port of `test_count_equal` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_COUNT_EQUAL.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_COUNT_EQUAL between the row argmaxes of two inputs.
pub const TestCountEqual = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 4, 500, 1, 1 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Builds `count_equal(argmax(a), argmax(b))`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        const a_argmax: *mod.c.ggml_tensor = mod.c.ggml_argmax(ctx, a);
        mod.Graph.setName(a_argmax, "a_argmax");
        const b = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(b, "b");
        const b_argmax: *mod.c.ggml_tensor = mod.c.ggml_argmax(ctx, b);
        mod.Graph.setName(b_argmax, "b_argmax");
        const out: *mod.c.ggml_tensor = mod.c.ggml_count_equal(ctx, a_argmax, b_argmax);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills F32 tensors row by row with a shuffle of `0..ne0`, so there
    /// are no ties; the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_F32) {
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
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }

    /// Exact results.
    ///
    /// - `self`: the case.
    /// - Returns: 0.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 0.0;
    }
};
