//! Port of `test_add_id` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ADD_ID.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ADD_ID: adds to each expert slot of `a` the row of `b` its id
/// selects, with some ids optionally skipped.
pub const TestAddId = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_b: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    n_embd: i64 = 128,
    n_experts: i64 = 16,
    n_experts_used: i64 = 8,
    n_token: i64 = 10,
    /// Set some ids to a negative value.
    skip: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_a", "type_b", "n_embd", "n_experts", "n_experts_used", "n_token", "skip" });
    }

    /// The bytes of the output, `a` and the ids.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the size.
    pub fn opSize(self: *Self, t: *mod.c.ggml_tensor) usize {
        _ = self;
        return mod.c.ggml_nbytes(t) + mod.c.ggml_nbytes(t.src[0]) + mod.c.ggml_nbytes(t.src[2]);
    }

    /// Builds `add_id(a, b, ids)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor3d(ctx, self.type_a, self.n_embd, self.n_experts_used, self.n_token);
        const b = g.newTensor2d(ctx, self.type_b, self.n_embd, self.n_experts);
        var ids = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, self.n_experts, self.n_token);
        if (self.n_experts_used != self.n_experts) {
            ids = mod.c.ggml_view_2d(ctx, ids, self.n_experts_used, self.n_token, ids.nb[1], 0);
            mod.Graph.setName(ids, "view_of_ids");
        }
        const out: *mod.c.ggml_tensor = mod.c.ggml_add_id(ctx, a, b, ids);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the ids row by row with a shuffle of `i % n_experts`, some
    /// masked when `skip`; the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                if (mod.Helpers.isViewOp(t.op)) continue;
                const n: usize = @intCast(t.ne[0]);
                const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
                defer ti.allocator.free(data);
                var r: i64 = 0;
                while (r < mod.c.ggml_nrows(t)) : (r += 1) {
                    for (data, 0..) |*d, i| d.* = @intCast(@mod(@as(i64, @intCast(i)), self.n_experts));
                    ti.random.shuffle(i32, data);
                    if (self.skip) mod.Helpers.maskSkippedIds(data, r);
                    mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], n * @sizeOf(i32));
                }
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
