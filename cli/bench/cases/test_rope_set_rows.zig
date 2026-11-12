//! Port of `test_rope_set_rows` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ROPE + GGML_OP_VIEW + GGML_OP_SET_ROWS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ROPE whose output, viewed as rows, is written by
/// GGML_OP_SET_ROWS: compared as one fused graph.
pub const TestRopeSetRows = struct {
    const Self = @This();

    // Constructor parameters, in order (upstream has no defaults but `n_dims`).
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_idx: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 0, 0, 0, 0 },
    mode: i32 = 0,
    n_dims: i32 = 128,

    // State.
    n_ctx: i32 = 512,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "type_idx", "ne_a", "mode", "n_dims" });
    }

    /// Always `ROPE_SET_ROWS`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("ROPE_SET_ROWS");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `set_rows(dst, view(rope(a, pos)), row_idxs)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne_a = self.ne_a;
        const a = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, ne_a[0], ne_a[1], ne_a[2], 1);
        mod.Graph.setName(a, "a");

        const is_mrope = self.mode & mod.c.GGML_ROPE_TYPE_MROPE != 0;
        const is_vision = self.mode == mod.c.GGML_ROPE_TYPE_VISION;

        const pos = if (is_mrope or is_vision)
            g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, ne_a[2] * 4)
        else
            g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, ne_a[2]);
        mod.Graph.setName(pos, "pos");

        const fs: f32 = 1.4245;
        const ef: f32 = 0.7465;
        const af: f32 = 1.4245;
        const freq: ?*mod.c.ggml_tensor = null;

        var rope: *mod.c.ggml_tensor = undefined;
        if (is_mrope) {
            if (is_vision) {
                std.debug.assert(@divTrunc(self.n_dims, 4) > 0);
                // Vision RoPE uses only the first two dimensions, for the image's (x, y).
                var rope_sections = [4]c_int{ @divTrunc(self.n_dims, 4), @divTrunc(self.n_dims, 4), 0, 0 };
                rope = mod.c.ggml_rope_multi(ctx, a, pos, freq, @divTrunc(self.n_dims, 2), &rope_sections, self.mode, 0, 10000.0, fs, ef, af, 1.0, 1.0);
            } else {
                std.debug.assert(@divTrunc(self.n_dims, 3) > 0);
                var rope_sections = [4]c_int{ @divTrunc(self.n_dims, 3), @divTrunc(self.n_dims, 3), @divTrunc(self.n_dims, 3), 0 };
                rope = mod.c.ggml_rope_multi(ctx, a, pos, freq, self.n_dims, &rope_sections, self.mode, 0, 10000.0, fs, ef, af, 1.0, 1.0);
            }
        } else {
            rope = mod.c.ggml_rope(ctx, a, pos, self.n_dims, self.mode);
        }

        const view: *mod.c.ggml_tensor = mod.c.ggml_view_2d(ctx, rope, ne_a[0] * ne_a[1], ne_a[2], rope.nb[2], 0);

        const dst = g.newTensor4d(ctx, self.type, ne_a[0] * ne_a[1], ne_a[2] * ne_a[3], 1, 1);
        mod.Graph.setName(dst, "dst");

        const row_idxs = g.newTensor3d(ctx, self.type_idx, ne_a[2], 1, 1);
        mod.Graph.setName(row_idxs, "row_idxs");

        const out: *mod.c.ggml_tensor = mod.c.ggml_set_rows(ctx, dst, view, row_idxs);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the row ids with shuffled distinct rows, the positions with
    /// random positions below `n_ctx`, frequency factors in [0.9, 1.1), and
    /// the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (std.mem.eql(u8, std.mem.sliceTo(&t.name, 0), "row_idxs")) {
                if (mod.Helpers.isViewOp(t.op)) continue;
                mod.Helpers.initSetRowsRowIds(ti, t, @intCast(self.ne_a[2]));
            } else if (t.type == mod.c.GGML_TYPE_I32) {
                // pos
                const num_pos_ids: usize = @intCast(if (self.mode & mod.c.GGML_ROPE_TYPE_MROPE != 0) self.ne_a[2] * 4 else self.ne_a[2]);
                const data = ti.allocator.alloc(i32, num_pos_ids) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = ti.random.intRangeLessThan(i32, 0, self.n_ctx);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, num_pos_ids * @sizeOf(i32));
            } else if (t.ne[0] == @divTrunc(self.n_dims, 2)) {
                // Frequency factors in [0.9, 1.1).
                ti.uniform(t, 0.9, 1.1);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
