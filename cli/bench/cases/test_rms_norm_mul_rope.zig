//! Port of `test_rms_norm_mul_rope` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_RMS_NORM with optional GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW and GGML_OP_SET_ROWS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_RMS_NORM, optionally fed by a chain of adds and followed by a
/// weight multiply, RoPE and a SET_ROWS of its rows: compared as one fused
/// graph.
pub const TestRmsNormMulRope = struct {
    const Self = @This();

    // Constructor parameters, in order (which differs from upstream's member
    // order), with the C++ defaults.
    ne: [4]i64 = .{ 0, 0, 0, 0 },
    eps: f32 = 1e-6,
    /// Feed rms_norm by a sequence of adds.
    multi_add: bool = false,
    set_rows: bool = false,
    /// Multiply by a 1D [ne0] weight, as model norm weights are.
    broadcast: bool = false,
    mode: i32 = mod.c.GGML_ROPE_TYPE_NORMAL,
    mul: bool = true,
    rope: bool = true,
    set_rows_type: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "ne", "eps", "multi_add", "mul", "rope", "set_rows", "broadcast", "mode", "set_rows_type" });
    }

    /// Always `RMS_NORM_MUL_ROPE`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("RMS_NORM_MUL_ROPE");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Looser bounds for wide rows and for large positions, which amplify
    /// the difference between GPU and CPU trig functions.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        if (self.ne[0] == 8192) return 5e-6;
        if (self.ne[2] > 8192) return 1e-5;
        return mod.Compare.default_max_nmse_err;
    }

    /// Builds the chain.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        var a = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);

        var b: ?*mod.c.ggml_tensor = null;
        var c: ?*mod.c.ggml_tensor = null;
        var w: ?*mod.c.ggml_tensor = null;

        if (self.multi_add or (self.mul and !self.broadcast)) {
            b = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, ne[0], ne[1], ne[2], 1);
        }
        if (self.multi_add) {
            c = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, ne[0], ne[1], ne[2], 1);
        }
        if (self.mul) {
            w = if (self.broadcast) g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, ne[0]) else b;
        }

        if (self.multi_add) {
            a = mod.c.ggml_add(ctx, mod.c.ggml_add(ctx, a, b), c);
        }

        a = mod.c.ggml_rms_norm(ctx, a, self.eps);

        if (self.mul) {
            a = mod.c.ggml_mul(ctx, a, w);
        }

        if (self.rope) {
            const is_mrope = self.mode & mod.c.GGML_ROPE_TYPE_MROPE != 0;
            const pos = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, ne[2] * @as(i64, if (is_mrope) 4 else 1));

            if (is_mrope) {
                const n_dims: c_int = @intCast(ne[0]);
                var sections = [4]c_int{ @divTrunc(n_dims, 3), @divTrunc(n_dims, 3), @divTrunc(n_dims, 3), 0 };
                a = mod.c.ggml_rope_multi(ctx, a, pos, null, n_dims, &sections, self.mode, 0, 10000.0, 1.0, 0.0, 1.0, 32.0, 1.0);
            } else {
                a = mod.c.ggml_rope(ctx, a, pos, @intCast(ne[0]), self.mode);
            }
        }

        if (self.set_rows) {
            const view: *mod.c.ggml_tensor = mod.c.ggml_view_2d(ctx, a, ne[0] * ne[1], ne[2], a.nb[2], 0);

            const dst = g.newTensor2d(ctx, self.set_rows_type, ne[0] * ne[1], ne[2] * 2);
            mod.Graph.setName(dst, "dst");

            const row_idxs = g.newTensor1d(ctx, mod.c.GGML_TYPE_I64, ne[2]);
            mod.Graph.setName(row_idxs, "row_idxs");

            a = mod.c.ggml_set_rows(ctx, dst, view, row_idxs);
        }

        mod.Graph.setName(a, "out");
        return a;
    }

    /// Fills the row ids with shuffled distinct rows, the positions with
    /// random values below 512, and the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I64) {
                mod.Helpers.initSetRowsRowIds(ti, t, @intCast(self.ne[2] * 2));
            } else if (t.type == mod.c.GGML_TYPE_I32) {
                const data = ti.allocator.alloc(i32, @intCast(mod.c.ggml_nelements(t))) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = ti.random.intRangeLessThan(i32, 0, 512);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, mod.c.ggml_nbytes(t));
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
