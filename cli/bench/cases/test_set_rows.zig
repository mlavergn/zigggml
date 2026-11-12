//! Port of `test_set_rows` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_SET_ROWS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_SET_ROWS, broadcast over dims 2 and 3, with optionally
/// non-contiguous source and row ids.
pub const TestSetRows = struct {
    const Self = @This();

    // Constructor parameters, in order (upstream has no defaults but `v`).
    type_src: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_dst: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_idx: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 0, 0, 0, 0 },
    /// Broadcast of dims 2 and 3 only.
    nr23: [2]i32 = .{ 0, 0 },
    /// Rows to set.
    r: i32 = 0,
    /// View: non-contiguous src1.
    v: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type_src", "type_dst", "type_idx", "ne", "nr23", "r", "v" });
    }

    /// Estimates the error of one quantized value off by one for quantized
    /// destinations; 1e-7 otherwise.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        const td = self.type_dst.raw();
        const ts = self.type_src.raw();
        if (td == mod.c.GGML_TYPE_Q2_0 or td == mod.c.GGML_TYPE_Q4_0 or td == mod.c.GGML_TYPE_Q4_1 or
            td == mod.c.GGML_TYPE_IQ4_NL or
            td == mod.c.GGML_TYPE_Q5_0 or td == mod.c.GGML_TYPE_Q5_1 or td == mod.c.GGML_TYPE_Q8_0)
        {
            // The values are in [-1, 1]: roughly (2 / 2^bits)^2, divided by
            // the mean square of the reference, about 0.25 times the count.
            var err_estimate: f64 = 1.0 / 8.0;
            if (ts == mod.c.GGML_TYPE_F16 and td == mod.c.GGML_TYPE_Q2_0) err_estimate *= 4.0;
            if (td == mod.c.GGML_TYPE_Q5_0 or td == mod.c.GGML_TYPE_Q5_1) err_estimate /= 2.0;
            if (td == mod.c.GGML_TYPE_Q8_0) err_estimate /= 8.0;
            err_estimate *= err_estimate;
            if (ts == mod.c.GGML_TYPE_F16) err_estimate *= 16.0;
            const count = self.ne[0] * self.r * self.ne[2] * self.nr23[0] * self.ne[3] * self.nr23[1];
            const denom: f32 = 0.25 * @as(f32, @floatFromInt(count));
            err_estimate /= denom;
            return err_estimate;
        }
        return 1e-7;
    }

    /// Looser Q8_0 bounds on WebGPU and HTP.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        const name = mod.Helpers.regName(backend);
        if (self.type_dst.raw() == mod.c.GGML_TYPE_Q8_0) {
            if (std.mem.eql(u8, name, "WebGPU")) return @max(self.maxNmseErr(), 2e-7);
            if (std.mem.eql(u8, name, "HTP")) return @max(self.maxNmseErr(), 5e-6);
        }
        return self.maxNmseErr();
    }

    /// Builds `set_rows(dst, src, row_idxs)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const ne2 = ne[2] * self.nr23[0];
        const ne3 = ne[3] * self.nr23[1];
        const dst = g.newTensor4d(ctx, self.type_dst, ne[0], ne[1], ne2, ne3);
        mod.Graph.setName(dst, "dst");

        var src = g.newTensor4d(ctx, self.type_src, ne[0], self.r, ne2, ne3);
        mod.Graph.setName(src, "src");

        var row_idxs = g.newTensor3d(ctx, self.type_idx, self.r, ne[2], ne[3]);
        mod.Graph.setName(row_idxs, "row_idxs");

        if (self.v) {
            const half = @divTrunc(self.r, 2);
            src = mod.c.ggml_view_4d(ctx, src, ne[0], half, ne2, ne3, src.nb[1], src.nb[2], src.nb[3], 0);
            row_idxs = mod.c.ggml_view_3d(ctx, row_idxs, half, ne[2], ne[3], row_idxs.nb[1], row_idxs.nb[2], 0);
            mod.Graph.setName(row_idxs, "view_of_rows");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_set_rows(ctx, dst, src, row_idxs);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the row ids with shuffled distinct rows, the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (mod.Helpers.isViewOp(t.op)) continue;
            if (t.type == mod.c.GGML_TYPE_I64 or t.type == mod.c.GGML_TYPE_I32) {
                mod.Helpers.initSetRowsRowIds(ti, t, @intCast(self.ne[1]));
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
