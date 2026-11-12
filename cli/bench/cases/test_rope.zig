//! Port of `test_rope` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ROPE.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ROPE (or GGML_OP_ROPE_BACK when not `forward`), in normal, NeoX,
/// multi-section and vision modes, with optional frequency factors, views
/// and a rotated-dims offset.
pub const TestRope = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 10, 5, 3, 1 },
    n_dims: i32 = 10,
    mode: i32 = mod.c.GGML_ROPE_TYPE_NORMAL,
    /// Used to generate positions.
    n_ctx: i32 = 512,
    /// freq_scale.
    fs: f32 = 1.0,
    /// ext_factor.
    ef: f32 = 0.0,
    /// attn_factor.
    af: f32 = 0.0,
    /// Use frequency factors.
    ff: bool = false,
    /// View: 1 for a non-contiguous `a`, 2 for the second half of dim 0.
    v: i32 = 0,
    forward: bool = true,
    inplace: bool = false,
    /// Offset of the rotated dims window, set via `ggml_rope_set_offset`.
    n_offs: i32 = 0,

    /// Writes the case's parameters; `forward` shows in the op.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "n_dims", "mode", "n_ctx", "fs", "ef", "af", "ff", "v", "inplace", "n_offs" });
    }

    /// Builds the rope (or rope back) of `a` at positions `pos`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne_a = self.ne_a;
        var a: *mod.c.ggml_tensor = undefined;
        if (self.v & 1 != 0) {
            var ne = ne_a;
            ne[0] *= 2;
            ne[1] *= 4;
            ne[2] *= 3;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
        } else if (self.v == 2) {
            // Second-half slice along dim 0 (as build_rope_2d in clip.cpp): the
            // offset is often unaligned, which exercises backends' alignment paths.
            var ne = ne_a;
            ne[0] *= 2;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a.nb[1], a.nb[2], a.nb[3], @as(usize, @intCast(ne_a[0])) * mod.c.ggml_element_size(a));
            mod.Graph.setName(a, "view_of_a");
        } else {
            a = g.newTensor(ctx, self.type, 4, &ne_a);
            mod.Graph.setName(a, "a");
        }

        const is_mrope = self.mode & mod.c.GGML_ROPE_TYPE_MROPE != 0;
        const is_vision = self.mode == mod.c.GGML_ROPE_TYPE_VISION;

        const pos = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, if (is_mrope or is_vision) ne_a[2] * 4 else ne_a[2]);
        mod.Graph.setName(pos, "pos");

        var freq: ?*mod.c.ggml_tensor = null;
        if (self.ff) {
            const f = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, @divTrunc(self.n_dims, 2));
            mod.Graph.setName(f, "freq");
            freq = f;
        }

        const fs = self.fs;
        const ef = self.ef;
        const af = self.af;
        const mode = self.mode;
        var out: *mod.c.ggml_tensor = undefined;
        if (is_mrope) {
            // Vision rope only uses the first two sections, for the image's
            // (x, y) coordinates.
            var sections: [4]c_int = undefined;
            var n_dims = self.n_dims;
            if (is_vision) {
                std.debug.assert(@divTrunc(self.n_dims, 4) > 0);
                sections = .{ @divTrunc(self.n_dims, 4), @divTrunc(self.n_dims, 4), 0, 0 };
                n_dims = @divTrunc(self.n_dims, 2);
            } else {
                std.debug.assert(@divTrunc(self.n_dims, 3) > 0);
                sections = .{ @divTrunc(self.n_dims, 3), @divTrunc(self.n_dims, 3), @divTrunc(self.n_dims, 3), 0 };
            }
            if (self.forward) {
                out = if (self.inplace)
                    mod.c.ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims, &sections, mode, 0, 10000.0, fs, ef, af, 1.0, 1.0)
                else
                    mod.c.ggml_rope_multi(ctx, a, pos, freq, n_dims, &sections, mode, 0, 10000.0, fs, ef, af, 1.0, 1.0);
            } else {
                out = mod.c.ggml_rope_multi_back(ctx, a, pos, freq, n_dims, &sections, mode, 0, 10000.0, fs, ef, af, 1.0, 1.0);
            }
        } else {
            const n_dims = self.n_dims;
            if (self.forward) {
                out = if (self.inplace)
                    mod.c.ggml_rope_ext_inplace(ctx, a, pos, freq, n_dims, mode, 0, 10000.0, fs, ef, af, 1.0, 1.0)
                else
                    mod.c.ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0, fs, ef, af, 1.0, 1.0);
            } else {
                out = mod.c.ggml_rope_ext_back(ctx, a, pos, freq, n_dims, mode, 0, 10000.0, fs, ef, af, 1.0, 1.0);
            }
        }
        if (self.n_offs != 0) out = mod.c.ggml_rope_set_offset(out, self.n_offs);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the positions with random values below `n_ctx`, the frequency
    /// factors in [0.9, 1.1), and the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                const num_pos_ids: usize = @intCast(if (self.mode & mod.c.GGML_ROPE_TYPE_MROPE != 0) self.ne_a[2] * 4 else self.ne_a[2]);
                const data = ti.allocator.alloc(i32, num_pos_ids) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = ti.random.intRangeLessThan(i32, 0, self.n_ctx);
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, num_pos_ids * @sizeOf(i32));
            } else if (t.ne[0] == @divTrunc(self.n_dims, 2)) {
                ti.uniform(t, 0.9, 1.1);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
