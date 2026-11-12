//! Port of `test_glu_split` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_GLU.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_GLU on two separate inputs, contiguous or (with `v & 1`)
/// non-contiguous.
pub const TestGluSplit = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    op: mod.c.enum_ggml_glu_op = 0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 128, 2, 2, 2 },
    /// View: 1 for non-contiguous inputs.
    v: i32 = 0,

    /// Writes the case's parameters, then `,split`.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "v" });
        try w.writeAll(",split");
    }

    /// Metal computes f32 geglu_quick in f32, so the CPU's fp16 table fails
    /// the default bound there.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        if (self.op == mod.c.GGML_GLU_OP_GEGLU_QUICK and self.type.raw() == mod.c.GGML_TYPE_F32 and
            std.mem.eql(u8, mod.Helpers.regName(backend), "MTL")) return 1e-10;
        return mod.Compare.default_max_nmse_err;
    }

    /// Builds `glu_split(a, b)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a: *mod.c.ggml_tensor = undefined;
        var b: *mod.c.ggml_tensor = undefined;
        if (self.v & 1 != 0) {
            var ne = self.ne_a;
            ne[0] *= 3;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, self.ne_a[0], self.ne_a[1], self.ne_a[2], self.ne_a[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
            b = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(b, "b");
            b = mod.c.ggml_view_4d(ctx, b, self.ne_a[0], self.ne_a[1], self.ne_a[2], self.ne_a[3], b.nb[1], b.nb[2], b.nb[3], 0);
            // Upstream names `a` here, not `b`.
            mod.Graph.setName(a, "view_of_b");
        } else {
            a = g.newTensor(ctx, self.type, 4, &self.ne_a);
            mod.Graph.setName(a, "a");
            b = g.newTensor(ctx, self.type, 4, &self.ne_a);
            mod.Graph.setName(b, "b");
        }
        const out: *mod.c.ggml_tensor = mod.c.ggml_glu_split(ctx, a, b, self.op);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs over an extended range, to catch NaNs in GELU.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -150.0, 150.0);
    }
};
