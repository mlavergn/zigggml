//! Port of `test_unary` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_UNARY.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_UNARY on a contiguous or (with `v & 1`) non-contiguous input.
pub const TestUnary = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    op: mod.c.enum_ggml_unary_op = 0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne_a: [4]i64 = .{ 128, 2, 2, 2 },
    /// View: 1 for a non-contiguous `a`.
    v: i32 = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne_a", "v" });
    }

    /// Metal computes f32 gelu_quick in f32, so the CPU's fp16 table fails
    /// the default bound there.
    ///
    /// - `self`: the case.
    /// - `backend`: the backend under test.
    /// - Returns: the bound.
    pub fn maxNmseErrBackend(self: *Self, backend: mod.c.ggml_backend_t) f64 {
        if (self.op == mod.c.GGML_UNARY_OP_GELU_QUICK and self.type.raw() == mod.c.GGML_TYPE_F32 and
            std.mem.eql(u8, mod.Helpers.regName(backend), "MTL")) return 1e-10;
        return mod.Compare.default_max_nmse_err;
    }

    /// Builds `unary(a)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        var a: *mod.c.ggml_tensor = undefined;
        if (self.v & 1 != 0) {
            var ne = self.ne_a;
            ne[0] *= 3;
            ne[1] *= 2;
            ne[2] *= 5;
            ne[3] *= 4;
            a = g.newTensor(ctx, self.type, 4, &ne);
            mod.Graph.setName(a, "a");
            a = mod.c.ggml_view_4d(ctx, a, self.ne_a[0], self.ne_a[1], self.ne_a[2], self.ne_a[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "view_of_a");
        } else {
            a = g.newTensor(ctx, self.type, 4, &self.ne_a);
            mod.Graph.setName(a, "a");
        }
        const out: *mod.c.ggml_tensor = mod.c.ggml_unary(ctx, a, self.op);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs over an extended range, to catch NaNs in GELU; FP16
    /// exp/expm1 inputs stay in range so every backend stays finite.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        var min: f32 = -150.0;
        var max: f32 = 150.0;
        if (self.type.raw() == mod.c.GGML_TYPE_F16 and (self.op == mod.c.GGML_UNARY_OP_EXP or self.op == mod.c.GGML_UNARY_OP_EXPM1)) {
            min = -10.0;
            max = 10.0;
        }
        ti.allRange(ctx, min, max);
    }
};
