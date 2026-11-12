//! Port of `test_rms_norm_mul_add` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_RMS_NORM + GGML_OP_MUL + GGML_OP_ADD (+ GGML_OP_MUL).

const std = @import("std");
const mod = @import("module.zig");

/// `rms_norm(a) * b + c`, with optional multiple feeding adds, a trailing
/// multiply and an in-place residual, compared as one fused graph.
pub const TestRmsNormMulAdd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    eps: f32 = 1e-6,
    /// Whether `a` is larger than `b` and `c`, which broadcast over it.
    broadcast: bool = false,
    /// A sequence of adds feeding into the RMS norm.
    multi_add: bool = false,
    /// A trailing multiply by a one-element `d`.
    post_mul: bool = false,
    /// The final add writes in place into the norm's input.
    alias_rms_input: bool = false,
    /// `b` is a single row, broadcast.
    weight_broadcast: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "eps", "broadcast", "multi_add", "post_mul", "alias_rms_input", "weight_broadcast" });
    }

    /// Always `RMS_NORM_MUL_ADD`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("RMS_NORM_MUL_ADD");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `rms_norm(a + b + c) * b + c`, with the variants above.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const broadcast_dims = [4]i64{ ne[0] * 2, ne[1] * 3, ne[2] * 3, ne[3] * 4 };

        var a = g.newTensor(ctx, self.type, 4, if (self.broadcast) &broadcast_dims else &self.ne);
        const b = if (self.weight_broadcast) g.newTensor1d(ctx, self.type, ne[0]) else g.newTensor(ctx, self.type, 4, &self.ne);
        const c = g.newTensor(ctx, self.type, 4, &self.ne);
        var d: ?*mod.c.ggml_tensor = null;

        mod.Graph.setName(a, "a");
        mod.Graph.setName(b, "b");
        mod.Graph.setName(c, "c");

        // Use a, b and c early, so there is no OP_NONE between rms_norm and mul.
        a = mod.c.ggml_add(ctx, mod.c.ggml_add(ctx, a, b), c);
        if (self.post_mul) {
            const dt = g.newTensor1d(ctx, self.type, 1);
            mod.Graph.setName(dt, "d");
            a = mod.c.ggml_add(ctx, a, dt);
            d = dt;
        }
        if (self.multi_add) {
            a = mod.c.ggml_add(ctx, mod.c.ggml_add(ctx, a, b), c);
        }
        const mul: *mod.c.ggml_tensor = mod.c.ggml_mul(ctx, mod.c.ggml_rms_norm(ctx, a, self.eps), b);
        var out: *mod.c.ggml_tensor = if (self.alias_rms_input) mod.c.ggml_add_inplace(ctx, a, mul) else mod.c.ggml_add(ctx, mul, c);
        if (d) |dt| {
            out = mod.c.ggml_mul(ctx, out, dt);
        }
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the inputs uniformly in [-10, 10).
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -10.0, 10.0);
    }
};
