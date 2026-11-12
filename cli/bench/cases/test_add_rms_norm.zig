//! Port of `test_add_rms_norm` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ADD + GGML_OP_RMS_NORM.

const std = @import("std");
const mod = @import("module.zig");

/// `rms_norm(a + b)`, compared as one fused graph.
pub const TestAddRmsNorm = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    eps: f32 = 1e-6,
    /// Whether `a` is larger than `b`, which broadcasts over it.
    broadcast: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "eps", "broadcast" });
    }

    /// Always `ADD_RMS_NORM`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("ADD_RMS_NORM");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Builds `rms_norm(a + b)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const broadcast_dims = [4]i64{ ne[0] * 2, ne[1] * 3, ne[2] * 3, ne[3] * 4 };

        const a = g.newTensor(ctx, self.type, 4, if (self.broadcast) &broadcast_dims else &self.ne);
        const b = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        mod.Graph.setName(b, "b");

        const add_result: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, a, b);
        mod.Graph.setName(add_result, "add_result");

        const out: *mod.c.ggml_tensor = mod.c.ggml_rms_norm(ctx, add_result, self.eps);
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
