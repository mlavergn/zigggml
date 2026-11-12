//! Port of `test_opt_step_sgd` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_OPT_STEP_SGD.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_OPT_STEP_SGD on one parameter tensor.
pub const TestOptStepSgd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 5, 4, 3 },

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne" });
    }

    /// Builds `opt_step_sgd(a, grad, sgd_params)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const a = g.newTensor4d(ctx, self.type, ne[0], ne[1], ne[2], ne[3]);
        mod.Graph.setName(a, "a");
        // Not a gradient detail: the optimizer step asserts its target is a parameter.
        mod.c.ggml_set_param(a);

        const grad = g.newTensor4d(ctx, self.type, ne[0], ne[1], ne[2], ne[3]);
        mod.Graph.setName(grad, "grad");

        const sgd_params = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, 2);
        mod.Graph.setName(sgd_params, "sgd_params");

        const out: *mod.c.ggml_tensor = mod.c.ggml_opt_step_sgd(ctx, a, grad, sgd_params);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills every tensor uniformly in [0, 1): sgd_params need non-negative
    /// values.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, 0.0, 1.0);
    }
};
