//! Port of `test_opt_step_adamw` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_OPT_STEP_ADAMW.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_OPT_STEP_ADAMW on one parameter tensor.
pub const TestOptStepAdamw = struct {
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

    /// Builds `opt_step_adamw(a, grad, grad_m, grad_v, adamw_params)`.
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

        const grad_m = g.newTensor4d(ctx, self.type, ne[0], ne[1], ne[2], ne[3]);
        mod.Graph.setName(grad_m, "grad_m");

        const grad_v = g.newTensor4d(ctx, self.type, ne[0], ne[1], ne[2], ne[3]);
        mod.Graph.setName(grad_v, "grad_v");

        const adamw_params = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, 7);
        mod.Graph.setName(adamw_params, "adamw_params");

        const out: *mod.c.ggml_tensor = mod.c.ggml_opt_step_adamw(ctx, a, grad, grad_m, grad_v, adamw_params);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills every tensor uniformly in [0, 1): grad_v and adamw_params need
    /// non-negative values.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, 0.0, 1.0);
    }
};
