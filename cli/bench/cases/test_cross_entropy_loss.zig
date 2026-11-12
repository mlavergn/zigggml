//! Port of `test_cross_entropy_loss` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_CROSS_ENTROPY_LOSS.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_CROSS_ENTROPY_LOSS of logits against softmax-normalized labels.
pub const TestCrossEntropyLoss = struct {
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

    /// Builds `cross_entropy_loss(logits, soft_max(labels))`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const logits = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(logits, "logits");

        // The labels are assumed to be constant: no gradients.
        var labels = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(labels, "labels");

        // Ensure the labels add up to 1.
        labels = mod.c.ggml_soft_max(ctx, labels);
        mod.Graph.setName(labels, "labels_normalized");

        const out: *mod.c.ggml_tensor = mod.c.ggml_cross_entropy_loss(ctx, logits, labels);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills every tensor uniformly in [-100, 100): larger differences
    /// between logits make the softmax more linear.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        ti.allRange(ctx, -100.0, 100.0);
    }
};
