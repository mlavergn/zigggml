//! Port of `test_diag_mask_inf` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_DIAG_MASK_INF.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_DIAG_MASK_INF: `-inf` above the diagonal shifted by `n_past`.
pub const TestDiagMaskInf = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 10, 10, 3, 2 },
    n_past: i32 = 5,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "ne", "n_past" });
    }

    /// Builds `diag_mask_inf(a, n_past)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor(ctx, self.type, 4, &self.ne);
        mod.Graph.setName(a, "a");
        const out: *mod.c.ggml_tensor = mod.c.ggml_diag_mask_inf(ctx, a, self.n_past);
        mod.Graph.setName(out, "out");
        return out;
    }
};
