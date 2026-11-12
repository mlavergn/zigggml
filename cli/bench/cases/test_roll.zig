//! Port of `test_roll` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ROLL.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_ROLL of a fixed 10x5x4x3 input, optionally permuted.
pub const TestRoll = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    shift0: i32 = 3,
    shift1: i32 = -2,
    shift3: i32 = 1,
    shift4: i32 = -1,
    permute: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "shift0", "shift1", "shift3", "shift4", "permute" });
    }

    /// Builds `roll(a, shift0, shift1, shift3, shift4)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = [4]i64{ 10, 5, 4, 3 };
        var a = g.newTensor(ctx, mod.c.GGML_TYPE_F32, 4, &ne);
        mod.Graph.setName(a, "a");

        if (self.permute) {
            // ggml_roll only requires nb[0] == type size, so a permuted src is valid.
            a = mod.c.ggml_permute(ctx, a, 0, 2, 1, 3);
            mod.Graph.setName(a, "a_permuted");
        }

        const out: *mod.c.ggml_tensor = mod.c.ggml_roll(ctx, a, self.shift0, self.shift1, self.shift3, self.shift4);
        mod.Graph.setName(out, "out");
        return out;
    }
};
