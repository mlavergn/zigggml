//! Port of `test_add_add` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_ADD + GGML_OP_ADD.

const std = @import("std");
const mod = @import("module.zig");

/// A fused residual chain, `(a + b) + c`, compared as one graph.
pub const TestAddAdd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    type_addend: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 64, 5, 4, 3 },
    /// `c` is a single row, broadcast.
    broadcast: bool = false,
    /// `a` is a non-contiguous view.
    view: bool = false,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "type_addend", "ne", "broadcast", "view" });
    }

    /// Always `ADD_ADD`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("ADD_ADD");
    }

    /// The chain is compared as one fused graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// Fused ADDs can keep FP32 intermediates while the CPU rounds each ADD
    /// to FP16/BF16.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        if (self.type.raw() == mod.c.GGML_TYPE_F16) return 1e-6;
        if (self.type.raw() == mod.c.GGML_TYPE_BF16) return 1e-4;
        return mod.Compare.default_max_nmse_err;
    }

    /// Builds `(a + b) + c`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const ne = self.ne;
        const broadcast_dims = [4]i64{ ne[0], 1, 1, 1 };

        var a: *mod.c.ggml_tensor = undefined;
        if (self.view) {
            const parent = [4]i64{ ne[0] * 3, ne[1] * 2, ne[2], ne[3] };
            a = g.newTensor(ctx, self.type, 4, &parent);
            mod.Graph.setName(a, "a_parent");
            a = mod.c.ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a.nb[1], a.nb[2], a.nb[3], 0);
            mod.Graph.setName(a, "a");
        } else {
            a = g.newTensor(ctx, self.type, 4, &self.ne);
            mod.Graph.setName(a, "a");
        }

        const b = g.newTensor(ctx, self.type_addend, 4, &self.ne);
        const c = g.newTensor(ctx, self.type_addend, 4, if (self.broadcast) &broadcast_dims else &self.ne);
        mod.Graph.setName(b, "b");
        mod.Graph.setName(c, "c");

        const out: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, mod.c.ggml_add(ctx, a, b), c);
        mod.Graph.setName(out, "out");
        return out;
    }
};
