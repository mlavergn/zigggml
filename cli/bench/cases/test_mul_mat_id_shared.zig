//! Port of `test_mul_mat_id_shared` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT_ID.

const std = @import("std");
const mod = @import("module.zig");

/// A routed SwiGLU expert block (GGML_OP_MUL_MAT_ID) added to a shared one
/// (GGML_OP_MUL_MAT) over the same input, compared as one fused graph.
pub const TestMulMatIdShared = struct {
    const Self = @This();

    // Constructor parameters, in order (no C++ defaults).
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    n: i64 = 0,

    // State.
    /// The fused nodes, `routed` then `shared`, set by `buildGraph`.
    nodes: [2]*mod.c.ggml_tensor = undefined,
    /// How many of `nodes` are set: 0 before `buildGraph`, then 2.
    n_nodes: usize = 0,

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try mod.Vars.fields(w, self, .{ "type", "n" });
    }

    /// Always `MUL_MAT_ID_SHARED`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_ID_SHARED");
    }

    /// The fused ops are compared as one graph.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// The routed and shared SwiGLU outputs.
    ///
    /// - `self`: the case.
    /// - Returns: the nodes; empty before the graph is built.
    pub fn fusionTestNodes(self: *Self) []const *mod.c.ggml_tensor {
        return self.nodes[0..self.n_nodes];
    }

    /// Builds `swiglu(routed experts) + swiglu(shared expert)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const input = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, 512, self.n);
        const input_ids: *mod.c.ggml_tensor = mod.c.ggml_reshape_3d(ctx, input, 512, 1, self.n);
        const ids = g.newTensor2d(ctx, mod.c.GGML_TYPE_I32, 3, self.n);
        mod.Graph.setName(ids, "ids");
        const up = g.newTensor3d(ctx, self.type, 512, 128, 8);
        const gate = g.newTensor3d(ctx, self.type, 512, 128, 8);
        const shared_up = g.newTensor2d(ctx, self.type, 512, 128);
        const shared_gate = g.newTensor2d(ctx, self.type, 512, 128);
        const routed: *mod.c.ggml_tensor = mod.c.ggml_swiglu_split(ctx, mod.c.ggml_mul_mat_id(ctx, gate, input_ids, ids), mod.c.ggml_mul_mat_id(ctx, up, input_ids, ids));
        const shared: *mod.c.ggml_tensor = mod.c.ggml_swiglu_split(ctx, mod.c.ggml_mul_mat(ctx, shared_gate, input), mod.c.ggml_mul_mat(ctx, shared_up, input));
        self.nodes = .{ routed, shared };
        self.n_nodes = 2;
        return mod.c.ggml_add(ctx, routed, mod.c.ggml_reshape_3d(ctx, shared, 128, 1, self.n));
    }

    /// Fills the matrices uniformly and the ids with shuffled experts of 8.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        mod.Helpers.initMulMatIdTensors(ti, ctx, 8, false);
    }
};
