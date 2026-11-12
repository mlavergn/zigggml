//! Port of `test_flash_attn_ext_large_logits` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_FLASH_ATTN_EXT.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_FLASH_ATTN_EXT with large Q values, so the online softmax has to
/// rescale the partial results. Takes `TestFlashAttnExt`'s parameters.
pub const TestFlashAttnExtLargeLogits = struct {
    const Self = @This();

    /// The range of Q values, ±`q_range`.
    pub const q_range: i32 = 20;

    // Constructor parameters (inherited from `test_flash_attn_ext`), in
    // order, with the C++ defaults.
    hsk: i64 = 128,
    hsv: i64 = 128,
    nh: i64 = 32,
    nr23: [2]i64 = .{ 1, 1 },
    kv: i64 = 96,
    nb: i64 = 8,
    mask: bool = true,
    sinks: bool = false,
    max_bias: f32 = 0.0,
    logit_softcap: f32 = 0.0,
    prec: mod.GgmlPrec = .of(mod.c.GGML_PREC_F32),
    type_k: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    type_v: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
    permute: [4]i32 = .{ 0, 1, 2, 3 },
    kv_view: bool = true,
    v_is_view_of_k: bool = false,
    n_kv_max: i64 = 0,

    // State.
    base: mod.TestFlashAttnExt = .{},

    /// Builds the base case from the parameters.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.base = .{
            .hsk = self.hsk,
            .hsv = self.hsv,
            .nh = self.nh,
            .nr23 = self.nr23,
            .kv = self.kv,
            .nb = self.nb,
            .mask = self.mask,
            .sinks = self.sinks,
            .max_bias = self.max_bias,
            .logit_softcap = self.logit_softcap,
            .prec = self.prec,
            .type_k = self.type_k,
            .type_v = self.type_v,
            .permute = self.permute,
            .kv_view = self.kv_view,
            .v_is_view_of_k = self.v_is_view_of_k,
            .n_kv_max = self.n_kv_max,
        };
    }

    /// Writes the base case's parameters, then `q_range`.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try self.base.vars(w);
        try w.print(",q_range={d}", .{q_range});
    }

    /// The base case's bound.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return self.base.maxNmseErr();
    }

    /// The base case's FLOP count.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        return self.base.opFlops(t);
    }

    /// Builds the base case's graph.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        return self.base.buildGraph(g);
    }

    /// Fills the tensors as the base case does, then Q over ±`q_range`.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.base.initializeTensors(ti, ctx);
        const q: *mod.c.ggml_tensor = mod.c.ggml_get_tensor(ctx, "q");
        ti.uniform(q, -@as(f32, @floatFromInt(q_range)), @floatFromInt(q_range));
    }
};
