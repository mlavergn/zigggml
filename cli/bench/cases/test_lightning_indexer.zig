//! Port of `test_lightning_indexer` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_LIGHTNING_INDEXER.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_LIGHTNING_INDEXER: weighted per-head Q·K scores over the KV cache,
/// masked.
pub const TestLightningIndexer = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    /// Indexer K head size.
    hsk: i64 = 128,
    /// Number of indexer heads.
    nh: i64 = 64,
    /// KV size.
    kv: i64 = 256,
    /// Batch size.
    nb: i64 = 128,
    /// Number of streams.
    ns: i64 = 1,
    /// ne[3] of the mask.
    nm: i64 = 1,
    type_k: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("hsk", self.hsk);
        try v.add("nh", self.nh);
        try v.add("kv", self.kv);
        try v.add("nb", self.nb);
        try v.add("ns", self.ns);
        try v.add("nm", self.nm);
        try v.add("type_K", self.type_k);
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 1e-6;
    }

    /// ((2·hsk + 2)·nh + 1) per KV cell, query and stream.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - Returns: the FLOP count.
    pub fn opFlops(self: *Self, t: *mod.c.ggml_tensor) u64 {
        _ = t;
        return @intCast(((2 * self.hsk + 2) * self.nh + 1) * self.kv * self.nb * self.ns);
    }

    /// Builds `lightning_indexer(q, k, w, m)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const q = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, self.hsk, self.nh, self.nb, self.ns);
        mod.Graph.setName(q, "q");

        const k = g.newTensor4d(ctx, self.type_k, self.hsk, 1, self.kv, self.ns);
        mod.Graph.setName(k, "k");

        const w = g.newTensor4d(ctx, mod.c.GGML_TYPE_F32, self.nh, self.nb, 1, self.ns);
        mod.Graph.setName(w, "w");

        const m = g.newTensor4d(ctx, mod.c.GGML_TYPE_F16, self.kv, self.nb, 1, self.nm);
        mod.Graph.setName(m, "m");

        const out: *mod.c.ggml_tensor = mod.c.ggml_lightning_indexer(ctx, q, k, w, m);
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the mask with a KQ mask and the rest uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        _ = self;
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (std.mem.eql(u8, std.mem.sliceTo(&t.name, 0), "m")) {
                ti.kqMask(t, -1.0, 1.0);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
