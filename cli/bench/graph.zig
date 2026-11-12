const std = @import("std");
const mod = @import("module.zig");

/// What a case's graph is being built for. Sentinels are only added when
/// checking correctness, as in `test-backend-ops`.
pub const Mode = enum {
    /// Comparing a backend against the CPU: sentinels guard every tensor.
    check,
    /// Timing: no sentinels.
    perf,
};

/// The context a case builds its graph in: the port of `test_case`'s
/// `ggml_new_tensor*` hijack, which adds a sentinel tensor after every input
/// in check mode so the backend comparison catches out-of-bounds writes.
///
/// Cases call `newTensor*` where the C++ calls the member `ggml_new_tensor*`,
/// passing the context explicitly (`g.ctx`, or `g.ctx_weights.?` for a case
/// with `useWeightContext`), and call every other ggml function directly on
/// `g.ctx`.
pub const Graph = struct {
    const Self = @This();

    /// The size, in floats, of each sentinel (`test_case::sentinel_size`).
    pub const sentinel_size: i64 = 1024;

    /// Allocator for the sentinel list and any case scratch; an arena.
    allocator: std.mem.Allocator,
    /// The context the graph's tensors are created in.
    ctx: *mod.c.ggml_context,
    /// A second context for weights; non-null only for a case whose
    /// `useWeightContext` returns true.
    ctx_weights: ?*mod.c.ggml_context = null,
    /// What the graph is for.
    mode: Mode,
    /// The comparison graph, created before the case builds (as
    /// `test_case::gf` is in `eval`); null when timing. A case may add nodes
    /// to it in check mode, as some C++ cases do.
    gf: ?*mod.c.ggml_cgraph = null,
    /// The sentinels added so far.
    sentinels: std.ArrayList(*mod.c.ggml_tensor) = .empty,

    /// Wraps contexts for one build.
    ///
    /// - `allocator`: an arena that outlives the build.
    /// - `ctx`: the main context.
    /// - `ctx_weights`: the weights context, or null.
    /// - `mode`: check or perf.
    /// - Returns: the wrapper.
    pub fn init(allocator: std.mem.Allocator, ctx: *mod.c.ggml_context, ctx_weights: ?*mod.c.ggml_context, mode: Mode) Self {
        return Self{ .allocator = allocator, .ctx = ctx, .ctx_weights = ctx_weights, .mode = mode };
    }

    /// Releases the sentinel list.
    ///
    /// - `self`: the wrapper.
    pub fn deinit(self: *Self) void {
        self.sentinels.deinit(self.allocator);
    }

    /// Adds a sentinel to `ctx` in check mode (`test_case::add_sentinel`).
    ///
    /// - `self`: the wrapper.
    /// - `ctx`: the context to add it to.
    pub fn addSentinel(self: *Self, ctx: *mod.c.ggml_context) void {
        if (self.mode != .check) return;
        const sentinel: *mod.c.ggml_tensor = mod.c.ggml_new_tensor_1d(ctx, mod.c.GGML_TYPE_F32, sentinel_size);
        var name: [32]u8 = undefined;
        const text: [:0]const u8 = std.fmt.bufPrintSentinel(&name, "sent_{d}", .{self.sentinels.items.len}, 0) catch "sent";
        _ = mod.c.ggml_set_name(sentinel, text.ptr);
        self.sentinels.append(self.allocator, sentinel) catch @panic("OOM");
    }

    /// `ggml_new_tensor`, followed by a sentinel in check mode.
    ///
    /// - `self`: the wrapper.
    /// - `ctx`: the context.
    /// - `ty`: the element type: a `GgmlType` or a `GGML_TYPE_*` value.
    /// - `n_dims`: the number of dimensions.
    /// - `ne`: the extents, at least `n_dims` of them.
    /// - Returns: the tensor.
    pub fn newTensor(self: *Self, ctx: *mod.c.ggml_context, ty: anytype, n_dims: c_int, ne: [*]const i64) *mod.c.ggml_tensor {
        const t: *mod.c.ggml_tensor = mod.c.ggml_new_tensor(ctx, Self.cType(ty), n_dims, ne);
        self.addSentinel(ctx);
        return t;
    }

    /// `ggml_new_tensor_1d`, followed by a sentinel in check mode.
    ///
    /// - `self`: the wrapper.
    /// - `ctx`: the context.
    /// - `ty`: the element type.
    /// - `ne0`: the extent.
    /// - Returns: the tensor.
    pub fn newTensor1d(self: *Self, ctx: *mod.c.ggml_context, ty: anytype, ne0: i64) *mod.c.ggml_tensor {
        const t: *mod.c.ggml_tensor = mod.c.ggml_new_tensor_1d(ctx, Self.cType(ty), ne0);
        self.addSentinel(ctx);
        return t;
    }

    /// `ggml_new_tensor_2d`, followed by a sentinel in check mode.
    ///
    /// - `self`: the wrapper.
    /// - `ctx`: the context.
    /// - `ty`: the element type.
    /// - `ne0`: extent 0.
    /// - `ne1`: extent 1.
    /// - Returns: the tensor.
    pub fn newTensor2d(self: *Self, ctx: *mod.c.ggml_context, ty: anytype, ne0: i64, ne1: i64) *mod.c.ggml_tensor {
        const t: *mod.c.ggml_tensor = mod.c.ggml_new_tensor_2d(ctx, Self.cType(ty), ne0, ne1);
        self.addSentinel(ctx);
        return t;
    }

    /// `ggml_new_tensor_3d`, followed by a sentinel in check mode.
    ///
    /// - `self`: the wrapper.
    /// - `ctx`: the context.
    /// - `ty`: the element type.
    /// - `ne0`: extent 0.
    /// - `ne1`: extent 1.
    /// - `ne2`: extent 2.
    /// - Returns: the tensor.
    pub fn newTensor3d(self: *Self, ctx: *mod.c.ggml_context, ty: anytype, ne0: i64, ne1: i64, ne2: i64) *mod.c.ggml_tensor {
        const t: *mod.c.ggml_tensor = mod.c.ggml_new_tensor_3d(ctx, Self.cType(ty), ne0, ne1, ne2);
        self.addSentinel(ctx);
        return t;
    }

    /// `ggml_new_tensor_4d`, followed by a sentinel in check mode.
    ///
    /// - `self`: the wrapper.
    /// - `ctx`: the context.
    /// - `ty`: the element type.
    /// - `ne0`: extent 0.
    /// - `ne1`: extent 1.
    /// - `ne2`: extent 2.
    /// - `ne3`: extent 3.
    /// - Returns: the tensor.
    pub fn newTensor4d(self: *Self, ctx: *mod.c.ggml_context, ty: anytype, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *mod.c.ggml_tensor {
        const t: *mod.c.ggml_tensor = mod.c.ggml_new_tensor_4d(ctx, Self.cType(ty), ne0, ne1, ne2, ne3);
        self.addSentinel(ctx);
        return t;
    }

    /// Names a tensor (`ggml_set_name`), for the C++'s many
    /// `ggml_set_name(t, "...")` calls.
    ///
    /// - `t`: the tensor.
    /// - `name`: the name, a string literal or other null-terminated string.
    pub fn setName(t: *mod.c.ggml_tensor, name: [*:0]const u8) void {
        _ = mod.c.ggml_set_name(t, name);
    }

    /// The C value of a type given as a `GgmlType` or an integer.
    ///
    /// - `ty`: the type.
    /// - Returns: the `ggml_type`.
    pub fn cType(ty: anytype) mod.c.enum_ggml_type {
        return if (@TypeOf(ty) == mod.GgmlType) ty.raw() else @intCast(ty);
    }
};
