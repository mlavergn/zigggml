//! Port of `test_generic_op` (tests/test-backend-ops.cpp @cd959bae9): any GGML_OP, deserialized.

const std = @import("std");
const mod = @import("module.zig");

/// One source of a generic op (`input_tensor`).
pub const InputTensor = struct {
    const Self = @This();

    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 0, 0, 0, 0 },
    /// Strides in bytes; all 0 for the default contiguous strides.
    nb: [4]usize = .{ 0, 0, 0, 0 },

    /// Whether the source has explicit strides other than the contiguous
    /// ones (`is_non_contiguous`).
    ///
    /// - `self`: the source.
    /// - Returns: true for a non-contiguous source.
    pub fn isNonContiguous(self: *const Self) bool {
        if (self.nb[0] == 0) return false;
        const ty = self.type.raw();
        const default_nb0 = mod.c.ggml_type_size(ty);
        const default_nb1 = default_nb0 * @as(usize, @intCast(@divTrunc(self.ne[0], mod.c.ggml_blck_size(ty))));
        const default_nb2 = default_nb1 * @as(usize, @intCast(self.ne[1]));
        const default_nb3 = default_nb2 * @as(usize, @intCast(self.ne[2]));
        return self.nb[0] != default_nb0 or self.nb[1] != default_nb1 or self.nb[2] != default_nb2 or self.nb[3] != default_nb3;
    }
};

/// The number of 32-bit op parameters (`GGML_MAX_OP_PARAMS / sizeof(int32_t)`).
const n_op_params = mod.c.GGML_MAX_OP_PARAMS / @sizeOf(i32);

/// Writes the sources as `type[ne0,ne1,ne2,ne3]`, followed by
/// `nb[nb0,nb1,nb2,nb3]` when non-contiguous, joined by commas (upstream's
/// `var_to_str(std::vector<input_tensor>)`).
fn writeSources(w: *std.Io.Writer, sources: []const InputTensor) !void {
    for (sources, 0..) |src, i| {
        if (i > 0) try w.writeByte(',');
        try w.print("{s}[{d},{d},{d},{d}]", .{ src.type.name(), src.ne[0], src.ne[1], src.ne[2], src.ne[3] });
        if (src.isNonContiguous()) {
            try w.print("nb[{d},{d},{d},{d}]", .{ src.nb[0], src.nb[1], src.nb[2], src.nb[3] });
        }
    }
}

/// Writes the non-zero op parameters as `[index:value,...]` (upstream's
/// `var_to_str` of the op parameter array).
fn writeOpParams(w: *std.Io.Writer, params: *const [n_op_params]i32) !void {
    try w.writeByte('[');
    var first = true;
    for (params, 0..) |p, i| {
        if (p == 0) continue;
        if (!first) try w.writeByte(',');
        try w.print("{d}:{d}", .{ i, p });
        first = false;
    }
    try w.writeByte(']');
}

/// Sets the first `nels` int32 elements of `t` to values uniform in [0, `max`].
fn setRandomIds(ti: *mod.TensorInit, t: *mod.c.ggml_tensor, nels: usize, max: i32) void {
    const data = ti.allocator.alloc(i32, nels) catch @panic("OOM");
    defer ti.allocator.free(data);
    for (data) |*d| d.* = ti.random.intRangeAtMost(i32, 0, max);
    mod.c.ggml_backend_tensor_set(t, data.ptr, 0, nels * @sizeOf(i32));
}

/// Sets each row of `t` to a shuffle of `i % modulus` (or `i` for a null
/// modulus) over its positions.
fn setShuffledRows(ti: *mod.TensorInit, t: *mod.c.ggml_tensor, modulus: ?i64) void {
    const n: usize = @intCast(t.ne[0]);
    const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
    defer ti.allocator.free(data);
    const nrows: usize = @intCast(mod.c.ggml_nrows(t));
    for (0..nrows) |r| {
        for (data, 0..) |*d, i| {
            const idx: i64 = @intCast(i);
            d.* = @intCast(if (modulus) |m| @mod(idx, m) else idx);
        }
        ti.random.shuffle(i32, data);
        mod.c.ggml_backend_tensor_set(t, data.ptr, r * t.nb[1], n * @sizeOf(i32));
    }
}

/// Any op, described by its type, shape, op parameters and sources: a case
/// deserialized from a recorded graph.
pub const TestGenericOp = struct {
    const Self = @This();

    /// The source descriptor, for building `sources`.
    pub const Input = InputTensor;

    // Constructor parameters, in order, with the C++ defaults.
    op: mod.c.enum_ggml_op = 0,
    type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    ne: [4]i64 = .{ 0, 0, 0, 0 },
    op_params: [n_op_params]i32 = @splat(0),
    sources: []const InputTensor = &.{},
    name: []const u8 = "",

    /// Writes the case's parameters, led by `name` when set.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        if (self.name.len > 0) try v.add("name", self.name);
        try v.add("type", self.type);
        try v.add("ne", self.ne);
        try w.writeAll(",op_params=");
        try writeOpParams(w, &self.op_params);
        try w.writeAll(",sources=");
        try writeSources(w, self.sources);
    }

    /// The bound by op, scaled with the KV length for flash attention.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        return switch (self.op) {
            mod.c.GGML_OP_MUL_MAT,
            mod.c.GGML_OP_MUL_MAT_ID,
            mod.c.GGML_OP_OUT_PROD,
            mod.c.GGML_OP_CONV_TRANSPOSE_2D,
            mod.c.GGML_OP_IM2COL,
            mod.c.GGML_OP_CONV_2D,
            mod.c.GGML_OP_CONV_3D,
            mod.c.GGML_OP_SET_ROWS,
            mod.c.GGML_OP_CPY,
            => 5e-4,
            mod.c.GGML_OP_SOFT_MAX => 1e-6,
            mod.c.GGML_OP_RWKV_WKV7 => 5e-3,
            mod.c.GGML_OP_FLASH_ATTN_EXT => blk: {
                // Accumulated floating-point error grows with the KV length.
                const kv: f64 = @floatFromInt(self.sources[1].ne[1]);
                break :blk 5e-4 * @max(1.0, kv / 20000.0);
            },
            else => 1e-7,
        };
    }

    /// Builds the op node by hand: sources (strided views of a backing
    /// buffer when non-contiguous), output, op and op parameters.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const source_count = @min(self.sources.len, mod.c.GGML_MAX_SRC);

        var source_tensors: [mod.c.GGML_MAX_SRC]*mod.c.ggml_tensor = undefined;
        for (self.sources[0..source_count], 0..) |*src, i| {
            if (src.isNonContiguous()) {
                const ty = src.type.raw();
                const blck_size: usize = @intCast(mod.c.ggml_blck_size(ty));
                const type_size = mod.c.ggml_type_size(ty);
                var total_size: usize = 0;
                if (blck_size == 1) {
                    total_size = type_size;
                    for (0..4) |d| total_size += @as(usize, @intCast(src.ne[d] - 1)) * src.nb[d];
                } else {
                    total_size = @as(usize, @intCast(src.ne[0])) * src.nb[0] / blck_size;
                    for (1..4) |d| total_size += @as(usize, @intCast(src.ne[d] - 1)) * src.nb[d];
                }

                // Bytes to elements, padded to the block size for quantized types.
                var backing_elements = (total_size * blck_size + type_size - 1) / type_size;
                backing_elements = ((backing_elements + blck_size - 1) / blck_size) * blck_size;
                const backing = g.newTensor1d(ctx, ty, @intCast(backing_elements));
                const view: *mod.c.ggml_tensor = mod.c.ggml_view_4d(ctx, backing, src.ne[0], src.ne[1], src.ne[2], src.ne[3], src.nb[1], src.nb[2], src.nb[3], 0);
                // ggml_view_4d does not set nb[0].
                view.nb[0] = src.nb[0];
                source_tensors[i] = view;
            } else {
                source_tensors[i] = g.newTensor4d(ctx, src.type, src.ne[0], src.ne[1], src.ne[2], src.ne[3]);
            }
        }

        // Ops with an inplace flag create a view of src[0] as their output.
        var inplace = false;
        if (self.op == mod.c.GGML_OP_SET or self.op == mod.c.GGML_OP_ACC) {
            inplace = self.op_params[4] != 0;
        } else if (self.op == mod.c.GGML_OP_ADD_REL_POS) {
            inplace = self.op_params[0] != 0;
        }

        var out: *mod.c.ggml_tensor = undefined;
        if (inplace and source_count > 0) {
            out = mod.c.ggml_view_tensor(ctx, source_tensors[0]);
        } else {
            out = g.newTensor4d(ctx, self.type, self.ne[0], self.ne[1], self.ne[2], self.ne[3]);
        }
        out.op = self.op;
        for (0..source_count) |i| out.src[i] = source_tensors[i];
        out.op_params = self.op_params;
        mod.Graph.setName(out, "out");
        return out;
    }

    /// Fills the sources by op: the KQ mask for flash attention, valid row,
    /// position, expert or state indices for integer sources, and the rest
    /// uniformly.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        const out: *mod.c.ggml_tensor = mod.c.ggml_get_tensor(ctx, "out");
        const op = self.op;
        const n = @min(self.sources.len, mod.c.GGML_MAX_SRC);
        for (0..n) |i| {
            const t: *mod.c.ggml_tensor = out.src[i] orelse break;

            // FLASH_ATTN_EXT: src[3] is the KQ mask.
            if (op == mod.c.GGML_OP_FLASH_ATTN_EXT and i == 3) {
                ti.kqMask(t, -1.0, 1.0);
                continue;
            }

            if (t.type != mod.c.GGML_TYPE_I32 and t.type != mod.c.GGML_TYPE_I64) {
                ti.uniform(t, -1.0, 1.0);
                continue;
            }

            if (op == mod.c.GGML_OP_GET_ROWS or op == mod.c.GGML_OP_GET_ROWS_BACK) {
                const num_rows: i32 = @intCast(self.sources[0].ne[1]);
                setRandomIds(ti, t, @intCast(mod.c.ggml_nelements(t)), num_rows - 1);
            } else if (op == mod.c.GGML_OP_SET_ROWS) {
                mod.Helpers.initSetRowsRowIds(ti, t, @intCast(self.ne[1]));
            } else if (op == mod.c.GGML_OP_ROPE) {
                const mode = self.op_params[2];
                const nels: usize = @intCast(if (mode & mod.c.GGML_ROPE_TYPE_MROPE != 0) self.ne[2] * 4 else self.ne[2]);
                setRandomIds(ti, t, nels, @intCast(self.ne[2] - 1));
            } else if (op == mod.c.GGML_OP_MUL_MAT_ID or op == mod.c.GGML_OP_ADD_ID) {
                const n_expert = if (op == mod.c.GGML_OP_MUL_MAT_ID) self.sources[0].ne[2] else self.sources[1].ne[1];
                setShuffledRows(ti, t, n_expert);
            } else if (op == mod.c.GGML_OP_SSM_SCAN) {
                setShuffledRows(ti, t, null);
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
