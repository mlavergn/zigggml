const std = @import("std");
const mod = @import("module.zig");

/// Free functions of `test-backend-ops.cpp` that several cases share.
pub const Helpers = struct {
    const Self = @This();

    /// Whether `backend`'s registry reports feature `name` as `"1"`
    /// (`backend_has_feature`).
    ///
    /// - `backend`: the backend.
    /// - `name`: the feature name, e.g. `"BLACKWELL_NATIVE_FP4"`.
    /// - Returns: true when the feature is present and enabled.
    pub fn backendHasFeature(backend: mod.c.ggml_backend_t, name: []const u8) bool {
        const dev = mod.c.ggml_backend_get_device(backend);
        const reg = mod.c.ggml_backend_dev_backend_reg(dev);
        const raw = mod.c.ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features") orelse return false;
        const get_features: mod.c.ggml_backend_get_features_t = @ptrCast(@alignCast(raw));
        var f: [*c]mod.c.struct_ggml_backend_feature = get_features.?(reg);
        if (f == null) return false;
        while (f.*.name != null) : (f += 1) {
            if (std.mem.eql(u8, std.mem.span(f.*.name), name) and std.mem.eql(u8, std.mem.span(f.*.value), "1")) return true;
        }
        return false;
    }

    /// The name of `backend`'s registry (`ggml_backend_reg_name`), e.g.
    /// `"CPU"`, `"CUDA"`, `"MTL"`, `"Vulkan"`.
    ///
    /// - `backend`: the backend.
    /// - Returns: the name, static.
    pub fn regName(backend: mod.c.ggml_backend_t) []const u8 {
        return std.mem.span(mod.c.ggml_backend_reg_name(mod.c.ggml_backend_dev_backend_reg(mod.c.ggml_backend_get_device(backend))));
    }

    /// The i-th 32-bit op parameter of `t` (`test_get_op_params_i32`).
    ///
    /// - `t`: the tensor.
    /// - `i`: the index.
    /// - Returns: the parameter.
    pub fn opParamsI32(t: *const mod.c.ggml_tensor, i: usize) i32 {
        std.debug.assert(i < t.op_params.len);
        return t.op_params[i];
    }

    /// Whether any node of op `op` in `gf` asks for 8-bit activations
    /// (`graph_mul_mat_hi_prec_act`).
    ///
    /// - `gf`: the graph, or null.
    /// - `op`: the op to look for.
    /// - Returns: true when such a node requests `GGML_PREC_Q8`.
    pub fn graphMulMatHiPrecAct(gf: ?*mod.c.ggml_cgraph, op: mod.c.enum_ggml_op) bool {
        const graph = gf orelse return false;
        const n: usize = @intCast(mod.c.ggml_graph_n_nodes(graph));
        for (0..n) |i| {
            const node: *mod.c.ggml_tensor = mod.c.ggml_graph_node(graph, @intCast(i));
            if (node.op == op and Self.opParamsI32(node, 3) == mod.c.GGML_PREC_Q8) return true;
        }
        return false;
    }

    /// Fills each row of index tensor `t` (I32 or I64) with a shuffled
    /// subset of `0..num_rows` (`init_set_rows_row_ids`).
    ///
    /// - `ti`: the value source.
    /// - `t`: the index tensor, `ne[0] <= num_rows`.
    /// - `num_rows`: the rows to choose from.
    pub fn initSetRowsRowIds(ti: *mod.TensorInit, t: *mod.c.ggml_tensor, num_rows: usize) void {
        const allocator = ti.allocator;
        const data = allocator.alloc(i64, num_rows) catch @panic("OOM");
        defer allocator.free(data);
        const n: usize = @intCast(t.ne[0]);
        const data_i32 = allocator.alloc(i32, n) catch @panic("OOM");
        defer allocator.free(data_i32);
        var j2: usize = 0;
        while (j2 < t.ne[2]) : (j2 += 1) {
            var j1: usize = 0;
            while (j1 < t.ne[1]) : (j1 += 1) {
                for (data, 0..) |*d, i| d.* = @intCast(i);
                ti.random.shuffle(i64, data);
                const offs = j1 * t.nb[1] + j2 * t.nb[2];
                if (t.type == mod.c.GGML_TYPE_I32) {
                    for (data_i32, 0..) |*d, i| d.* = @intCast(data[i]);
                    mod.c.ggml_backend_tensor_set(t, data_i32.ptr, offs, n * @sizeOf(i32));
                } else {
                    mod.c.ggml_backend_tensor_set(t, data.ptr, offs, n * @sizeOf(i64));
                }
            }
        }
    }

    /// Marks some expert ids `-1` (skipped), by a fixed pattern of row and
    /// position (`mask_skipped_ids`).
    ///
    /// - `ids`: one row of ids.
    /// - `row`: the row's index.
    pub fn maskSkippedIds(ids: []i32, row: i64) void {
        for (ids, 0..) |*id, i| {
            if (@mod(row, 4) == 1 or @mod(row + @as(i64, @intCast(i)), 3) == 0) id.* = -1;
        }
    }

    /// Fills every non-view I32 tensor of `ctx` with shuffled expert ids
    /// `i % n_mats` per row, optionally masking some as skipped
    /// (`init_mul_mat_id_ids`).
    ///
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    /// - `n_mats`: the number of experts.
    /// - `skip`: whether to mask ids with `maskSkippedIds`.
    pub fn initMulMatIdIds(ti: *mod.TensorInit, ctx: *mod.c.ggml_context, n_mats: i32, skip: bool) void {
        var it = Self.tensors(ctx);
        while (it.next()) |t| {
            if (t.type != mod.c.GGML_TYPE_I32 or Self.isViewOp(t.op)) continue;
            const n: usize = @intCast(t.ne[0]);
            const data = ti.allocator.alloc(i32, n) catch @panic("OOM");
            defer ti.allocator.free(data);
            var r: i64 = 0;
            while (r < mod.c.ggml_nrows(t)) : (r += 1) {
                for (data, 0..) |*d, i| d.* = @intCast(@rem(@as(i64, @intCast(i)), @as(i64, n_mats)));
                ti.random.shuffle(i32, data);
                if (skip) Self.maskSkippedIds(data, r);
                mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], n * @sizeOf(i32));
            }
        }
    }

    /// Fills the non-I32 tensors of `ctx` uniformly and the I32 ones with
    /// expert ids (`init_mul_mat_id_tensors`).
    ///
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    /// - `n_mats`: the number of experts.
    /// - `skip`: whether to mask some ids as skipped.
    pub fn initMulMatIdTensors(ti: *mod.TensorInit, ctx: *mod.c.ggml_context, n_mats: i32, skip: bool) void {
        var it = Self.tensors(ctx);
        while (it.next()) |t| {
            if (t.type != mod.c.GGML_TYPE_I32) ti.uniform(t, -1.0, 1.0);
        }
        Self.initMulMatIdIds(ti, ctx, n_mats, skip);
    }

    /// Whether `op` only reinterprets its source (`ggml_is_view_op`).
    ///
    /// - `op`: the op.
    /// - Returns: true for VIEW, RESHAPE, PERMUTE and TRANSPOSE.
    pub fn isViewOp(op: mod.c.enum_ggml_op) bool {
        return op == mod.c.GGML_OP_VIEW or op == mod.c.GGML_OP_RESHAPE or op == mod.c.GGML_OP_PERMUTE or op == mod.c.GGML_OP_TRANSPOSE;
    }

    /// Iterates the tensors of `ctx`: `while (it.next()) |t|`.
    ///
    /// - `ctx`: the context.
    /// - Returns: the iterator.
    pub fn tensors(ctx: *mod.c.ggml_context) TensorIterator {
        return .{ .ctx = ctx, .next_tensor = mod.c.ggml_get_first_tensor(ctx) };
    }
};

/// An iterator over a context's tensors; see `Helpers.tensors`.
pub const TensorIterator = struct {
    ctx: *mod.c.ggml_context,
    next_tensor: ?*mod.c.ggml_tensor,

    /// The next tensor, or null at the end.
    ///
    /// - `self`: the iterator.
    /// - Returns: the tensor.
    pub fn next(self: *TensorIterator) ?*mod.c.ggml_tensor {
        const t = self.next_tensor orelse return null;
        self.next_tensor = mod.c.ggml_get_next_tensor(self.ctx, t);
        return t;
    }
};
