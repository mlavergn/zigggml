//! Port of `test_top_k_inf` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_TOP_K.

const std = @import("std");
const mod = @import("module.zig");

/// GGML_OP_TOP_K over rows like log-probabilities: distinct negative values,
/// fewer than `k` `+inf` (none for `k = 1`, so the expected indices are
/// unique) and many `-inf` (masked tokens).
pub const TestTopKInf = struct {
    const Self = @This();

    // Constructor parameters, in order (the C++ has no defaults).
    ne: [4]i64 = .{ 0, 0, 0, 0 },
    k: i32 = 0,

    // State.
    /// The `test_top_k` this case extends: F32, `ne`, `k`, no ties.
    base: mod.TestTopK = .{},

    /// Builds the base case from the parameters.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.base = .{ .type = .of(mod.c.GGML_TYPE_F32), .ne = self.ne, .k = self.k, .ties = false };
    }

    /// Writes the base case's parameters, then `inf=1`.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        try self.base.vars(w);
        try w.writeAll(",inf=1");
    }

    /// The results must match exactly.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxErr(self: *Self) f64 {
        return self.base.maxErr();
    }

    /// Compares only the output: the input holds infinities, which `err`
    /// would read as indices.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The base case's set comparison of indices.
    ///
    /// - `self`: the case.
    /// - `allocator`: scratch allocator.
    /// - `a`: the reference.
    /// - `b`: the result.
    /// - Returns: the error.
    pub fn err(self: *Self, allocator: std.mem.Allocator, a: []const f32, b: []const f32) f64 {
        return self.base.err(allocator, a, b);
    }

    /// Builds `top_k(a, k)`.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        return self.base.buildGraph(g);
    }

    /// Fills each row with shuffled distinct negative values, `k / 2` of
    /// them `+inf` and about a third `-inf`.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        const n_pinf: usize = @intCast(@divTrunc(self.k, 2));
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            const n: usize = @intCast(t.ne[0]);
            const data = ti.allocator.alloc(f32, n) catch @panic("OOM");
            defer ti.allocator.free(data);
            var r: i64 = 0;
            while (r < mod.c.ggml_nrows(t)) : (r += 1) {
                for (data, 0..) |*d, i| d.* = -1.0 - @as(f32, @floatFromInt(i));
                ti.random.shuffle(f32, data);
                for (data, 0..) |*d, i| {
                    if (i < n_pinf) {
                        d.* = std.math.inf(f32);
                    } else if (i % 3 == 0) {
                        d.* = -std.math.inf(f32);
                    }
                }
                ti.random.shuffle(f32, data);
                mod.c.ggml_backend_tensor_set(t, data.ptr, @as(usize, @intCast(r)) * t.nb[1], n * @sizeOf(f32));
            }
        }
    }
};
