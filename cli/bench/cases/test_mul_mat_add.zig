//! Port of `test_mul_mat_add` (tests/test-backend-ops.cpp @cd959bae9): GGML_OP_MUL_MAT + GGML_OP_ADD.

const std = @import("std");
const mod = @import("module.zig");

/// How the residual is added to the mat-mul (upstream's `mul_mat_add_mode`).
pub const MulMatAddMode = enum(i32) {
    /// `mm + res`.
    mmRes = 0,
    /// `res + mm`.
    resMm = 1,
    /// `mm` + a one-row `res`, which a same-shape fusion must leave alone.
    row = 2,
    /// `res += mm`.
    resInplace = 3,
    /// `b += mm`: the sum overwrites the mat-mul input (m == k).
    bInplace = 4,
    /// `mm2 + mm`: the residual is itself a mat-mul output.
    mmMm = 5,

    /// The name upstream's `var_to_str` prints.
    ///
    /// - `self`: the mode.
    /// - Returns: the name, static.
    pub fn name(self: MulMatAddMode) []const u8 {
        return switch (self) {
            .mmRes => "mm+res",
            .resMm => "res+mm",
            .row => "mm+row",
            .resInplace => "res+=mm",
            .bInplace => "b+=mm",
            .mmMm => "mm2+mm",
        };
    }
};

/// `mul_mat` followed by an add of a residual, which backends may fuse into
/// the mat-mul.
pub const TestMulMatAdd = struct {
    const Self = @This();

    // Constructor parameters, in order, with the C++ defaults.
    type_a: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
    m: i64 = 0,
    n: i64 = 0,
    k: i64 = 0,
    mode: MulMatAddMode = .mmRes,

    /// Checks the constructor's invariant.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        std.debug.assert(self.mode != .bInplace or self.m == self.k);
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("type_a", self.type_a);
        try v.add("m", self.m);
        try v.add("n", self.n);
        try v.add("k", self.k);
        try v.add("mode", self.mode.name());
    }

    /// Always `MUL_MAT_ADD`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("MUL_MAT_ADD");
    }

    /// The fused graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: true.
    pub fn runWholeGraph(self: *Self) bool {
        _ = self;
        return true;
    }

    /// The bound for a mat-mul.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 5e-4;
    }

    /// Builds `mul_mat(a, b)` plus the residual, as `mode` says.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the output.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const a = g.newTensor2d(ctx, self.type_a, self.k, self.m);
        const b = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, self.k, self.n);
        const res = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, self.m, if (self.mode == .row) 1 else self.n);

        const mm: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, a, b);
        const out: *mod.c.ggml_tensor = switch (self.mode) {
            .mmRes, .row => mod.c.ggml_add(ctx, mm, res),
            .resMm => mod.c.ggml_add(ctx, res, mm),
            .resInplace => mod.c.ggml_add_inplace(ctx, res, mm),
            .bInplace => mod.c.ggml_add_inplace(ctx, b, mm),
            .mmMm => blk: {
                const a2 = g.newTensor2d(ctx, self.type_a, self.k, self.m);
                const mm2: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, a2, b);
                break :blk mod.c.ggml_add(ctx, mm2, mm);
            },
        };
        mod.Graph.setName(out, "out");
        return out;
    }
};
