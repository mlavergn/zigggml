const std = @import("std");
const mod = @import("module.zig");

/// A `ggml_prec` held by a bench case; `vars` prints it as `test-backend-ops`
/// does: `f32` for `GGML_PREC_F32`, otherwise `def`.
pub const GgmlPrec = enum(mod.c.enum_ggml_prec) {
    _,

    const Self = @This();

    /// Wraps a C `ggml_prec` value.
    ///
    /// - `value`: a `GGML_PREC_*` constant.
    /// - Returns: the wrapped precision.
    pub fn of(value: anytype) Self {
        return @fromBackingInt(@intCast(value));
    }

    /// The C value, for calls into ggml.
    ///
    /// - `self`: the wrapped precision.
    /// - Returns: the `ggml_prec`.
    pub fn raw(self: Self) mod.c.enum_ggml_prec {
        return @backingInt(self);
    }

    /// The name `test-backend-ops` prints.
    ///
    /// - `self`: the wrapped precision.
    /// - Returns: `f32` or `def`.
    pub fn name(self: Self) []const u8 {
        return if (self.raw() == mod.c.GGML_PREC_F32) "f32" else "def";
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "prec names" {
    try std.testing.expectEqualStrings("f32", GgmlPrec.of(mod.c.GGML_PREC_F32).name());
    try std.testing.expectEqualStrings("def", GgmlPrec.of(mod.c.GGML_PREC_DEFAULT).name());
}
