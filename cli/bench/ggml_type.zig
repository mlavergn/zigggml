const std = @import("std");
const mod = @import("module.zig");

/// A `ggml_type` held by a bench case, distinct from a plain integer so a
/// case's `vars` print it by name (`type=f32`), as `test-backend-ops` does.
pub const GgmlType = enum(mod.c.enum_ggml_type) {
    _,

    const Self = @This();

    /// Wraps a C `ggml_type` value.
    ///
    /// - `value`: a `GGML_TYPE_*` constant or any integer holding one.
    /// - Returns: the wrapped type.
    pub fn of(value: anytype) Self {
        return @fromBackingInt(@intCast(value));
    }

    /// The C value, for calls into ggml.
    ///
    /// - `self`: the wrapped type.
    /// - Returns: the `ggml_type`.
    pub fn raw(self: Self) mod.c.enum_ggml_type {
        return @backingInt(self);
    }

    /// The name ggml gives the type (`ggml_type_name`).
    ///
    /// - `self`: the wrapped type.
    /// - Returns: the name, static.
    pub fn name(self: Self) []const u8 {
        return std.mem.span(mod.c.ggml_type_name(self.raw()));
    }
};
