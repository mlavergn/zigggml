const std = @import("std");
const mod = @import("module.zig");

/// A `ggml_op_pool` held by a bench case; `vars` prints `avg`, `max` or the
/// number, as `test-backend-ops` does.
pub const OpPool = enum(mod.c.enum_ggml_op_pool) {
    _,

    const Self = @This();

    /// Wraps a C `ggml_op_pool` value.
    ///
    /// - `value`: a `GGML_OP_POOL_*` constant.
    /// - Returns: the wrapped pool kind.
    pub fn of(value: anytype) Self {
        return @fromBackingInt(@intCast(value));
    }

    /// The C value, for calls into ggml.
    ///
    /// - `self`: the wrapped pool kind.
    /// - Returns: the `ggml_op_pool`.
    pub fn raw(self: Self) mod.c.enum_ggml_op_pool {
        return @backingInt(self);
    }

    /// Writes the name `test-backend-ops` prints.
    ///
    /// - `self`: the wrapped pool kind.
    /// - `writer`: the destination.
    pub fn write(self: Self, writer: *std.Io.Writer) !void {
        switch (self.raw()) {
            mod.c.GGML_OP_POOL_AVG => try writer.writeAll("avg"),
            mod.c.GGML_OP_POOL_MAX => try writer.writeAll("max"),
            else => try writer.print("{d}", .{self.raw()}),
        }
    }
};
