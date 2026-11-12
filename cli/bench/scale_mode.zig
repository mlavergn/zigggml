const std = @import("std");
const mod = @import("module.zig");

/// A `ggml_scale_mode` (with its flag bits) held by a bench case; `vars`
/// prints it as `test-backend-ops` does: `bilinear|align_corners`.
pub const ScaleMode = enum(u32) {
    _,

    const Self = @This();

    /// Wraps a C `ggml_scale_mode` value, flags included.
    ///
    /// - `value`: a `GGML_SCALE_MODE_*` constant, optionally or-ed with `GGML_SCALE_FLAG_*`.
    /// - Returns: the wrapped mode.
    pub fn of(value: anytype) Self {
        return @fromBackingInt(@intCast(value));
    }

    /// The C value, for calls into ggml.
    ///
    /// - `self`: the wrapped mode.
    /// - Returns: the mode as the `uint32_t` ggml's scale functions take.
    pub fn raw(self: Self) u32 {
        return @backingInt(self);
    }

    /// Writes the name `test-backend-ops` prints.
    ///
    /// - `self`: the wrapped mode.
    /// - `writer`: the destination.
    pub fn write(self: Self, writer: *std.Io.Writer) !void {
        const value = self.raw();
        switch (value & 0xFF) {
            mod.c.GGML_SCALE_MODE_NEAREST => try writer.writeAll("nearest"),
            mod.c.GGML_SCALE_MODE_BILINEAR => try writer.writeAll("bilinear"),
            mod.c.GGML_SCALE_MODE_BICUBIC => try writer.writeAll("bicubic"),
            else => try writer.print("{d}", .{value}),
        }
        if (value & mod.c.GGML_SCALE_FLAG_ALIGN_CORNERS != 0) try writer.writeAll("|align_corners");
        if (value & mod.c.GGML_SCALE_FLAG_ANTIALIAS != 0) try writer.writeAll("|antialias");
    }
};
