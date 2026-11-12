const std = @import("std");
const mod = @import("module.zig");

/// Formats a bench case's parameters the way `test-backend-ops`'
/// `VARS_TO_STR` macros do: `name=value` pairs joined by commas.
///
/// Values follow upstream's `var_to_str`: integers in decimal, `bool` as
/// `1`/`0`, floats as `%f` (six decimals), arrays and slices as `[a,b,c]`,
/// strings verbatim, `GgmlType` by name, `GgmlPrec` as `f32`/`def`, `OpPool`
/// as `avg`/`max`, and `ScaleMode` as `bilinear|align_corners`.
pub const Vars = struct {
    const Self = @This();

    writer: *std.Io.Writer,
    first: bool = true,

    /// Starts a list on `writer`.
    ///
    /// - `writer`: the destination.
    /// - Returns: an empty list.
    pub fn init(writer: *std.Io.Writer) Self {
        return Self{ .writer = writer };
    }

    /// Writes the named fields of `case`, in order: the equivalent of
    /// `VARS_TO_STRn(a, b, ...)` for fields of the same names.
    ///
    /// - `writer`: the destination.
    /// - `case`: the case, a pointer to a struct.
    /// - `names`: the field names, as a tuple of string literals.
    pub fn fields(writer: *std.Io.Writer, case: anytype, comptime names: anytype) !void {
        var self = Self.init(writer);
        inline for (names) |field_name| {
            try self.add(field_name, @field(case, field_name));
        }
    }

    /// Writes one `name=value` pair, preceded by a comma unless it is the first.
    ///
    /// - `self`: the list.
    /// - `name`: the name printed before `=`.
    /// - `value`: the value, formatted as described on the struct.
    pub fn add(self: *Self, name: []const u8, value: anytype) !void {
        if (!self.first) try self.writer.writeByte(',');
        self.first = false;
        try self.writer.print("{s}=", .{name});
        try Self.format(self.writer, value);
    }

    /// Writes one value as upstream's `var_to_str` would.
    ///
    /// - `writer`: the destination.
    /// - `v`: the value.
    pub fn format(writer: *std.Io.Writer, v: anytype) !void {
        const T = @TypeOf(v);
        if (T == mod.GgmlType or T == mod.GgmlPrec) return writer.writeAll(v.name());
        if (T == mod.OpPool or T == mod.ScaleMode) return v.write(writer);
        switch (@typeInfo(T)) {
            .bool => try writer.writeByte(if (v) '1' else '0'),
            .int, .comptime_int => try writer.print("{d}", .{v}),
            .float, .comptime_float => try writer.print("{d:.6}", .{v}),
            .@"enum" => try writer.print("{d}", .{@backingInt(v)}),
            .array => |info| {
                if (info.child == u8) return writer.writeAll(&v);
                try Self.list(writer, &v);
            },
            .pointer => |info| {
                if (info.size == .one) {
                    switch (@typeInfo(info.child)) {
                        .array => |arr| {
                            if (arr.child == u8) return writer.writeAll(v);
                            return Self.list(writer, v);
                        },
                        else => return Self.format(writer, v.*),
                    }
                }
                if (info.child == u8) return writer.writeAll(v);
                try Self.list(writer, v);
            },
            .optional => if (v) |inner| try Self.format(writer, inner) else try writer.writeAll("null"),
            else => @compileError("Vars: cannot format " ++ @typeName(T)),
        }
    }

    fn list(writer: *std.Io.Writer, items: anytype) std.Io.Writer.Error!void {
        try writer.writeByte('[');
        for (items, 0..) |item, i| {
            if (i > 0) try writer.writeByte(',');
            try Self.format(writer, item);
        }
        try writer.writeByte(']');
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "vars match test-backend-ops' formatting" {
    const Example = struct {
        type: mod.GgmlType = .of(mod.c.GGML_TYPE_F16),
        ne: [4]i64 = .{ 10, 5, 4, 3 },
        v: i32 = 1,
        inplace: bool = false,
        eps: f32 = 1e-6,
    };
    const example: Example = .{};

    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try Vars.fields(&out.writer, &example, .{ "type", "ne", "v", "inplace", "eps" });
    try std.testing.expectEqualStrings("type=f16,ne=[10,5,4,3],v=1,inplace=0,eps=0.000001", out.written());
}
