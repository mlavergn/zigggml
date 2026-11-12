const std = @import("std");
const mod = @import("module.zig");

/// The cases one bench run draws from, all owned by one arena: the port of
/// `make_test_cases_*`'s `std::vector<std::unique_ptr<test_case>>`.
pub const CaseList = struct {
    const Self = @This();

    arena: std.heap.ArenaAllocator,
    /// The cases, in the order they were added.
    items: std.ArrayList(mod.Case) = .empty,

    /// Starts an empty list.
    ///
    /// - `allocator`: the parent allocator of the list's arena.
    /// - Returns: the list.
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .arena = std.heap.ArenaAllocator.init(allocator) };
    }

    /// Frees every case.
    ///
    /// - `self`: the list.
    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// The allocator cases and their slices live in.
    ///
    /// - `self`: the list.
    /// - Returns: the arena's allocator.
    pub fn arenaAllocator(self: *Self) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Adds a case of type `T` built from positional arguments: the
    /// equivalent of `test_cases.emplace_back(new T(args...))`.
    ///
    /// `args[i]` is assigned to `T`'s i-th field; fields past the arguments
    /// keep their defaults. An integer argument for a `GgmlType`, `GgmlPrec`,
    /// `OpPool` or `ScaleMode` field is converted, so `GGML_TYPE_*` constants
    /// pass as they do in C++. `T.setup` runs afterwards when declared.
    ///
    /// - `self`: the list.
    /// - `T`: the case type.
    /// - `args`: the constructor arguments, as a tuple.
    pub fn add(self: *Self, comptime T: type, args: anytype) !void {
        try self.addValue(T, Self.fromArgs(T, args));
    }

    /// Adds an already-built case.
    ///
    /// - `self`: the list.
    /// - `T`: the case type.
    /// - `value`: the case; copied into the arena. `T.setup` runs on the copy
    ///   when declared.
    pub fn addValue(self: *Self, comptime T: type, value: T) !void {
        const p = try self.arenaAllocator().create(T);
        p.* = value;
        if (@hasDecl(T, "setup")) p.setup();
        try self.items.append(self.arenaAllocator(), mod.Case.of(T, p));
    }

    /// Copies `items` into the arena, for a case field that holds a slice
    /// built at run time (a C++ `std::vector` argument).
    ///
    /// - `self`: the list.
    /// - `T`: the element type.
    /// - `items`: the elements.
    /// - Returns: the copy, owned by the list.
    pub fn dupe(self: *Self, comptime T: type, items: []const T) ![]const T {
        return self.arenaAllocator().dupe(T, items);
    }

    /// Builds a `T` from positional arguments; see `add`.
    ///
    /// - `T`: the case type.
    /// - `args`: the arguments.
    /// - Returns: the case.
    pub fn fromArgs(comptime T: type, args: anytype) T {
        const info = @typeInfo(T).@"struct";
        const arg_names = @typeInfo(@TypeOf(args)).@"struct".field_names;
        if (arg_names.len > info.field_names.len) @compileError(@typeName(T) ++ ": too many arguments");
        var result: T = .{};
        inline for (arg_names, 0..) |arg_name, i| {
            const value = @field(args, arg_name);
            @field(result, info.field_names[i]) = Self.convert(info.field_types[i], value);
        }
        return result;
    }

    fn convert(comptime F: type, value: anytype) F {
        const V = @TypeOf(value);
        if (V == F) return value;
        if (F == mod.GgmlType or F == mod.GgmlPrec or F == mod.OpPool or F == mod.ScaleMode) {
            switch (@typeInfo(V)) {
                .int, .comptime_int => return F.of(value),
                else => {},
            }
        }
        switch (@typeInfo(F)) {
            .array => |info| {
                if (info.child == mod.GgmlType) {
                    var out: F = undefined;
                    inline for (0..info.len) |i| out[i] = Self.convert(info.child, value[i]);
                    return out;
                }
            },
            .float => switch (@typeInfo(V)) {
                .int, .comptime_int => return @floatFromInt(value),
                else => {},
            },
            .bool => switch (@typeInfo(V)) {
                .int, .comptime_int => return value != 0,
                else => {},
            },
            .int => switch (@typeInfo(V)) {
                .bool => return @intFromBool(value),
                .int => return @intCast(value),
                else => {},
            },
            else => {},
        }
        return value;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "fromArgs assigns positionally and keeps defaults" {
    const Example = struct {
        op: c_int = 0,
        type: mod.GgmlType = .of(mod.c.GGML_TYPE_F32),
        ne: [4]i64 = .{ 128, 2, 2, 2 },
        v: i32 = 0,
    };
    const e = CaseList.fromArgs(Example, .{ 7, mod.c.GGML_TYPE_F16, .{ 1, 2, 3, 4 } });
    try std.testing.expectEqual(@as(c_int, 7), e.op);
    try std.testing.expectEqual(mod.GgmlType.of(mod.c.GGML_TYPE_F16), e.type);
    try std.testing.expectEqual([4]i64{ 1, 2, 3, 4 }, e.ne);
    try std.testing.expectEqual(@as(i32, 0), e.v);
}
