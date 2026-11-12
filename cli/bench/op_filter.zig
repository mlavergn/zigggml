const std = @import("std");

/// The `--op` filter: the port of `test-backend-ops`' `-o` matching.
///
/// The filter is a comma-separated list. An entry matches an op when it is
/// the op's name exactly (`ADD`); when it contains `(`, it must equal the
/// full case string `NAME(vars)`; anything else is a pattern searched for in
/// the name. Upstream uses `std::regex`; this supports the subset its
/// filters use: literal characters, `.`, `*` after a character or `.`, and
/// the anchors `^` and `$`.
pub const OpFilter = struct {
    const Self = @This();

    /// The raw filter, or null to match everything.
    filter: ?[]const u8,

    /// Wraps a filter string.
    ///
    /// - `filter`: the comma-separated list, or null.
    /// - Returns: the filter.
    pub fn init(filter: ?[]const u8) Self {
        return Self{ .filter = filter };
    }

    /// Whether a case matches.
    ///
    /// - `self`: the filter.
    /// - `op_name`: the case's op description.
    /// - `full_name`: `op_name(vars)`.
    /// - Returns: true when any entry matches, or there is no filter.
    pub fn matches(self: Self, op_name: []const u8, full_name: []const u8) bool {
        const filter = self.filter orelse return true;
        var rest = filter;
        while (rest.len > 0) {
            var comma = std.mem.findScalar(u8, rest, ',') orelse rest.len;
            if (std.mem.findScalar(u8, rest, '(')) |lparen| {
                if (lparen < comma) {
                    const rparen = std.mem.findScalarPos(u8, rest, lparen, ')') orelse rest.len;
                    comma = std.mem.findScalarPos(u8, rest, rparen, ',') orelse rest.len;
                }
            }
            const entry = rest[0..comma];
            rest = if (comma < rest.len) rest[comma + 1 ..] else "";

            if (std.mem.findScalar(u8, entry, '(') != null) {
                if (std.mem.eql(u8, entry, full_name)) return true;
            } else if (Self.entryMatches(entry, op_name)) {
                return true;
            }
        }
        return false;
    }

    /// Whether one entry matches an op name (`op_filter_entry_matches`).
    ///
    /// - `entry`: the entry.
    /// - `op_name`: the op name.
    /// - Returns: true on a match.
    pub fn entryMatches(entry: []const u8, op_name: []const u8) bool {
        if (std.mem.eql(u8, entry, op_name)) return true;
        // Plain op names match exactly; anything else is a pattern.
        const plain = for (entry) |ch| {
            if (!(std.ascii.isUpper(ch) or std.ascii.isDigit(ch) or ch == '_')) break false;
        } else true;
        if (plain) return false;
        return Self.search(entry, op_name);
    }

    /// Searches `text` for `pattern` (the regex subset described above).
    fn search(pattern: []const u8, text: []const u8) bool {
        if (pattern.len > 0 and pattern[0] == '^') return Self.matchHere(pattern[1..], text);
        var i: usize = 0;
        while (true) : (i += 1) {
            if (Self.matchHere(pattern, text[i..])) return true;
            if (i >= text.len) return false;
        }
    }

    fn matchHere(pattern: []const u8, text: []const u8) bool {
        if (pattern.len == 0) return true;
        if (pattern.len >= 2 and pattern[1] == '*') return Self.matchStar(pattern[0], pattern[2..], text);
        if (pattern.len == 1 and pattern[0] == '$') return text.len == 0;
        if (text.len > 0 and (pattern[0] == '.' or pattern[0] == text[0])) return Self.matchHere(pattern[1..], text[1..]);
        return false;
    }

    fn matchStar(ch: u8, pattern: []const u8, text: []const u8) bool {
        var i: usize = 0;
        while (true) : (i += 1) {
            if (Self.matchHere(pattern, text[i..])) return true;
            if (i >= text.len or (ch != '.' and text[i] != ch)) return false;
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "op filter matches names, full strings and patterns" {
    const f = OpFilter.init("ADD,DSV4.*,MUL_MAT(type_a=f16,ne=[1,2])");
    try std.testing.expect(f.matches("ADD", "ADD(type=f32)"));
    try std.testing.expect(!f.matches("ADD1", "ADD1(type=f32)"));
    try std.testing.expect(f.matches("DSV4_HC_PRE", "DSV4_HC_PRE()"));
    try std.testing.expect(f.matches("MUL_MAT", "MUL_MAT(type_a=f16,ne=[1,2])"));
    try std.testing.expect(!f.matches("MUL_MAT", "MUL_MAT(type_a=f32,ne=[1,2])"));
    try std.testing.expect(OpFilter.init(null).matches("ANY", "ANY()"));
    try std.testing.expect(OpFilter.entryMatches("^SOFT.*$", "SOFT_MAX"));
}
