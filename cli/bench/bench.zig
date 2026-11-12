const std = @import("std");
const log = std.log.scoped(.zigggml_bench);
const mod = @import("module.zig");

/// What `bench` was asked to do.
pub const Options = struct {
    /// Run the eval case list (`make_test_cases_eval`) instead of the perf list.
    full: bool = false,
    /// One small case per op on the CPU, once, untimed warmup skipped: the
    /// `make test` smoke run.
    smoke: bool = false,
    /// Only these backends (comma-separated names); null for all.
    backend: ?[]const u8 = null,
    /// The `--op` filter.
    op: ?[]const u8 = null,
    /// Only cases with a parameter of this ggml type.
    type: ?[]const u8 = null,
    /// CPU thread counts (comma-separated); null for the sweep.
    threads: ?[]const u8 = null,
    /// Untimed runs per case.
    warmup: u32 = 1,
    /// Timed runs per case.
    runs: u32 = 10,
    /// Machine-readable output.
    json: bool = false,
    /// Skip the comparison against the CPU.
    no_check: bool = false,
};

/// A backend the bench runs on.
const Target = struct {
    backend: mod.c.ggml_backend_t,
    name: []const u8,
    is_cpu: bool,
};

/// The `bench` command: times every case of the chosen list on every
/// selected backend, sweeping CPU thread counts, and checks each non-CPU
/// backend's results against the CPU.
pub const Bench = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    options: Options,

    /// Prepares a run.
    ///
    /// - `allocator`: allocator for the run's state.
    /// - `io`: clocks and randomness.
    /// - `options`: what to run.
    /// - Returns: the command.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, options: Options) Self {
        return Self{ .allocator = allocator, .io = io, .options = options };
    }

    /// Releases the command.
    ///
    /// - `self`: the command.
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Parses `bench`'s arguments.
    ///
    /// - `args`: the arguments after `bench`.
    /// - Returns: the options; `error.InvalidArgument` for an unknown flag or
    ///   a missing or malformed value.
    pub fn parse(args: []const [:0]const u8) !Options {
        var o: Options = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const a: []const u8 = args[i];
            if (std.mem.eql(u8, a, "--full")) {
                o.full = true;
            } else if (std.mem.eql(u8, a, "--smoke")) {
                o.smoke = true;
            } else if (std.mem.eql(u8, a, "--json")) {
                o.json = true;
            } else if (std.mem.eql(u8, a, "--no-check")) {
                o.no_check = true;
            } else if (std.mem.eql(u8, a, "--backend") or std.mem.eql(u8, a, "-b")) {
                o.backend = try Self.value(args, &i);
            } else if (std.mem.eql(u8, a, "--op") or std.mem.eql(u8, a, "-o")) {
                o.op = try Self.value(args, &i);
            } else if (std.mem.eql(u8, a, "--type")) {
                o.type = try Self.value(args, &i);
            } else if (std.mem.eql(u8, a, "--threads") or std.mem.eql(u8, a, "-t")) {
                o.threads = try Self.value(args, &i);
            } else if (std.mem.eql(u8, a, "--warmup")) {
                o.warmup = std.fmt.parseInt(u32, try Self.value(args, &i), 10) catch return error.InvalidArgument;
            } else if (std.mem.eql(u8, a, "--runs")) {
                o.runs = std.fmt.parseInt(u32, try Self.value(args, &i), 10) catch return error.InvalidArgument;
                if (o.runs == 0) return error.InvalidArgument;
            } else {
                return error.InvalidArgument;
            }
        }
        return o;
    }

    /// Runs the bench and writes its report.
    ///
    /// - `self`: the command.
    /// - `writer`: the destination.
    pub fn run(self: *Self, writer: *std.Io.Writer) !void {
        var list = mod.CaseList.init(self.allocator);
        defer list.deinit();
        if (self.options.full or self.options.smoke) {
            try mod.EvalCases.build(&list);
        } else {
            try mod.PerfCases.build(&list);
        }

        var targets: std.ArrayList(Target) = .empty;
        defer {
            for (targets.items) |t| mod.c.ggml_backend_free(t.backend);
            targets.deinit(self.allocator);
        }
        try self.openTargets(&targets);
        if (targets.items.len == 0) {
            log.err("bench: no backend matches --backend {s}", .{self.options.backend orelse ""});
            return error.NoBackend;
        }
        const cpu_ref: mod.c.ggml_backend_t = mod.c.ggml_backend_init_by_type(mod.c.GGML_BACKEND_DEVICE_TYPE_CPU, null) orelse return error.NoBackend;
        defer mod.c.ggml_backend_free(cpu_ref);

        const smoke = self.options.smoke;
        const settings: mod.Settings = if (smoke)
            .{ .warmup = 0, .runs = 1, .single = true }
        else
            .{ .warmup = self.options.warmup, .runs = self.options.runs };

        var prng: std.Random.DefaultPrng = .init(seed: {
            var bytes: [8]u8 = undefined;
            self.io.random(&bytes);
            break :seed std.mem.readInt(u64, &bytes, .little);
        });
        var runner = mod.Runner.init(self.allocator, self.io, settings, prng.random());
        defer runner.deinit();

        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();

        // Describe every case once: op, vars, size; filter; for the smoke
        // run, keep the smallest case of each op.
        const Selected = struct { index: usize, op: []const u8, vars: []const u8, size: usize };
        var selected: std.ArrayList(Selected) = .empty;
        const filter = mod.OpFilter.init(self.options.op);
        for (list.items.items, 0..) |case, i| {
            const d = try runner.describe(arena.allocator(), case);
            const full_name = try std.fmt.allocPrint(arena.allocator(), "{s}({s})", .{ d.op, d.vars });
            if (!filter.matches(d.op, full_name)) continue;
            if (self.options.type) |ty| if (!Self.hasType(d.vars, ty)) continue;
            if (smoke) {
                const existing = for (selected.items) |*s| {
                    if (std.mem.eql(u8, s.op, d.op)) break s;
                } else null;
                if (existing) |s| {
                    if (d.size < s.size) s.* = .{ .index = i, .op = d.op, .vars = d.vars, .size = d.size };
                    continue;
                }
            }
            try selected.append(arena.allocator(), .{ .index = i, .op = d.op, .vars = d.vars, .size = d.size });
        }

        var names: std.ArrayList([]const u8) = .empty;
        for (targets.items) |t| try names.append(arena.allocator(), t.name);
        var report = mod.Report.init(writer, if (self.options.json) .json else .table);
        try report.begin(.{
            .version = mod.config.version,
            .default_n_threads = mod.config.default_n_threads,
            .backends = names.items,
            .cases = selected.items.len,
            .warmup = settings.warmup,
            .runs = settings.runs,
        });

        const sweep = try self.threadCounts(arena.allocator());
        for (selected.items) |s| {
            const case = list.items.items[s.index];
            for (targets.items) |t| {
                var verdict: mod.Check = .none;
                if (!t.is_cpu and !self.options.no_check and !smoke) {
                    verdict = runner.check(case, t.backend, cpu_ref) catch |e| blk: {
                        log.warn("{s} {s}: check failed: {any}", .{ t.name, s.op, e });
                        break :blk .fail;
                    };
                }
                const counts: []const u32 = if (t.is_cpu) sweep else &.{0};
                for (counts) |n| {
                    if (t.is_cpu) mod.c.ggml_backend_cpu_set_n_threads(t.backend, @intCast(n));
                    var row: mod.Row = .{
                        .backend = t.name,
                        .threads = if (t.is_cpu) n else null,
                        .op = s.op,
                        .vars = s.vars,
                        .check = verdict,
                    };
                    runner.perf(case, t.backend, &row) catch |e| {
                        row.supported = false;
                        row.message = @errorName(e);
                    };
                    try report.row(&row);
                }
            }
        }
        try report.end();
    }

    /// Opens a backend for every device the `--backend` filter admits (all
    /// when there is none); the smoke run uses the CPU alone.
    fn openTargets(self: *Self, targets: *std.ArrayList(Target)) !void {
        const count = mod.c.ggml_backend_dev_count();
        for (0..count) |i| {
            const dev = mod.c.ggml_backend_dev_get(i);
            const is_cpu = mod.c.ggml_backend_dev_type(dev) == mod.c.GGML_BACKEND_DEVICE_TYPE_CPU;
            if (self.options.smoke and !is_cpu) continue;
            const dev_name = std.mem.span(mod.c.ggml_backend_dev_name(dev));
            if (self.options.backend) |want| if (!Self.listHas(want, dev_name)) continue;
            const backend: mod.c.ggml_backend_t = mod.c.ggml_backend_dev_init(dev, null) orelse {
                log.warn("cannot initialize {s}", .{dev_name});
                continue;
            };
            try targets.append(self.allocator, .{ .backend = backend, .name = std.mem.span(mod.c.ggml_backend_name(backend)), .is_cpu = is_cpu });
        }
    }

    /// The CPU thread counts to run: `--threads`, else (outside the smoke
    /// run) 1, 2, 4, ... up to `common_cpu_get_num_math()`, plus that count.
    fn threadCounts(self: *Self, arena: std.mem.Allocator) ![]const u32 {
        var counts: std.ArrayList(u32) = .empty;
        const math: u32 = @intCast(@max(mod.cpu.numMath(), 1));
        if (self.options.smoke) {
            try counts.append(arena, math);
        } else if (self.options.threads) |spec| {
            var it = std.mem.tokenizeScalar(u8, spec, ',');
            while (it.next()) |tok| {
                const n = std.fmt.parseInt(u32, std.mem.trim(u8, tok, " "), 10) catch return error.InvalidArgument;
                if (n == 0) return error.InvalidArgument;
                try counts.append(arena, n);
            }
        } else {
            var n: u32 = 1;
            while (n < math) : (n *= 2) try counts.append(arena, n);
            try counts.append(arena, math);
        }
        return counts.items;
    }

    /// Whether a comma-separated list names `name`, ignoring case.
    fn listHas(list: []const u8, name: []const u8) bool {
        var it = std.mem.tokenizeScalar(u8, list, ',');
        while (it.next()) |tok| {
            if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, tok, " "), name)) return true;
        }
        return false;
    }

    /// Whether a case's vars hold a parameter equal to type name `ty`.
    fn hasType(vars: []const u8, ty: []const u8) bool {
        var i: usize = 0;
        while (std.mem.findPos(u8, vars, i, ty)) |at| : (i = at + 1) {
            const before_ok = at > 0 and (vars[at - 1] == '=' or vars[at - 1] == '[' or vars[at - 1] == ',');
            const end = at + ty.len;
            const after_ok = end == vars.len or vars[end] == ',' or vars[end] == ']';
            if (before_ok and after_ok) return true;
        }
        return false;
    }

    fn value(args: []const [:0]const u8, i: *usize) ![]const u8 {
        if (i.* + 1 >= args.len) return error.InvalidArgument;
        i.* += 1;
        return args[i.*];
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "bench parses its flags" {
    const o = try Bench.parse(&.{ "--full", "--backend", "CPU,CUDA0", "--op", "MUL_MAT", "--threads", "1,8", "--runs", "3", "--json" });
    try std.testing.expect(o.full);
    try std.testing.expect(o.json);
    try std.testing.expectEqualStrings("CPU,CUDA0", o.backend.?);
    try std.testing.expectEqualStrings("MUL_MAT", o.op.?);
    try std.testing.expectEqualStrings("1,8", o.threads.?);
    try std.testing.expectEqual(@as(u32, 3), o.runs);
    try std.testing.expectError(error.InvalidArgument, Bench.parse(&.{"--nope"}));
    try std.testing.expectError(error.InvalidArgument, Bench.parse(&.{ "--runs", "0" }));
}

test "type filter matches whole parameter values" {
    try std.testing.expect(Bench.hasType("type_a=q4_0,type_b=f32", "q4_0"));
    try std.testing.expect(Bench.hasType("type_a=q4_0,type_b=f32", "f32"));
    try std.testing.expect(!Bench.hasType("type_a=q4_0_4,type_b=f32", "q4_0"));
    try std.testing.expect(Bench.hasType("types=[f16,q8_0]", "q8_0"));
}

test "backend list matching ignores case" {
    try std.testing.expect(Bench.listHas("cpu,CUDA0", "CPU"));
    try std.testing.expect(!Bench.listHas("CUDA0", "CPU"));
}
