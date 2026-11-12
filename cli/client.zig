const std = @import("std");
const log = std.log.scoped(.zigggml_client);
const cli = @import("module.zig");

/// The CLI's commands: `devices`, `bench`, `version`, `help`.
///
/// Everything the executable does lives here rather than in `main`, so the
/// whole behavior stays reachable from the unit tests below.
pub const Client = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,

    /// Constructs a client bound to the process capabilities.
    ///
    /// Parameters:
    /// - `allocator`: allocator for per-command state.
    /// - `io`: IO interface for output and timing.
    ///
    /// Return: an initialized `Client`; the error union is reserved for the
    /// lifecycle convention and yields no error today.
    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Self {
        return Self{ .allocator = allocator, .io = io };
    }

    /// Releases the client.
    ///
    /// Parameters:
    /// - `self`: the client to tear down.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Runs the command named by `args[1]`.
    ///
    /// Parameters:
    /// - `self`: the client.
    /// - `writer`: destination for the command's output; flushed before returning.
    /// - `args`: the full argument vector, program name included.
    ///
    /// Return: nothing on success; `error.UnknownCommand` for an unknown
    /// command; propagates write failures.
    pub fn run(self: *Self, writer: *std.Io.Writer, args: []const [:0]const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });

        const command: []const u8 = if (args.len > 1) args[1] else "help";
        if (std.mem.eql(u8, command, "devices")) {
            try self.devices(writer);
        } else if (std.mem.eql(u8, command, "version")) {
            try writer.print("zigggml {s}\n", .{cli.config.version});
        } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help")) {
            try writer.writeAll(usage);
        } else if (std.mem.eql(u8, command, "bench")) {
            const options = cli.bench.Bench.parse(args[2..]) catch |err| {
                try writer.writeAll(bench_usage);
                try writer.flush();
                return err;
            };
            var bench = cli.bench.Bench.init(self.allocator, self.io, options);
            defer bench.deinit();
            try bench.run(writer);
        } else {
            try writer.print("unknown command: {s}\n\n{s}", .{ command, usage });
            try writer.flush();
            return error.UnknownCommand;
        }
        try writer.flush();
    }

    /// Lists every registered backend device: name, type, description and
    /// memory.
    ///
    /// Parameters:
    /// - `self`: the client.
    /// - `writer`: destination for the listing.
    ///
    /// Return: nothing on success; propagates write failures.
    pub fn devices(self: *Self, writer: *std.Io.Writer) !void {
        _ = self;
        const count = cli.c.ggml_backend_dev_count();
        for (0..count) |i| {
            const dev = cli.c.ggml_backend_dev_get(i);
            var free: usize = 0;
            var total: usize = 0;
            cli.c.ggml_backend_dev_memory(dev, &free, &total);
            try writer.print("{s:<12} {s:<5} {s}  ({d} MiB free / {d} MiB)\n", .{
                std.mem.span(cli.c.ggml_backend_dev_name(dev)),
                devTypeName(cli.c.ggml_backend_dev_type(dev)),
                std.mem.span(cli.c.ggml_backend_dev_description(dev)),
                free / (1024 * 1024),
                total / (1024 * 1024),
            });
        }
    }
};

/// The short name of a device type.
///
/// Parameters:
/// - `t`: the type.
///
/// Return: `CPU`, `GPU`, `IGPU`, `ACCEL` or `?`.
pub fn devTypeName(t: cli.c.enum_ggml_backend_dev_type) []const u8 {
    return switch (t) {
        cli.c.GGML_BACKEND_DEVICE_TYPE_CPU => "CPU",
        cli.c.GGML_BACKEND_DEVICE_TYPE_GPU => "GPU",
        cli.c.GGML_BACKEND_DEVICE_TYPE_IGPU => "IGPU",
        cli.c.GGML_BACKEND_DEVICE_TYPE_ACCEL => "ACCEL",
        else => "?",
    };
}

const usage =
    \\usage: zigggml-cli <command> [options]
    \\
    \\commands:
    \\  devices   list backends and devices
    \\  bench     run and time every op's cases on every backend
    \\  version   print the version
    \\  help      print this message
    \\
;

const bench_usage =
    \\usage: zigggml-cli bench [options]
    \\
    \\Times every op's cases on every built backend (1 warmup + 10 timed runs),
    \\sweeping CPU thread counts, and checks each non-CPU backend against the CPU.
    \\
    \\options:
    \\  --full             the eval case list (make_test_cases_eval) instead of the perf list
    \\  --smoke            one small case per op, CPU only, one untimed-warmup-free run
    \\  -b, --backend L    only these backends (comma-separated device names, e.g. CPU,CUDA0)
    \\  -o, --op F         op filter: names, NAME(vars), or patterns (e.g. MUL_MAT,SOFT.*)
    \\  --type T           only cases with a parameter of ggml type T (e.g. q4_0)
    \\  -t, --threads L    CPU thread counts (e.g. 1,8); default 1,2,4,... up to the math cores
    \\  --warmup N         untimed runs per case (default 1)
    \\  --runs N           timed runs per case (default 10)
    \\  --json             JSON output
    \\  --no-check         skip the comparison against the CPU
    \\
;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Client);
}

test "the generated ggml.h carries config.zon's default thread count" {
    try std.testing.expectEqual(@as(c_int, @intCast(cli.config.default_n_threads)), cli.c.GGML_DEFAULT_N_THREADS);
}

test "devices lists at least the CPU" {
    const allocator = std.testing.allocator;

    var client = try Client.init(allocator, std.testing.io);
    defer client.deinit();

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();

    try client.run(&output.writer, &.{ "zigggml-cli", "devices" });
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "CPU") != null);
}

test "an unknown command is an error" {
    const allocator = std.testing.allocator;

    var client = try Client.init(allocator, std.testing.io);
    defer client.deinit();

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();

    try std.testing.expectError(error.UnknownCommand, client.run(&output.writer, &.{ "zigggml-cli", "nope" }));
}
