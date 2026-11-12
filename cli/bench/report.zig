const std = @import("std");
const mod = @import("module.zig");

/// The output format.
pub const Format = enum { table, json };

/// Writes bench results as they arrive: an aligned table, or a JSON object
/// whose `results` array is streamed row by row.
pub const Report = struct {
    const Self = @This();

    writer: *std.Io.Writer,
    format: Format,
    rows: usize = 0,

    /// Binds a destination.
    ///
    /// - `writer`: the destination.
    /// - `format`: table or JSON.
    /// - Returns: the report.
    pub fn init(writer: *std.Io.Writer, format: Format) Self {
        return Self{ .writer = writer, .format = format };
    }

    /// What the header describes.
    pub const Header = struct {
        version: []const u8,
        default_n_threads: u32,
        backends: []const []const u8,
        cases: usize,
        warmup: u32,
        runs: u32,
    };

    /// Writes the header: the build, the backends and the run shape.
    ///
    /// - `self`: the report.
    /// - `header`: what to describe.
    pub fn begin(self: *Self, header: Header) !void {
        switch (self.format) {
            .table => {
                try self.writer.print("zigggml {s} · GGML_DEFAULT_N_THREADS={d} · backends:", .{ header.version, header.default_n_threads });
                for (header.backends) |b| try self.writer.print(" {s}", .{b});
                try self.writer.print(" · {d} cases · {d} warmup + {d} timed runs\n\n", .{ header.cases, header.warmup, header.runs });
                try self.writer.print("{s:<10} {s:>4} {s:<22} {s:>6} {s:>11} {s:>11} {s:>11} {s:>6} {s:>5} {s:>14} {s:<5} {s}\n", .{
                    "BACKEND", "THR", "OP", "N/RUN", "MIN us", "MEDIAN us", "MAX us", "CPU%", "GPU%", "THROUGHPUT", "CHECK", "VARS",
                });
            },
            .json => {
                try self.writer.writeAll("{\"version\":");
                try std.json.Stringify.value(header.version, .{}, self.writer);
                try self.writer.print(",\"default_n_threads\":{d},\"warmup\":{d},\"runs\":{d},\"backends\":", .{ header.default_n_threads, header.warmup, header.runs });
                try std.json.Stringify.value(header.backends, .{}, self.writer);
                try self.writer.writeAll(",\"results\":[\n");
            },
        }
        try self.writer.flush();
    }

    /// Writes one row.
    ///
    /// - `self`: the report.
    /// - `row`: the result.
    pub fn row(self: *Self, r: *const mod.Row) !void {
        defer self.rows += 1;
        switch (self.format) {
            .table => try self.tableRow(r),
            .json => {
                if (self.rows > 0) try self.writer.writeAll(",\n");
                try std.json.Stringify.value(r.*, .{ .emit_null_optional_fields = false }, self.writer);
            },
        }
        try self.writer.flush();
    }

    /// Closes the output.
    ///
    /// - `self`: the report.
    pub fn end(self: *Self) !void {
        switch (self.format) {
            .table => try self.writer.print("\n{d} rows\n", .{self.rows}),
            .json => try self.writer.writeAll("\n]}\n"),
        }
        try self.writer.flush();
    }

    fn tableRow(self: *Self, r: *const mod.Row) !void {
        const w = self.writer;
        try w.print("{s:<10} ", .{r.backend});
        if (r.threads) |t| try w.print("{d:>4} ", .{t}) else try w.print("{s:>4} ", .{"-"});
        try w.print("{s:<22} ", .{r.op});
        if (!r.supported) {
            try w.print("{s:>6} {s:>11} {s:>11} {s:>11} {s:>6} {s:>5} {s:>14} {s:<5} {s}  ({s})\n", .{
                "-", "-", "-", "-", "-", "-", "-", checkName(r.check), r.vars, r.message,
            });
            return;
        }
        try w.print("{d:>6} {d:>11.2} {d:>11.2} {d:>11.2} {d:>6.0} ", .{ r.n_runs, r.min_us, r.median_us, r.max_us, r.cpu_pct });
        if (r.gpu_pct) |g| try w.print("{d:>5.0} ", .{g}) else try w.print("{s:>5} ", .{"n/a"});
        var buf: [32]u8 = undefined;
        const throughput = if (r.gflops) |g|
            std.fmt.bufPrint(&buf, "{d:.2} GFLOPS", .{g}) catch "?"
        else if (r.gbps) |g|
            std.fmt.bufPrint(&buf, "{d:.2} GB/s", .{g}) catch "?"
        else
            "-";
        try w.print("{s:>14} {s:<5} {s}\n", .{ throughput, checkName(r.check), r.vars });
    }

    fn checkName(c: mod.Check) []const u8 {
        return switch (c) {
            .none => "-",
            .pass => "pass",
            .fail => "FAIL",
            .unsupported => "n/s",
        };
    }
};
