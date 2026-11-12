const std = @import("std");
const log = std.log.scoped(.zigggml_runner);
const mod = @import("module.zig");

/// How a backend fared against the CPU on one case.
pub const Check = enum {
    /// Not compared: the backend is the CPU, or checking was not asked for.
    none,
    /// Within the case's error bound.
    pass,
    /// Outside it, or a sentinel, NaN or infinity mismatch.
    fail,
    /// The backend or the CPU does not support an op in the graph.
    unsupported,
};

/// One row of bench output: a case timed on a backend at a thread count.
pub const Row = struct {
    /// The backend's name (`ggml_backend_name`).
    backend: []const u8,
    /// The CPU thread count, or null for a non-CPU backend.
    threads: ?u32 = null,
    /// The op description.
    op: []const u8,
    /// The case's parameters.
    vars: []const u8,
    /// False when the backend does not support the op; nothing was timed.
    supported: bool = true,
    /// Why the case could not be timed, when it could not.
    message: []const u8 = "",
    /// Op executions per timed graph (the op is duplicated to reach a work target).
    n_runs: u32 = 0,
    /// Timed graph runs.
    runs: u32 = 0,
    /// Per-op wall time over the timed runs, in µs.
    min_us: f64 = 0,
    median_us: f64 = 0,
    max_us: f64 = 0,
    /// Process CPU time over wall time across the timed runs, in percent
    /// (above 100 with several threads busy).
    cpu_pct: f64 = 0,
    /// Mean GPU utilization over the timed runs, when measurable.
    gpu_pct: ?f64 = null,
    /// GFLOPS at the median time, for ops with a FLOP count.
    gflops: ?f64 = null,
    /// GB/s (GiB, as `test-backend-ops` reports) at the median time, for the rest.
    gbps: ?f64 = null,
    /// The comparison against the CPU.
    check: Check = .none,
};

/// Timing and checking knobs.
pub const Settings = struct {
    /// Untimed runs before timing.
    warmup: u32 = 1,
    /// Timed runs.
    runs: u32 = 10,
    /// Whether each op executes once per graph instead of being duplicated
    /// up to the work target (the smoke run).
    single: bool = false,
};

/// Runs cases on backends: the port of `test_case::eval` (the comparison)
/// and `test_case::eval_perf` (the timing), with `eval_perf`'s "run for at
/// least a second" replaced by a fixed warmup and number of timed runs.
pub const Runner = struct {
    const Self = @This();

    /// Graph capacity for timing (`eval_perf`'s `graph_nodes`).
    const graph_nodes: usize = 8192;

    allocator: std.mem.Allocator,
    io: std.Io,
    settings: Settings,
    tensor_init: mod.TensorInit,
    nvml: mod.Nvml,

    /// Prepares a runner.
    ///
    /// - `allocator`: allocator for per-case scratch; all of it is freed per case.
    /// - `io`: clocks and randomness.
    /// - `settings`: timing knobs.
    /// - `random`: the source of input data.
    /// - Returns: the runner.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, settings: Settings, random: std.Random) Self {
        return Self{
            .allocator = allocator,
            .io = io,
            .settings = settings,
            .tensor_init = mod.TensorInit.init(allocator, random),
            .nvml = mod.Nvml.init(),
        };
    }

    /// Releases NVML.
    ///
    /// - `self`: the runner.
    pub fn deinit(self: *Self) void {
        self.nvml.deinit();
    }

    /// Describes a case without running it: its op description and vars.
    ///
    /// - `self`: the runner.
    /// - `arena`: owns the returned strings.
    /// - `case`: the case.
    /// - Returns: the op description, the vars, and the bytes one op moves.
    pub fn describe(self: *Self, arena: std.mem.Allocator, case: mod.Case) !struct { op: []const u8, vars: []const u8, size: usize } {
        _ = self;
        const ctx = Self.newContext(Self.perfMemSize()) orelse return error.OutOfMemory;
        defer mod.c.ggml_free(ctx);
        const ctx_w = if (case.useWeightContext()) Self.newContext(Self.perfMemSize()) else null;
        defer if (ctx_w) |w| mod.c.ggml_free(w);
        var g = mod.Graph.init(arena, ctx, ctx_w, .perf);
        defer g.deinit();
        const out = try case.buildGraph(&g);
        return .{
            .op = try Self.opDesc(arena, case, out),
            .vars = try Self.varsOf(arena, case),
            .size = case.opSize(out),
        };
    }

    /// Compares `backend` with `cpu` on `case`, node by node, with sentinels
    /// around every tensor (`test_case::eval`).
    ///
    /// - `self`: the runner.
    /// - `case`: the case.
    /// - `backend`: the backend under test.
    /// - `cpu`: a CPU backend, the reference.
    /// - Returns: the outcome.
    pub fn check(self: *Self, case: mod.Case, backend: mod.c.ggml_backend_t, cpu: mod.c.ggml_backend_t) !Check {
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();

        const mem = mod.c.ggml_tensor_overhead() * 128 + mod.c.ggml_graph_overhead();
        const ctx = Self.newContext(mem) orelse return error.OutOfMemory;
        defer mod.c.ggml_free(ctx);
        const ctx_w = if (case.useWeightContext()) Self.newContext(mem) else null;
        defer if (ctx_w) |w| mod.c.ggml_free(w);

        var g = mod.Graph.init(arena.allocator(), ctx, ctx_w, .check);
        defer g.deinit();
        const gf: *mod.c.ggml_cgraph = mod.c.ggml_new_graph(ctx).?;
        g.gf = gf;

        g.addSentinel(ctx);
        if (ctx_w) |w| g.addSentinel(w);
        const out = try case.buildGraph(&g);

        for ([_]mod.c.ggml_backend_t{ backend, cpu }) |b| {
            var t: ?*mod.c.ggml_tensor = mod.c.ggml_get_first_tensor(ctx);
            while (t) |tensor| : (t = mod.c.ggml_get_next_tensor(ctx, tensor)) {
                if (!mod.c.ggml_backend_supports_op(b, tensor)) return .unsupported;
            }
        }

        g.addSentinel(ctx);
        if (ctx_w) |w| g.addSentinel(w);

        var buf_w: ?*mod.c.struct_ggml_backend_buffer = null;
        defer if (buf_w) |bw| mod.c.ggml_backend_buffer_free(bw);
        if (ctx_w) |w| {
            buf_w = mod.c.ggml_backend_alloc_ctx_tensors(w, backend) orelse return error.AllocationFailed;
            mod.c.ggml_backend_buffer_set_usage(buf_w, mod.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        const buf = mod.c.ggml_backend_alloc_ctx_tensors(ctx, backend) orelse return error.AllocationFailed;
        defer mod.c.ggml_backend_buffer_free(buf);

        mod.c.ggml_build_forward_expand(gf, out);
        for (g.sentinels.items) |s| mod.c.ggml_graph_add_node(gf, s);

        case.initializeTensors(&self.tensor_init, ctx);
        if (ctx_w) |w| case.initializeTensors(&self.tensor_init, w);

        if (Self.lacksSkippedIds(arena.allocator(), backend, gf)) return .unsupported;

        var ud: CompareData = .{ .case = case, .backend = backend, .allocator = arena.allocator() };
        var nodes = case.fusionTestNodes();
        const whole = case.runWholeGraph();
        var out_nodes = [_]*mod.c.ggml_tensor{out};
        if (nodes.len == 0 and whole) nodes = &out_nodes;
        const cmp_ok = mod.c.ggml_backend_compare_graph_backend(
            backend,
            cpu,
            gf,
            CompareData.callback,
            &ud,
            if (whole) @ptrCast(nodes.ptr) else null,
            nodes.len,
        );
        return if (ud.ok and cmp_ok) .pass else .fail;
    }

    /// Times `case` on `backend` (`test_case::eval_perf`).
    ///
    /// The op is added to the graph repeatedly until one graph run does a
    /// target amount of work (8 GFLOP on a CPU, 100 on a GPU; or 8/32 GiB of
    /// traffic for ops without a FLOP count), as upstream does. Then
    /// `settings.warmup` untimed and `settings.runs` timed graph runs follow;
    /// per-op times are graph times divided by the copies.
    ///
    /// - `self`: the runner.
    /// - `case`: the case.
    /// - `backend`: the backend.
    /// - `row`: filled with the result; its `backend`, `threads`, `op` and
    ///   `vars` are the caller's.
    pub fn perf(self: *Self, case: mod.Case, backend: mod.c.ggml_backend_t, row: *Row) !void {
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();

        const ctx = Self.newContext(Self.perfMemSize()) orelse return error.OutOfMemory;
        defer mod.c.ggml_free(ctx);
        const ctx_w = if (case.useWeightContext()) Self.newContext(Self.perfMemSize()) else null;
        defer if (ctx_w) |w| mod.c.ggml_free(w);

        var g = mod.Graph.init(arena.allocator(), ctx, ctx_w, .perf);
        defer g.deinit();
        const out = try case.buildGraph(&g);

        // Upstream asks about `out` alone, which suffices for its perf list.
        // Eval cases can wrap an unsupported node in a supported one, and
        // running it faults, so every node must be supported here, as in `check`.
        var next: ?*mod.c.ggml_tensor = mod.c.ggml_get_first_tensor(ctx);
        while (next) |tensor| : (next = mod.c.ggml_get_next_tensor(ctx, tensor)) {
            if (!mod.c.ggml_backend_supports_op(backend, tensor)) {
                row.supported = false;
                row.message = "not supported";
                return;
            }
        }

        var buf_w: ?*mod.c.struct_ggml_backend_buffer = null;
        defer if (buf_w) |bw| mod.c.ggml_backend_buffer_free(bw);
        if (ctx_w) |w| {
            buf_w = mod.c.ggml_backend_alloc_ctx_tensors(w, backend) orelse return error.AllocationFailed;
            mod.c.ggml_backend_buffer_set_usage(buf_w, mod.c.GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        const buf = mod.c.ggml_backend_alloc_ctx_tensors(ctx, backend) orelse return error.AllocationFailed;
        defer mod.c.ggml_backend_buffer_free(buf);

        case.initializeTensors(&self.tensor_init, ctx);
        if (ctx_w) |w| case.initializeTensors(&self.tensor_init, w);

        const gf: *mod.c.ggml_cgraph = mod.c.ggml_new_graph_custom(ctx, graph_nodes, false).?;
        mod.c.ggml_build_forward_expand(gf, out);

        // Upstream only checks this in `eval`, because its perf list has no
        // skipped expert ids; the eval list does, and CUDA faults on them.
        if (Self.lacksSkippedIds(arena.allocator(), backend, gf)) {
            row.supported = false;
            row.message = "skipped expert ids not supported";
            return;
        }

        const op_flops = case.opFlops(out);
        const op_size = @max(case.opSize(out), 1);
        const is_cpu = mod.c.ggml_backend_dev_type(mod.c.ggml_backend_get_device(backend)) == mod.c.GGML_BACKEND_DEVICE_TYPE_CPU;
        const room: u64 = @intCast(mod.c.ggml_graph_size(gf) - mod.c.ggml_graph_n_nodes(gf));
        const n_runs: u32 = if (self.settings.single) 1 else blk: {
            if (op_flops > 0) {
                const gflop: u64 = 1000 * 1000 * 1000;
                const target: u64 = if (is_cpu) 8 * gflop else 100 * gflop;
                break :blk @intCast(@min(room, target / op_flops) + 1);
            }
            const gib: u64 = 1 << 30;
            const target: u64 = if (is_cpu) 8 * gib else 32 * gib;
            break :blk @intCast(@min(room, target / op_size) + 1);
        };
        for (1..n_runs) |_| mod.c.ggml_graph_add_node(gf, out);

        // Bytes moved per graph run: the op's copies plus every other node.
        var mem: u64 = @as(u64, n_runs) * op_size;
        for (0..@intCast(mod.c.ggml_graph_n_nodes(gf))) |i| {
            const node: *mod.c.ggml_tensor = mod.c.ggml_graph_node(gf, @intCast(i));
            if (Self.isViewOp(node.op) or node == out) continue;
            mem += mod.Case.defaultOpSize(node);
        }

        for (0..self.settings.warmup) |_| try Self.compute(backend, gf);

        const runs = @max(self.settings.runs, 1);
        const times = try arena.allocator().alloc(f64, runs);
        const window = self.nvml.begin(backend);
        const wall_start = std.Io.Clock.awake.now(self.io);
        const cpu_start = std.Io.Clock.cpu_process.now(self.io);
        for (times) |*t| {
            const start = std.Io.Clock.awake.now(self.io);
            try Self.compute(backend, gf);
            const end = std.Io.Clock.awake.now(self.io);
            const ns = start.durationTo(end).nanoseconds;
            t.* = @as(f64, @floatFromInt(ns)) / 1000.0;
            case.reinitPerfIter(&self.tensor_init, ctx);
        }
        const cpu_ns = cpu_start.durationTo(std.Io.Clock.cpu_process.now(self.io)).nanoseconds;
        const wall_ns = wall_start.durationTo(std.Io.Clock.awake.now(self.io)).nanoseconds;
        row.gpu_pct = self.nvml.end(window);

        std.mem.sort(f64, times, {}, std.sort.asc(f64));
        const per_op: f64 = @floatFromInt(n_runs);
        const median_graph_us = Self.median(times);
        row.n_runs = n_runs;
        row.runs = runs;
        row.min_us = times[0] / per_op;
        row.median_us = median_graph_us / per_op;
        row.max_us = times[times.len - 1] / per_op;
        row.cpu_pct = if (wall_ns > 0) 100.0 * @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(wall_ns)) else 0;
        const median_s = median_graph_us / 1e6;
        if (median_s > 0) {
            if (op_flops > 0) {
                row.gflops = @as(f64, @floatFromInt(op_flops)) * per_op / median_s / 1e9;
            } else {
                row.gbps = @as(f64, @floatFromInt(mem)) / median_s / (1024.0 * 1024.0 * 1024.0);
            }
        }
    }

    /// The op description of a built case.
    fn opDesc(arena: std.mem.Allocator, case: mod.Case, out: *mod.c.ggml_tensor) ![]const u8 {
        var w: std.Io.Writer.Allocating = .init(arena);
        try case.opDesc(out, &w.writer);
        return w.written();
    }

    /// The vars of a case.
    fn varsOf(arena: std.mem.Allocator, case: mod.Case) ![]const u8 {
        var w: std.Io.Writer.Allocating = .init(arena);
        try case.vars(&w.writer);
        return w.written();
    }

    fn perfMemSize() usize {
        return mod.c.ggml_tensor_overhead() * 128 + mod.c.ggml_graph_overhead_custom(graph_nodes, false);
    }

    fn newContext(mem_size: usize) ?*mod.c.ggml_context {
        return mod.c.ggml_init(.{ .mem_size = mem_size, .mem_buffer = null, .no_alloc = true });
    }

    fn compute(backend: mod.c.ggml_backend_t, gf: *mod.c.ggml_cgraph) !void {
        const status = mod.c.ggml_backend_graph_compute(backend, gf);
        if (status != mod.c.GGML_STATUS_SUCCESS) {
            log.err("graph compute failed: {s}", .{std.mem.span(mod.c.ggml_status_to_string(status))});
            return error.ComputeFailed;
        }
    }

    fn median(sorted: []const f64) f64 {
        const n = sorted.len;
        if (n % 2 == 1) return sorted[n / 2];
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    fn isViewOp(op: mod.c.enum_ggml_op) bool {
        return op == mod.c.GGML_OP_VIEW or op == mod.c.GGML_OP_RESHAPE or op == mod.c.GGML_OP_PERMUTE or op == mod.c.GGML_OP_TRANSPOSE;
    }

    /// Whether `backend` is one that cannot yet skip `-1` expert ids, and the
    /// graph has one (`backend_lacks_skipped_ids`).
    fn lacksSkippedIds(arena: std.mem.Allocator, backend: mod.c.ggml_backend_t, gf: *mod.c.ggml_cgraph) bool {
        const name = std.mem.span(mod.c.ggml_backend_reg_name(mod.c.ggml_backend_dev_backend_reg(mod.c.ggml_backend_get_device(backend))));
        if (!std.mem.eql(u8, name, "CUDA") and !std.mem.eql(u8, name, "ROCm") and !std.mem.eql(u8, name, "MUSA")) return false;
        for (0..@intCast(mod.c.ggml_graph_n_nodes(gf))) |i| {
            const node: *mod.c.ggml_tensor = mod.c.ggml_graph_node(gf, @intCast(i));
            if (node.op != mod.c.GGML_OP_MUL_MAT_ID and node.op != mod.c.GGML_OP_ADD_ID) continue;
            const ids: *mod.c.ggml_tensor = node.src[2];
            const row = arena.alloc(i32, @intCast(ids.ne[0])) catch return false;
            var k1: i64 = 0;
            while (k1 < ids.ne[1]) : (k1 += 1) {
                mod.c.ggml_backend_tensor_get(ids, row.ptr, @as(usize, @intCast(k1)) * ids.nb[1], row.len * @sizeOf(i32));
                if (std.mem.findScalar(i32, row, -1) != null) return true;
            }
        }
        return false;
    }
};

/// What the comparison callback needs (`callback_userdata`).
const CompareData = struct {
    case: mod.Case,
    backend: mod.c.ggml_backend_t,
    allocator: std.mem.Allocator,
    ok: bool = true,

    /// Compares one node's two results: sentinels must be byte-identical,
    /// NaNs and infinities must agree, and the error must be within the
    /// case's bound.
    fn callback(index: c_int, t1_c: [*c]mod.c.ggml_tensor, t2_c: [*c]mod.c.ggml_tensor, user_data: ?*anyopaque) callconv(.c) bool {
        _ = index;
        const ud: *CompareData = @ptrCast(@alignCast(user_data.?));
        const t1: *mod.c.ggml_tensor = t1_c;
        const t2: *mod.c.ggml_tensor = t2_c;
        const desc = std.mem.span(mod.c.ggml_op_desc(t1));

        if (t1.op == mod.c.GGML_OP_NONE) {
            const n = mod.c.ggml_nbytes(t1);
            const d1 = ud.allocator.alloc(u8, n) catch return ud.fail("{s}: out of memory", .{desc});
            const d2 = ud.allocator.alloc(u8, n) catch return ud.fail("{s}: out of memory", .{desc});
            mod.c.ggml_backend_tensor_get(t1, d1.ptr, 0, n);
            mod.c.ggml_backend_tensor_get(t2, d2.ptr, 0, n);
            if (!std.mem.eql(u8, d1, d2)) return ud.fail("sentinel mismatch: {s}", .{std.mem.span(@as([*:0]const u8, @ptrCast(&t1.name)))});
        }

        const f1 = mod.Compare.tensorToFloat(ud.allocator, t1) catch return ud.fail("{s}: cannot read", .{desc});
        const f2 = mod.Compare.tensorToFloat(ud.allocator, t2) catch return ud.fail("{s}: cannot read", .{desc});
        for (f1, f2, 0..) |a, b, i| {
            if (std.math.isNan(a) or std.math.isNan(b)) return ud.fail("[{s}] NaN at index {d}", .{ desc, i });
            const inf_a = mod.Compare.isInfOrMax(a);
            const inf_b = mod.Compare.isInfOrMax(b);
            if (inf_a or inf_b) {
                if (!(inf_a and inf_b) or std.math.signbit(a) != std.math.signbit(b)) {
                    return ud.fail("[{s}] inf mismatch at index {d}", .{ desc, i });
                }
            }
        }
        const e = ud.case.err(ud.allocator, f1, f2);
        const bound = ud.case.maxErr(ud.backend);
        if (e > bound) _ = ud.fail("[{s}] ERR = {d:.9} > {d:.9}", .{ desc, e, bound });
        return true;
    }

    fn fail(self: *CompareData, comptime fmt: []const u8, args: anytype) bool {
        log.info(fmt, args);
        self.ok = false;
        return true;
    }
};
