//! Bench barrel: the `bench` command and the port of `test-backend-ops`'
//! machinery it runs on.

const std = @import("std");

// External dependencies.
pub const c = @import("ggml_c");
pub const config = @import("config");
pub const cpu = @import("../cpu.zig");

pub const Bench = @import("bench.zig").Bench;
pub const BenchOptions = @import("bench.zig").Options;
pub const Case = @import("case.zig").Case;
pub const CaseList = @import("case_list.zig").CaseList;
pub const Check = @import("runner.zig").Check;
pub const Compare = @import("compare.zig").Compare;
pub const EvalCases = @import("eval_cases.zig").EvalCases;
pub const EvalCasesHead = @import("eval_cases_head.zig").EvalCasesHead;
pub const EvalCasesTail = @import("eval_cases_tail.zig").EvalCasesTail;
pub const GgmlPrec = @import("ggml_prec.zig").GgmlPrec;
pub const GgmlType = @import("ggml_type.zig").GgmlType;
pub const Graph = @import("graph.zig").Graph;
pub const Helpers = @import("helpers.zig").Helpers;
pub const Nvml = @import("nvml.zig").Nvml;
pub const OpFilter = @import("op_filter.zig").OpFilter;
pub const OpPool = @import("op_pool.zig").OpPool;
pub const PerfCases = @import("perf_cases.zig").PerfCases;
pub const Report = @import("report.zig").Report;
pub const Row = @import("runner.zig").Row;
pub const Runner = @import("runner.zig").Runner;
pub const ScaleMode = @import("scale_mode.zig").ScaleMode;
pub const Settings = @import("runner.zig").Settings;
pub const TensorInit = @import("tensor_init.zig").TensorInit;
pub const TypeLists = @import("type_lists.zig").TypeLists;
pub const Vars = @import("vars.zig").Vars;

/// The case structs, one per `test_*` class.
pub const cases = @import("cases/module.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
