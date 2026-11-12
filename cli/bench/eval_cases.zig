//! Port of `make_test_cases_eval` (tests/test-backend-ops.cpp @cd959bae9).

const mod = @import("module.zig");

/// The eval case list: the shapes `test-backend-ops test` checks. Ported in
/// two halves, split at a top-level statement of the C++ function.
pub const EvalCases = struct {
    const Self = @This();

    /// Adds every eval case to `list`, in upstream's order.
    ///
    /// - `list`: the destination.
    pub fn build(list: *mod.CaseList) !void {
        try mod.EvalCasesHead.build(list);
        try mod.EvalCasesTail.build(list);
    }
};
