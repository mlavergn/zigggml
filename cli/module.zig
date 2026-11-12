//! CLI barrel for the zigggml executable.
//!
//! Re-exports every public type in `cli/` so the entry point imports through
//! this module rather than reaching for sibling files directly.

const std = @import("std");

// External dependencies.
pub const c = @import("ggml_c");
pub const config = @import("config");

pub const Client = @import("client.zig").Client;

/// The `bench` command and its case machinery.
pub const bench = @import("bench/module.zig");

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
