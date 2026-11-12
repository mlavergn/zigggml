const std = @import("std");
const mod = @import("module.zig");

/// The type tables the case lists iterate: `test-backend-ops`' `all_types`,
/// `base_types` and `other_types` (tests/test-backend-ops.cpp @cd959bae9).
pub const TypeLists = struct {
    const Self = @This();

    /// Every type `test-backend-ops` exercises (`all_types`).
    pub const all_types = [_]mod.c.enum_ggml_type{
        mod.c.GGML_TYPE_F32,     mod.c.GGML_TYPE_F16,    mod.c.GGML_TYPE_BF16,
        mod.c.GGML_TYPE_Q4_0,    mod.c.GGML_TYPE_Q4_1,   mod.c.GGML_TYPE_Q5_0,
        mod.c.GGML_TYPE_Q5_1,    mod.c.GGML_TYPE_Q8_0,   mod.c.GGML_TYPE_Q1_0,
        mod.c.GGML_TYPE_Q2_0,    mod.c.GGML_TYPE_MXFP4,  mod.c.GGML_TYPE_NVFP4,
        mod.c.GGML_TYPE_F8_E4M3, mod.c.GGML_TYPE_Q2_K,   mod.c.GGML_TYPE_Q3_K,
        mod.c.GGML_TYPE_Q4_K,    mod.c.GGML_TYPE_Q5_K,   mod.c.GGML_TYPE_Q6_K,
        mod.c.GGML_TYPE_TQ2_0,   mod.c.GGML_TYPE_TQ1_0,  mod.c.GGML_TYPE_IQ2_XXS,
        mod.c.GGML_TYPE_IQ2_XS,  mod.c.GGML_TYPE_IQ2_S,  mod.c.GGML_TYPE_IQ3_XXS,
        mod.c.GGML_TYPE_IQ1_S,   mod.c.GGML_TYPE_IQ1_M,  mod.c.GGML_TYPE_IQ4_NL,
        mod.c.GGML_TYPE_IQ3_S,   mod.c.GGML_TYPE_IQ4_XS,
    };

    /// The types most cases are checked with (`base_types`).
    pub const base_types = [_]mod.c.enum_ggml_type{
        mod.c.GGML_TYPE_F32,   mod.c.GGML_TYPE_F16,     mod.c.GGML_TYPE_Q8_0,
        mod.c.GGML_TYPE_Q1_0,  mod.c.GGML_TYPE_Q2_0,    mod.c.GGML_TYPE_Q4_0,
        mod.c.GGML_TYPE_Q4_1,  mod.c.GGML_TYPE_Q4_K,    mod.c.GGML_TYPE_MXFP4,
        mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F8_E4M3, mod.c.GGML_TYPE_IQ2_XXS,
    };

    /// The remaining types, checked with fewer shapes (`other_types`).
    pub const other_types = [_]mod.c.enum_ggml_type{
        mod.c.GGML_TYPE_Q4_1,   mod.c.GGML_TYPE_Q5_0,   mod.c.GGML_TYPE_Q5_1,
        mod.c.GGML_TYPE_Q8_0,   mod.c.GGML_TYPE_Q1_0,   mod.c.GGML_TYPE_Q2_0,
        mod.c.GGML_TYPE_Q2_K,   mod.c.GGML_TYPE_Q3_K,   mod.c.GGML_TYPE_Q5_K,
        mod.c.GGML_TYPE_Q6_K,   mod.c.GGML_TYPE_TQ2_0,  mod.c.GGML_TYPE_TQ1_0,
        mod.c.GGML_TYPE_IQ2_XS, mod.c.GGML_TYPE_IQ2_S,  mod.c.GGML_TYPE_IQ3_XXS,
        mod.c.GGML_TYPE_IQ1_S,  mod.c.GGML_TYPE_IQ1_M,  mod.c.GGML_TYPE_IQ4_NL,
        mod.c.GGML_TYPE_IQ3_S,  mod.c.GGML_TYPE_IQ4_XS, mod.c.GGML_TYPE_BF16,
    };
};

// -----------------------------------------------------------------------------
// Unit Tests

test "type tables have upstream's sizes" {
    try std.testing.expectEqual(29, TypeLists.all_types.len);
    try std.testing.expectEqual(12, TypeLists.base_types.len);
    try std.testing.expectEqual(21, TypeLists.other_types.len);
}
