//! Type traits, and the shape and layout queries built on them.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml.c` (v0.9.0, `cd959bae9`), roughly
//! lines 631-1605. Each function names the C function it replaces and the line
//! it began at.
//!
//! # The traits table
//!
//! `type_traits` is the table every quantized format is described by: block
//! size, bytes per block, and the reference conversion routines. It is
//! generated from the C table rather than retyped, and the entries point at the
//! same `dequantize_row_*` and `quantize_row_*_ref` symbols the C does, which
//! still live in `ggml-quants.c`.
//!
//! `GGML_TYPE_COUNT` is 43 but only 35 entries are populated; the gaps are
//! retired formats and stay zeroed, exactly as the C designated initialisers
//! leave them.

const std = @import("std");
const impl = @import("impl.zig");
const c = impl.c;

const op_name = [_][*:0]const u8{
    "NONE",               "DUP",                "ADD",                     "ADD_ID",
    "ADD1",               "ACC",                "SUB",                     "MUL",
    "DIV",                "SQR",                "SQRT",                    "LOG",
    "SIN",                "COS",                "SUM",                     "SUM_ROWS",
    "CUMSUM",             "MEAN",               "ARGMAX",                  "COUNT_EQUAL",
    "REPEAT",             "REPEAT_BACK",        "CONCAT",                  "SILU_BACK",
    "NORM",               "RMS_NORM",           "RMS_NORM_BACK",           "GROUP_NORM",
    "L2_NORM",            "MUL_MAT",            "MUL_MAT_ID",              "OUT_PROD",
    "SCALE",              "SET",                "CPY",                     "CONT",
    "RESHAPE",            "VIEW",               "PERMUTE",                 "TRANSPOSE",
    "GET_ROWS",           "GET_ROWS_BACK",      "SET_ROWS",                "DIAG",
    "DIAG_MASK_INF",      "DIAG_MASK_ZERO",     "SOFT_MAX",                "SOFT_MAX_BACK",
    "ROPE",               "ROPE_BACK",          "CLAMP",                   "CONV_TRANSPOSE_1D",
    "IM2COL",             "IM2COL_BACK",        "IM2COL_3D",               "COL2IM_1D",
    "CONV_2D",            "CONV_3D",            "CONV_2D_DW",              "CONV_TRANSPOSE_2D",
    "POOL_1D",            "POOL_2D",            "POOL_2D_BACK",            "UPSCALE",
    "PAD",                "PAD_REFLECT_1D",     "ROLL",                    "ARANGE",
    "TIMESTEP_EMBEDDING", "ARGSORT",            "TOP_K",                   "LEAKY_RELU",
    "TRI",                "FILL",               "FLASH_ATTN_EXT",          "FLASH_ATTN_BACK",
    "SSM_CONV",           "SSM_SCAN",           "WIN_PART",                "WIN_UNPART",
    "GET_REL_POS",        "ADD_REL_POS",        "RWKV_WKV6",               "GATED_LINEAR_ATTN",
    "RWKV_WKV7",          "SOLVE_TRI",          "GATED_DELTA_NET",         "LIGHTNING_INDEXER",
    "DSV4_HC_COMB",       "DSV4_HC_PRE",        "DSV4_HC_POST",            "TOP_K_MASK",
    "UNARY",              "MAP_CUSTOM1",        "MAP_CUSTOM2",             "MAP_CUSTOM3",
    "CUSTOM",             "CROSS_ENTROPY_LOSS", "CROSS_ENTROPY_LOSS_BACK", "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",       "GLU",
};

const op_symbol = [_][*:0]const u8{
    "none",                                   "x",                                             "x+y",
    "x[i]+y",                                 "x+y",                                           "view(x,nb,offset)+=y->x",
    "x-y",                                    "x*y",                                           "x/y",
    "x^2",
    "√x",
    "log(x)",                                 "sin(x)",                                        "cos(x)",
    "Σx",
    "Σx_k",
    "cumsum(x)",
    "Σx/n",
    "argmax(x)",                              "count_equal(x)",                                "repeat(x)",
    "repeat_back(x)",                         "concat(x, y)",                                  "silu_back(x)",
    "norm(x)",                                "rms_norm(x)",                                   "rms_norm_back(x)",
    "group_norm(x)",                          "l2_norm(x)",                                    "X*Y",
    "X[i]*Y",                                 "X*Y",                                           "x*v",
    "y-\\>view(x)",                           "x-\\>y",                                        "cont(x)",
    "reshape(x)",                             "view(x)",                                       "permute(x)",
    "transpose(x)",                           "get_rows(x)",                                   "get_rows_back(x)",
    "set_rows(x)",                            "diag(x)",                                       "diag_mask_inf(x)",
    "diag_mask_zero(x)",                      "soft_max(x)",                                   "soft_max_back(x)",
    "rope(x)",                                "rope_back(x)",                                  "clamp(x)",
    "conv_transpose_1d(x)",                   "im2col(x)",                                     "im2col_back(x)",
    "im2col_3d(x)",                           "col2im_1d(x)",                                  "conv_2d(x)",
    "conv_3d(x)",                             "conv_2d_dw(x)",                                 "conv_transpose_2d(x)",
    "pool_1d(x)",                             "pool_2d(x)",                                    "pool_2d_back(x)",
    "upscale(x)",                             "pad(x)",                                        "pad_reflect_1d(x)",
    "roll(x)",                                "arange(start, stop, step)",                     "timestep_embedding(timesteps, dim, max_period)",
    "argsort(x)",                             "top_k(x)",                                      "leaky_relu(x)",
    "tri(x)",                                 "fill(x, c)",                                    "flash_attn_ext(x)",
    "flash_attn_back(x)",                     "ssm_conv(x)",                                   "ssm_scan(x)",
    "win_part(x)",                            "win_unpart(x)",                                 "get_rel_pos(x)",
    "add_rel_pos(x)",                         "rwkv_wkv6(k, v, r, tf, td, s)",                 "gated_linear_attn(k, v, q, gate, s)",
    "rwkv_wkv7(r, w, k, v, a, b, s)",         "A X = B, A triangular, solve X",                "gated_delta_net(q, k, v, g, beta, s)",
    "lightning_indexer(q, k, weights, mask)", "dsv4_hc_comb(mixes, scale, base)",              "dsv4_hc_pre(x, weights)",
    "dsv4_hc_post(x, residual, post, comb)",  "top_k_mask(mask, top_k, score, blocks, extra)", "unary(x)",
    "map_custom(x)",                          "map_custom(x,y)",                               "map_custom(x,y,z)",
    "custom(x)",                              "cross_entropy_loss(x,y)",                       "cross_entropy_loss_back(x,y)",
    "adamw(x)",                               "sgd(x)",                                        "glu(x)",
};

const unary_op_name = [_][*:0]const u8{
    "ABS",         "SGN",        "NEG",   "STEP",
    "TANH",        "ELU",        "RELU",  "SIGMOID",
    "GELU",        "GELU_QUICK", "SILU",  "HARDSWISH",
    "HARDSIGMOID", "EXP",        "EXPM1", "SOFTPLUS",
    "GELU_ERF",    "XIELU",      "FLOOR", "CEIL",
    "ROUND",       "TRUNC",
};

/// Ports `GGML_GLU_OP_NAME` (ggml.c:1267 @cd959bae9).
const glu_op_name = [_][*:0]const u8{
    "REGLU",     "GEGLU",       "SWIGLU",       "SWIGLU_OAI",
    "GEGLU_ERF", "GEGLU_QUICK", "SWIGLU_CLAMP",
};

const type_traits = blk: {
    var t: [c.GGML_TYPE_COUNT]c.ggml_type_traits = @splat(std.mem.zeroes(c.ggml_type_traits));
    t[c.GGML_TYPE_I8] = .{
        .type_name = "i8",
        .blck_size = 1,
        .type_size = @sizeOf(i8),
        .is_quantized = false,
    };
    t[c.GGML_TYPE_I16] = .{
        .type_name = "i16",
        .blck_size = 1,
        .type_size = @sizeOf(i16),
        .is_quantized = false,
    };
    t[c.GGML_TYPE_I32] = .{
        .type_name = "i32",
        .blck_size = 1,
        .type_size = @sizeOf(i32),
        .is_quantized = false,
    };
    t[c.GGML_TYPE_I64] = .{
        .type_name = "i64",
        .blck_size = 1,
        .type_size = @sizeOf(i64),
        .is_quantized = false,
    };
    t[c.GGML_TYPE_F64] = .{
        .type_name = "f64",
        .blck_size = 1,
        .type_size = @sizeOf(f64),
        .is_quantized = false,
    };
    t[c.GGML_TYPE_F32] = .{
        .type_name = "f32",
        .blck_size = 1,
        .type_size = @sizeOf(f32),
        .is_quantized = false,
    };
    t[c.GGML_TYPE_F16] = .{
        .type_name = "f16",
        .blck_size = 1,
        .type_size = @sizeOf(c.ggml_fp16_t),
        .is_quantized = false,
        .to_float = @ptrCast(&c.ggml_fp16_to_fp32_row),
        .from_float_ref = @ptrCast(&c.ggml_fp32_to_fp16_row),
    };
    t[c.GGML_TYPE_Q1_0] = .{
        .type_name = "q1_0",
        .blck_size = c.QK1_0,
        .type_size = @sizeOf(c.block_q1_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q1_0),
        .from_float_ref = @ptrCast(&c.quantize_row_q1_0_ref),
    };
    t[c.GGML_TYPE_Q2_0] = .{
        .type_name = "q2_0",
        .blck_size = c.QK2_0,
        .type_size = @sizeOf(c.block_q2_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q2_0),
        .from_float_ref = @ptrCast(&c.quantize_row_q2_0_ref),
    };
    t[c.GGML_TYPE_Q4_0] = .{
        .type_name = "q4_0",
        .blck_size = c.QK4_0,
        .type_size = @sizeOf(c.block_q4_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q4_0),
        .from_float_ref = @ptrCast(&c.quantize_row_q4_0_ref),
    };
    t[c.GGML_TYPE_Q4_1] = .{
        .type_name = "q4_1",
        .blck_size = c.QK4_1,
        .type_size = @sizeOf(c.block_q4_1),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q4_1),
        .from_float_ref = @ptrCast(&c.quantize_row_q4_1_ref),
    };
    t[4] = .{
        .type_name = "DEPRECATED",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[5] = .{
        .type_name = "DEPRECATED",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[c.GGML_TYPE_Q5_0] = .{
        .type_name = "q5_0",
        .blck_size = c.QK5_0,
        .type_size = @sizeOf(c.block_q5_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q5_0),
        .from_float_ref = @ptrCast(&c.quantize_row_q5_0_ref),
    };
    t[c.GGML_TYPE_Q5_1] = .{
        .type_name = "q5_1",
        .blck_size = c.QK5_1,
        .type_size = @sizeOf(c.block_q5_1),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q5_1),
        .from_float_ref = @ptrCast(&c.quantize_row_q5_1_ref),
    };
    t[c.GGML_TYPE_Q8_0] = .{
        .type_name = "q8_0",
        .blck_size = c.QK8_0,
        .type_size = @sizeOf(c.block_q8_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q8_0),
        .from_float_ref = @ptrCast(&c.quantize_row_q8_0_ref),
    };
    t[c.GGML_TYPE_Q8_1] = .{
        .type_name = "q8_1",
        .blck_size = c.QK8_1,
        .type_size = @sizeOf(c.block_q8_1),
        .is_quantized = true,
        .from_float_ref = @ptrCast(&c.quantize_row_q8_1_ref),
    };
    t[c.GGML_TYPE_MXFP4] = .{
        .type_name = "mxfp4",
        .blck_size = c.QK_MXFP4,
        .type_size = @sizeOf(c.block_mxfp4),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_mxfp4),
        .from_float_ref = @ptrCast(&c.quantize_row_mxfp4_ref),
    };
    t[c.GGML_TYPE_NVFP4] = .{
        .type_name = "nvfp4",
        .blck_size = c.QK_NVFP4,
        .type_size = @sizeOf(c.block_nvfp4),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_nvfp4),
        .from_float_ref = @ptrCast(&c.quantize_row_nvfp4_ref),
    };
    t[c.GGML_TYPE_F8_E4M3] = .{
        .type_name = "f8_e4m3",
        .blck_size = 1,
        .type_size = @sizeOf(c.ggml_fp8_e4m3_t),
        .is_quantized = false,
        .to_float = @ptrCast(&c.dequantize_row_f8_e4m3),
        .from_float_ref = @ptrCast(&c.quantize_row_f8_e4m3_ref),
    };
    t[c.GGML_TYPE_Q2_K] = .{
        .type_name = "q2_K",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_q2_K),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q2_K),
        .from_float_ref = @ptrCast(&c.quantize_row_q2_K_ref),
    };
    t[c.GGML_TYPE_Q3_K] = .{
        .type_name = "q3_K",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_q3_K),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q3_K),
        .from_float_ref = @ptrCast(&c.quantize_row_q3_K_ref),
    };
    t[c.GGML_TYPE_Q4_K] = .{
        .type_name = "q4_K",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_q4_K),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q4_K),
        .from_float_ref = @ptrCast(&c.quantize_row_q4_K_ref),
    };
    t[c.GGML_TYPE_Q5_K] = .{
        .type_name = "q5_K",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_q5_K),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q5_K),
        .from_float_ref = @ptrCast(&c.quantize_row_q5_K_ref),
    };
    t[c.GGML_TYPE_Q6_K] = .{
        .type_name = "q6_K",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_q6_K),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_q6_K),
        .from_float_ref = @ptrCast(&c.quantize_row_q6_K_ref),
    };
    t[c.GGML_TYPE_IQ2_XXS] = .{
        .type_name = "iq2_xxs",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq2_xxs),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq2_xxs),
    };
    t[c.GGML_TYPE_IQ2_XS] = .{
        .type_name = "iq2_xs",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq2_xs),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq2_xs),
    };
    t[c.GGML_TYPE_IQ3_XXS] = .{
        .type_name = "iq3_xxs",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq3_xxs),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq3_xxs),
        .from_float_ref = @ptrCast(&c.quantize_row_iq3_xxs_ref),
    };
    t[c.GGML_TYPE_IQ3_S] = .{
        .type_name = "iq3_s",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq3_s),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq3_s),
        .from_float_ref = @ptrCast(&c.quantize_row_iq3_s_ref),
    };
    t[c.GGML_TYPE_IQ2_S] = .{
        .type_name = "iq2_s",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq2_s),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq2_s),
        .from_float_ref = @ptrCast(&c.quantize_row_iq2_s_ref),
    };
    t[c.GGML_TYPE_IQ1_S] = .{
        .type_name = "iq1_s",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq1_s),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq1_s),
    };
    t[c.GGML_TYPE_IQ1_M] = .{
        .type_name = "iq1_m",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq1_m),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq1_m),
    };
    t[c.GGML_TYPE_IQ4_NL] = .{
        .type_name = "iq4_nl",
        .blck_size = c.QK4_NL,
        .type_size = @sizeOf(c.block_iq4_nl),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq4_nl),
        .from_float_ref = @ptrCast(&c.quantize_row_iq4_nl_ref),
    };
    t[c.GGML_TYPE_IQ4_XS] = .{
        .type_name = "iq4_xs",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_iq4_xs),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_iq4_xs),
        .from_float_ref = @ptrCast(&c.quantize_row_iq4_xs_ref),
    };
    t[c.GGML_TYPE_Q8_K] = .{
        .type_name = "q8_K",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_q8_K),
        .is_quantized = true,
    };
    t[c.GGML_TYPE_BF16] = .{
        .type_name = "bf16",
        .blck_size = 1,
        .type_size = @sizeOf(c.ggml_bf16_t),
        .is_quantized = false,
        .to_float = @ptrCast(&c.ggml_bf16_to_fp32_row),
        .from_float_ref = @ptrCast(&c.ggml_fp32_to_bf16_row_ref),
    };
    t[31] = .{
        .type_name = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[32] = .{
        .type_name = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[33] = .{
        .type_name = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[c.GGML_TYPE_TQ1_0] = .{
        .type_name = "tq1_0",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_tq1_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_tq1_0),
        .from_float_ref = @ptrCast(&c.quantize_row_tq1_0_ref),
    };
    t[c.GGML_TYPE_TQ2_0] = .{
        .type_name = "tq2_0",
        .blck_size = c.QK_K,
        .type_size = @sizeOf(c.block_tq2_0),
        .is_quantized = true,
        .to_float = @ptrCast(&c.dequantize_row_tq2_0),
        .from_float_ref = @ptrCast(&c.quantize_row_tq2_0_ref),
    };
    t[36] = .{
        .type_name = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[37] = .{
        .type_name = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    t[38] = .{
        .type_name = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_size = 0,
        .type_size = 0,
        .is_quantized = false,
    };
    break :blk t;
};

// -----------------------------------------------------------------------------
// Table accessors

comptime {
    @export(&ggmlGetTypeTraits, .{ .name = "ggml_get_type_traits" });
}

/// Ports `ggml_get_type_traits` (ggml.c:963 @cd959bae9).
pub fn ggmlGetTypeTraits(t: c.enum_ggml_type) callconv(.c) *const c.ggml_type_traits {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    return &type_traits[@intCast(t)];
}

comptime {
    @export(&ggmlBlckSize, .{ .name = "ggml_blck_size" });
}

/// Ports `ggml_blck_size` (ggml.c:1344 @cd959bae9).
pub fn ggmlBlckSize(t: c.enum_ggml_type) callconv(.c) i64 {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    return type_traits[@intCast(t)].blck_size;
}

comptime {
    @export(&ggmlTypeSize, .{ .name = "ggml_type_size" });
}

/// Ports `ggml_type_size` (ggml.c:1350 @cd959bae9).
pub fn ggmlTypeSize(t: c.enum_ggml_type) callconv(.c) usize {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    return type_traits[@intCast(t)].type_size;
}

comptime {
    @export(&ggmlRowSize, .{ .name = "ggml_row_size" });
}

/// Ports `ggml_row_size` (ggml.c:1356 @cd959bae9).
///
/// A row's size is not `ne * type_size`: for a quantized type, `type_size` is
/// the size of a whole block, so the element count divides by the block size.
pub fn ggmlRowSize(t: c.enum_ggml_type, ne: i64) callconv(.c) usize {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    std.debug.assert(@rem(ne, ggmlBlckSize(t)) == 0);
    return ggmlTypeSize(t) * @as(usize, @intCast(ne)) / @as(usize, @intCast(ggmlBlckSize(t)));
}

comptime {
    @export(&ggmlTypeSizef, .{ .name = "ggml_type_sizef" });
}

/// Ports `ggml_type_sizef` (ggml.c:1363 @cd959bae9).
///
/// Bytes per element as a fraction, which is how quantized formats are usually
/// quoted (a 4-bit type reports 0.5 and change).
fn ggmlTypeSizef(t: c.enum_ggml_type) callconv(.c) f64 {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    const traits = type_traits[@intCast(t)];
    return @as(f64, @floatFromInt(traits.type_size)) / @as(f64, @floatFromInt(traits.blck_size));
}

comptime {
    @export(&ggmlTypeName, .{ .name = "ggml_type_name" });
}

/// Ports `ggml_type_name` (ggml.c:1369 @cd959bae9).
pub fn ggmlTypeName(t: c.enum_ggml_type) callconv(.c) [*:0]const u8 {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    return @ptrCast(type_traits[@intCast(t)].type_name);
}

comptime {
    @export(&ggmlIsQuantized, .{ .name = "ggml_is_quantized" });
}

/// Ports `ggml_is_quantized` (ggml.c:1375 @cd959bae9).
pub fn ggmlIsQuantized(t: c.enum_ggml_type) callconv(.c) bool {
    std.debug.assert(t >= 0 and t < c.GGML_TYPE_COUNT);
    return type_traits[@intCast(t)].is_quantized;
}

comptime {
    @export(&ggmlOpName, .{ .name = "ggml_op_name" });
}

/// Ports `ggml_op_name` (ggml.c:1381 @cd959bae9).
pub fn ggmlOpName(op: c.enum_ggml_op) callconv(.c) [*:0]const u8 {
    return op_name[@intCast(op)];
}

comptime {
    @export(&ggmlOpSymbol, .{ .name = "ggml_op_symbol" });
}

/// Ports `ggml_op_symbol` (ggml.c:1385 @cd959bae9).
pub fn ggmlOpSymbol(op: c.enum_ggml_op) callconv(.c) [*:0]const u8 {
    return op_symbol[@intCast(op)];
}

comptime {
    @export(&ggmlUnaryOpName, .{ .name = "ggml_unary_op_name" });
}

/// Ports `ggml_unary_op_name` (ggml.c:1389 @cd959bae9).
pub fn ggmlUnaryOpName(op: c.enum_ggml_unary_op) callconv(.c) [*:0]const u8 {
    return unary_op_name[@intCast(op)];
}

comptime {
    @export(&ggmlGluOpName, .{ .name = "ggml_glu_op_name" });
}

/// Ports `ggml_glu_op_name` (ggml.c:1393 @cd959bae9).
pub fn ggmlGluOpName(op: c.enum_ggml_glu_op) callconv(.c) [*:0]const u8 {
    return glu_op_name[@intCast(op)];
}

comptime {
    @export(&ggmlOpDesc, .{ .name = "ggml_op_desc" });
}

/// Ports `ggml_op_desc` (ggml.c:1397 @cd959bae9).
///
/// Unary and GLU ops all share one `ggml_op`, so a useful description has to
/// reach into `op_params` for the real operation.
pub fn ggmlOpDesc(t: *const c.ggml_tensor) callconv(.c) [*:0]const u8 {
    if (t.op == c.GGML_OP_UNARY) return ggmlUnaryOpName(ggmlGetUnaryOp(t));
    if (t.op == c.GGML_OP_GLU) return ggmlGluOpName(ggmlGetGluOp(t));
    return ggmlOpName(t.op);
}

comptime {
    @export(&ggmlGetUnaryOp, .{ .name = "ggml_get_unary_op" });
}

/// Ports `ggml_get_unary_op` (ggml.c:1969 @cd959bae9).
pub fn ggmlGetUnaryOp(t: *const c.ggml_tensor) callconv(.c) c.enum_ggml_unary_op {
    impl.assert(t.op == c.GGML_OP_UNARY, "tensor->op == GGML_OP_UNARY");
    return @intCast(t.op_params[0]);
}

comptime {
    @export(&ggmlGetGluOp, .{ .name = "ggml_get_glu_op" });
}

/// Ports `ggml_get_glu_op` (ggml.c:1974 @cd959bae9).
pub fn ggmlGetGluOp(t: *const c.ggml_tensor) callconv(.c) c.enum_ggml_glu_op {
    impl.assert(t.op == c.GGML_OP_GLU, "tensor->op == GGML_OP_GLU");
    return @intCast(t.op_params[0]);
}

comptime {
    @export(&ggmlFtypeToGgmlType, .{ .name = "ggml_ftype_to_ggml_type" });
}

/// Ports `ggml_ftype_to_ggml_type` (ggml.c:1444 @cd959bae9).
///
/// A file type names the format most tensors in a model use; this maps it to
/// the tensor type that implies. Aborts on an unknown or mixed file type,
/// which the C does too.
fn ggmlFtypeToGgmlType(ftype: c.enum_ggml_ftype) callconv(.c) c.enum_ggml_type {
    const wtype: c.enum_ggml_type = switch (ftype) {
        c.GGML_FTYPE_ALL_F32 => c.GGML_TYPE_F32,
        c.GGML_FTYPE_MOSTLY_F16 => c.GGML_TYPE_F16,
        c.GGML_FTYPE_MOSTLY_BF16 => c.GGML_TYPE_BF16,
        c.GGML_FTYPE_MOSTLY_Q4_0 => c.GGML_TYPE_Q4_0,
        c.GGML_FTYPE_MOSTLY_Q4_1 => c.GGML_TYPE_Q4_1,
        c.GGML_FTYPE_MOSTLY_Q1_0 => c.GGML_TYPE_Q1_0,
        c.GGML_FTYPE_MOSTLY_Q2_0 => c.GGML_TYPE_Q2_0,
        c.GGML_FTYPE_MOSTLY_Q5_0 => c.GGML_TYPE_Q5_0,
        c.GGML_FTYPE_MOSTLY_Q5_1 => c.GGML_TYPE_Q5_1,
        c.GGML_FTYPE_MOSTLY_Q8_0 => c.GGML_TYPE_Q8_0,
        c.GGML_FTYPE_MOSTLY_MXFP4 => c.GGML_TYPE_MXFP4,
        c.GGML_FTYPE_MOSTLY_NVFP4 => c.GGML_TYPE_NVFP4,
        c.GGML_FTYPE_MOSTLY_F8_E4M3 => c.GGML_TYPE_F8_E4M3,
        c.GGML_FTYPE_MOSTLY_Q2_K => c.GGML_TYPE_Q2_K,
        c.GGML_FTYPE_MOSTLY_Q3_K => c.GGML_TYPE_Q3_K,
        c.GGML_FTYPE_MOSTLY_Q4_K => c.GGML_TYPE_Q4_K,
        c.GGML_FTYPE_MOSTLY_Q5_K => c.GGML_TYPE_Q5_K,
        c.GGML_FTYPE_MOSTLY_Q6_K => c.GGML_TYPE_Q6_K,
        c.GGML_FTYPE_MOSTLY_IQ2_XXS => c.GGML_TYPE_IQ2_XXS,
        c.GGML_FTYPE_MOSTLY_IQ2_XS => c.GGML_TYPE_IQ2_XS,
        c.GGML_FTYPE_MOSTLY_IQ3_XXS => c.GGML_TYPE_IQ3_XXS,
        c.GGML_FTYPE_MOSTLY_IQ1_S => c.GGML_TYPE_IQ1_S,
        c.GGML_FTYPE_MOSTLY_IQ1_M => c.GGML_TYPE_IQ1_M,
        c.GGML_FTYPE_MOSTLY_IQ4_NL => c.GGML_TYPE_IQ4_NL,
        c.GGML_FTYPE_MOSTLY_IQ4_XS => c.GGML_TYPE_IQ4_XS,
        c.GGML_FTYPE_MOSTLY_IQ3_S => c.GGML_TYPE_IQ3_S,
        c.GGML_FTYPE_MOSTLY_IQ2_S => c.GGML_TYPE_IQ2_S,
        // Unknown, and the mixed Q4_1/F16 format, have no single tensor type.
        else => c.GGML_TYPE_COUNT,
    };

    impl.assert(wtype != c.GGML_TYPE_COUNT, "wtype != GGML_TYPE_COUNT");
    return wtype;
}

// -----------------------------------------------------------------------------
// Element counts and sizes

comptime {
    @export(&ggmlNelements, .{ .name = "ggml_nelements" });
}

/// Ports `ggml_nelements` (ggml.c:1303 @cd959bae9).
pub fn ggmlNelements(tensor: *const c.ggml_tensor) callconv(.c) i64 {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return tensor.ne[0] * tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
}

comptime {
    @export(&ggmlNrows, .{ .name = "ggml_nrows" });
}

/// Ports `ggml_nrows` (ggml.c:1309 @cd959bae9).
pub fn ggmlNrows(tensor: *const c.ggml_tensor) callconv(.c) i64 {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return tensor.ne[1] * tensor.ne[2] * tensor.ne[3];
}

comptime {
    @export(&ggmlNbytes, .{ .name = "ggml_nbytes" });
}

/// Ports `ggml_nbytes` (ggml.c:1315 @cd959bae9).
///
/// Computed from strides rather than from the element count, so a view of a
/// larger tensor reports the span it actually occupies.
pub fn ggmlNbytes(tensor: *const c.ggml_tensor) callconv(.c) usize {
    for (0..c.GGML_MAX_DIMS) |i| {
        if (tensor.ne[i] <= 0) return 0;
    }

    const blck_size: usize = @intCast(ggmlBlckSize(tensor.type));
    var nbytes: usize = undefined;
    if (blck_size == 1) {
        nbytes = ggmlTypeSize(tensor.type);
        for (0..c.GGML_MAX_DIMS) |i| {
            nbytes += @as(usize, @intCast(tensor.ne[i] - 1)) * tensor.nb[i];
        }
    } else {
        nbytes = @as(usize, @intCast(tensor.ne[0])) * tensor.nb[0] / blck_size;
        for (1..c.GGML_MAX_DIMS) |i| {
            nbytes += @as(usize, @intCast(tensor.ne[i] - 1)) * tensor.nb[i];
        }
    }
    return nbytes;
}

comptime {
    @export(&ggmlNbytesPad, .{ .name = "ggml_nbytes_pad" });
}

/// Ports `ggml_nbytes_pad` (ggml.c:1340 @cd959bae9).
pub fn ggmlNbytesPad(tensor: *const c.ggml_tensor) callconv(.c) usize {
    return impl.pad(ggmlNbytes(tensor), c.GGML_MEM_ALIGN);
}

comptime {
    @export(&ggmlElementSize, .{ .name = "ggml_element_size" });
}

/// Ports `ggml_element_size` (ggml.c:1409 @cd959bae9).
pub fn ggmlElementSize(tensor: *const c.ggml_tensor) callconv(.c) usize {
    return ggmlTypeSize(tensor.type);
}

comptime {
    @export(&ggmlTensorOverhead, .{ .name = "ggml_tensor_overhead" });
}

/// Ports `ggml_tensor_overhead` (ggml.c:1484 @cd959bae9).
///
/// What a tensor costs a context beyond its data: the object header plus the
/// tensor struct.
pub fn ggmlTensorOverhead() callconv(.c) usize {
    return object_size + c.GGML_TENSOR_SIZE;
}

/// Ports `GGML_OBJECT_SIZE` (ggml.c:984 @cd959bae9).
pub const object_size: usize = @sizeOf(impl.Object);

// -----------------------------------------------------------------------------
// Shape predicates

comptime {
    @export(&ggmlIsScalar, .{ .name = "ggml_is_scalar" });
}

/// Ports `ggml_is_scalar` (ggml.c:1413 @cd959bae9).
pub fn ggmlIsScalar(tensor: *const c.ggml_tensor) callconv(.c) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return tensor.ne[0] == 1 and tensor.ne[1] == 1 and tensor.ne[2] == 1 and tensor.ne[3] == 1;
}

comptime {
    @export(&ggmlIsVector, .{ .name = "ggml_is_vector" });
}

/// Ports `ggml_is_vector` (ggml.c:1419 @cd959bae9).
pub fn ggmlIsVector(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return tensor.ne[1] == 1 and tensor.ne[2] == 1 and tensor.ne[3] == 1;
}

comptime {
    @export(&ggmlIsMatrix, .{ .name = "ggml_is_matrix" });
}

/// Ports `ggml_is_matrix` (ggml.c:1425 @cd959bae9).
pub fn ggmlIsMatrix(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return tensor.ne[2] == 1 and tensor.ne[3] == 1;
}

comptime {
    @export(&ggmlIs3d, .{ .name = "ggml_is_3d" });
}

/// Ports `ggml_is_3d` (ggml.c:1431 @cd959bae9).
pub fn ggmlIs3d(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return tensor.ne[3] == 1;
}

comptime {
    @export(&ggmlNDims, .{ .name = "ggml_n_dims" });
}

/// Ports `ggml_n_dims` (ggml.c:1435 @cd959bae9).
///
/// - Returns: the index of the highest non-unit dimension, plus one. Always at
/// least 1, so a scalar reports one dimension rather than none.
pub fn ggmlNDims(tensor: *const c.ggml_tensor) callconv(.c) c_int {
    var i: usize = c.GGML_MAX_DIMS - 1;
    while (i >= 1) : (i -= 1) {
        if (tensor.ne[i] > 1) return @intCast(i + 1);
    }
    return 1;
}

comptime {
    @export(&ggmlIsEmpty, .{ .name = "ggml_is_empty" });
}

/// Ports `ggml_is_empty` (ggml.c:1572 @cd959bae9).
pub fn ggmlIsEmpty(tensor: *const c.ggml_tensor) callconv(.c) bool {
    for (0..c.GGML_MAX_DIMS) |i| {
        if (tensor.ne[i] == 0) return true;
    }
    return false;
}

comptime {
    @export(&ggmlAreSameShape, .{ .name = "ggml_are_same_shape" });
}

/// Ports `ggml_are_same_shape` (ggml.c:1582 @cd959bae9).
pub fn ggmlAreSameShape(t0: *const c.ggml_tensor, t1: *const c.ggml_tensor) callconv(.c) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return t0.ne[0] == t1.ne[0] and t0.ne[1] == t1.ne[1] and
        t0.ne[2] == t1.ne[2] and t0.ne[3] == t1.ne[3];
}

comptime {
    @export(&ggmlAreSameStride, .{ .name = "ggml_are_same_stride" });
}

/// Ports `ggml_are_same_stride` (ggml.c:1592 @cd959bae9).
pub fn ggmlAreSameStride(t0: *const c.ggml_tensor, t1: *const c.ggml_tensor) callconv(.c) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return t0.nb[0] == t1.nb[0] and t0.nb[1] == t1.nb[1] and
        t0.nb[2] == t1.nb[2] and t0.nb[3] == t1.nb[3];
}

comptime {
    @export(&ggmlIsView, .{ .name = "ggml_is_view" });
}

/// Ports `ggml_is_view` (ggml.c:1602 @cd959bae9).
pub fn ggmlIsView(t: *const c.ggml_tensor) callconv(.c) bool {
    return impl.isView(t);
}

comptime {
    @export(&ggmlCanRepeat, .{ .name = "ggml_can_repeat" });
}

/// Ports `ggml_can_repeat` (ggml.c:1607 @cd959bae9).
///
/// - Returns: whether `t1`'s shape is a whole-number tiling of `t0`'s, which is
/// what broadcasting requires. Two empty tensors count as compatible.
pub fn ggmlCanRepeat(t0: *const c.ggml_tensor, t1: *const c.ggml_tensor) callconv(.c) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    if (ggmlIsEmpty(t0)) return ggmlIsEmpty(t1);
    return @rem(t1.ne[0], t0.ne[0]) == 0 and
        @rem(t1.ne[1], t0.ne[1]) == 0 and
        @rem(t1.ne[2], t0.ne[2]) == 0 and
        @rem(t1.ne[3], t0.ne[3]) == 0;
}

/// Ports `ggml_can_repeat_rows` (ggml.c:1617 @cd959bae9).
pub fn canRepeatRows(t0: *const c.ggml_tensor, t1: *const c.ggml_tensor) bool {
    return t0.ne[0] == t1.ne[0] and ggmlCanRepeat(t0, t1);
}

// -----------------------------------------------------------------------------
// Layout predicates

comptime {
    @export(&ggmlIsTransposed, .{ .name = "ggml_is_transposed" });
}

/// Ports `ggml_is_transposed` (ggml.c:1488 @cd959bae9).
pub fn ggmlIsTransposed(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return tensor.nb[0] > tensor.nb[1];
}

/// Ports `ggml_is_contiguous_m_n` (ggml.c:1492 @cd959bae9).
///
/// Walks the strides checking each equals the running product of the lower
/// dimensions. Dimensions at or below `m` are exempt, which is how the
/// `_1`/`_2` variants allow a gap at a chosen level.
///
/// - `tensor`: the tensor to inspect.
/// - `m`: highest dimension allowed to be non-contiguous.
/// - `n`: number of dimensions to check.
///
/// - Returns: whether the layout is contiguous under those relaxations.
fn isContiguousMN(tensor: *const c.ggml_tensor, m: usize, n: usize) bool {
    const blck: usize = @intCast(ggmlBlckSize(tensor.type));
    var next_nb = ggmlTypeSize(tensor.type);
    if (tensor.ne[0] != ggmlBlckSize(tensor.type) and tensor.nb[0] != next_nb) {
        return false;
    }
    next_nb *= @as(usize, @intCast(tensor.ne[0])) / blck;
    for (1..n) |i| {
        if (i > m) {
            if (tensor.ne[i] != 1 and tensor.nb[i] != next_nb) return false;
            next_nb *= @intCast(tensor.ne[i]);
        } else {
            // This dimension is allowed a gap, so the running product restarts
            // from whatever stride it actually has.
            next_nb = @as(usize, @intCast(tensor.ne[i])) * tensor.nb[i];
        }
    }
    return true;
}

comptime {
    @export(&ggmlIsContiguous, .{ .name = "ggml_is_contiguous" });
}

/// Ports `ggml_is_contiguous` (ggml.c:1512 @cd959bae9).
pub fn ggmlIsContiguous(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return ggmlIsContiguous0(tensor);
}

comptime {
    @export(&ggmlIsContiguous0, .{ .name = "ggml_is_contiguous_0" });
}

/// Ports `ggml_is_contiguous_0` (ggml.c:1516 @cd959bae9).
fn ggmlIsContiguous0(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return isContiguousMN(tensor, 0, c.GGML_MAX_DIMS);
}

comptime {
    @export(&ggmlIsContiguous1, .{ .name = "ggml_is_contiguous_1" });
}

/// Ports `ggml_is_contiguous_1` (ggml.c:1520 @cd959bae9).
pub fn ggmlIsContiguous1(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return isContiguousMN(tensor, 1, c.GGML_MAX_DIMS);
}

comptime {
    @export(&ggmlIsContiguous2, .{ .name = "ggml_is_contiguous_2" });
}

/// Ports `ggml_is_contiguous_2` (ggml.c:1524 @cd959bae9).
pub fn ggmlIsContiguous2(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return isContiguousMN(tensor, 2, c.GGML_MAX_DIMS);
}

comptime {
    @export(&ggmlIsContiguousTo1, .{ .name = "ggml_is_contiguous_to_1" });
}

/// Ports `ggml_is_contiguous_to_1` (ggml.c:1528 @cd959bae9).
fn ggmlIsContiguousTo1(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return isContiguousMN(tensor, 0, 1);
}

comptime {
    @export(&ggmlIsContiguousTo2, .{ .name = "ggml_is_contiguous_to_2" });
}

/// Ports `ggml_is_contiguous_to_2` (ggml.c:1532 @cd959bae9).
fn ggmlIsContiguousTo2(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return isContiguousMN(tensor, 0, 2);
}

comptime {
    @export(&ggmlIsContiguousTo3, .{ .name = "ggml_is_contiguous_to_3" });
}

/// Ports `ggml_is_contiguous_to_3` (ggml.c:1536 @cd959bae9).
fn ggmlIsContiguousTo3(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return isContiguousMN(tensor, 0, 3);
}

comptime {
    @export(&ggmlIsContiguouslyAllocated, .{ .name = "ggml_is_contiguously_allocated" });
}

/// Ports `ggml_is_contiguously_allocated` (ggml.c:1540 @cd959bae9).
///
/// Stronger than `ggml_is_contiguous`: the span the tensor occupies must equal
/// what its elements need, so there is no trailing slack either.
pub fn ggmlIsContiguouslyAllocated(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return ggmlNbytes(tensor) == @as(usize, @intCast(ggmlNelements(tensor))) *
        ggmlTypeSize(tensor.type) / @as(usize, @intCast(ggmlBlckSize(tensor.type)));
}

comptime {
    @export(&ggmlIsPermuted, .{ .name = "ggml_is_permuted" });
}

/// Ports `ggml_is_permuted` (ggml.c:1544 @cd959bae9).
pub fn ggmlIsPermuted(tensor: *const c.ggml_tensor) callconv(.c) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return tensor.nb[0] > tensor.nb[1] or tensor.nb[1] > tensor.nb[2] or tensor.nb[2] > tensor.nb[3];
}

comptime {
    @export(&ggmlIsContiguousChannels, .{ .name = "ggml_is_contiguous_channels" });
}

/// Ports `ggml_is_contiguous_channels` (ggml.c:1550 @cd959bae9).
pub fn ggmlIsContiguousChannels(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return tensor.nb[0] > tensor.nb[2] and
        tensor.nb[1] > tensor.nb[0] and
        tensor.nb[2] == ggmlTypeSize(tensor.type);
}

comptime {
    @export(&ggmlIsContiguousRows, .{ .name = "ggml_is_contiguous_rows" });
}

/// Ports `ggml_is_contiguous_rows` (ggml.c:1557 @cd959bae9).
pub fn ggmlIsContiguousRows(tensor: *const c.ggml_tensor) callconv(.c) bool {
    return tensor.ne[0] == ggmlBlckSize(tensor.type) or
        tensor.nb[0] == ggmlTypeSize(tensor.type);
}

/// Ports `ggml_is_padded_1d` (ggml.c:1563 @cd959bae9).
pub fn isPadded1d(tensor: *const c.ggml_tensor) bool {
    comptime std.debug.assert(c.GGML_MAX_DIMS == 4);
    return tensor.nb[0] == ggmlTypeSize(tensor.type) and
        tensor.nb[2] == tensor.nb[1] * @as(usize, @intCast(tensor.ne[1])) and
        tensor.nb[3] == tensor.nb[2] * @as(usize, @intCast(tensor.ne[2]));
}

// -----------------------------------------------------------------------------
// Unit Tests

// Golden checksums of the C tables, captured from a C-only build before any
// of ggml.c was ported.
//
// These cannot be direct comparisons against `c.ggml_get_type_traits` or
// `c.ggml_op_name`: this file exports those very symbols, so such a test would
// resolve back to the code under test and pass no matter how wrong the table
// became. That is not hypothetical -- it happened, and corrupting a block size
// left the comparison green.
//
// Regenerate only from a C build: `scripts/types-golden` prints the traits and
// four name checksums from the stock ggml `llama.cpp.zmake` builds (v0.4.0 added
// SWIGLU_CLAMP, which moved `golden_glu_name` alone).
const golden_traits: u64 = 0xea2f8c7b0e2dba72;
const golden_op_name: u64 = 0x7849ce3c514237af;
const golden_op_symbol: u64 = 0xbf7bcbc1d68353c3;
const golden_unary_name: u64 = 0x78d971e2f43bd703;
const golden_glu_name: u64 = 0xbcc1426328c349eb;

fn mix(h: u64, v: u64) u64 {
    return h ^ (v +% 0x9e3779b97f4a7c15 +% (h << 6) +% (h >> 2));
}

/// Folds a C string into the checksum, distinguishing absent from empty.
fn mixStr(h_in: u64, s: ?[*:0]const u8) u64 {
    const str = s orelse return mix(h_in, 0xdeadbeef);
    var h = h_in;
    var i: usize = 0;
    while (str[i] != 0) : (i += 1) h = mix(h, str[i]);
    return mix(h, 0);
}

test "traits table matches the C checksum" {
    var h: u64 = 0;
    var t: usize = 0;
    while (t < c.GGML_TYPE_COUNT) : (t += 1) {
        const tr = &type_traits[t];
        h = mixStr(h, @ptrCast(tr.type_name));
        h = mix(h, @bitCast(tr.blck_size));
        h = mix(h, tr.type_size);
        h = mix(h, if (tr.is_quantized) 1 else 0);
        h = mix(h, if (tr.to_float != null) 1 else 0);
        h = mix(h, if (tr.from_float_ref != null) 1 else 0);
    }
    try std.testing.expectEqual(golden_traits, h);
}

test "op name tables match the C checksums" {
    var hn: u64 = 0;
    var hs: u64 = 0;
    for (op_name, op_symbol) |n, sym| {
        hn = mixStr(hn, n);
        hs = mixStr(hs, sym);
    }
    try std.testing.expectEqual(golden_op_name, hn);
    try std.testing.expectEqual(golden_op_symbol, hs);

    var hu: u64 = 0;
    for (unary_op_name) |n| hu = mixStr(hu, n);
    try std.testing.expectEqual(golden_unary_name, hu);

    var hg: u64 = 0;
    for (glu_op_name) |n| hg = mixStr(hg, n);
    try std.testing.expectEqual(golden_glu_name, hg);

    // The tables must also be exactly as long as the enums they index.
    try std.testing.expectEqual(@as(usize, c.GGML_OP_COUNT), op_name.len);
    try std.testing.expectEqual(@as(usize, c.GGML_OP_COUNT), op_symbol.len);
    try std.testing.expectEqual(@as(usize, c.GGML_UNARY_OP_COUNT), unary_op_name.len);
    try std.testing.expectEqual(@as(usize, c.GGML_GLU_OP_COUNT), glu_op_name.len);
}

test "traits conversion pointers name the right routines" {
    // The checksum above only records whether a pointer is set. This checks
    // identity against the real symbols in ggml-quants.c, which this file does
    // not export, so the comparison is genuine.
    const cases = .{
        .{ c.GGML_TYPE_Q4_K, &c.dequantize_row_q4_K, &c.quantize_row_q4_K_ref },
        .{ c.GGML_TYPE_Q5_K, &c.dequantize_row_q5_K, &c.quantize_row_q5_K_ref },
        .{ c.GGML_TYPE_Q6_K, &c.dequantize_row_q6_K, &c.quantize_row_q6_K_ref },
        .{ c.GGML_TYPE_Q8_0, &c.dequantize_row_q8_0, &c.quantize_row_q8_0_ref },
        .{ c.GGML_TYPE_IQ4_XS, &c.dequantize_row_iq4_xs, &c.quantize_row_iq4_xs_ref },
    };
    inline for (cases) |case| {
        const tr = &type_traits[@intCast(case[0])];
        try std.testing.expectEqual(@intFromPtr(case[1]), @intFromPtr(tr.to_float));
        try std.testing.expectEqual(@intFromPtr(case[2]), @intFromPtr(tr.from_float_ref));
    }
    // Types that genuinely have no reference quantizer must stay null.
    try std.testing.expect(type_traits[@intCast(c.GGML_TYPE_IQ2_XXS)].from_float_ref == null);
    try std.testing.expect(type_traits[@intCast(c.GGML_TYPE_Q8_K)].to_float == null);
}

test "row size accounts for block quantization" {
    // f32 is one element per block, so a row is the plain product.
    try std.testing.expectEqual(@as(usize, 256 * 4), ggmlRowSize(c.GGML_TYPE_F32, 256));
    // Q4_K packs 256 elements into one 144-byte block.
    try std.testing.expectEqual(@as(usize, 144), ggmlRowSize(c.GGML_TYPE_Q4_K, 256));
    try std.testing.expectEqual(@as(usize, 288), ggmlRowSize(c.GGML_TYPE_Q4_K, 512));
}
