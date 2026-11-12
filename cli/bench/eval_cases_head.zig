//! Port of `make_test_cases_eval`, lines 10131-11500 (tests/test-backend-ops.cpp @cd959bae9).

const std = @import("std");
const mod = @import("module.zig");

/// The head half of the eval case list.
pub const EvalCasesHead = struct {
    const Self = @This();

    /// Adds this half's cases to `list`, in upstream's order.
    ///
    /// - `list`: the destination.
    pub fn build(list: *mod.CaseList) !void {
        // upstream's `rng` is only used by the tail half.

        // unary ops
        // TODO: add GGML_UNARY_OP_SOFTPLUS test that is sensitive to using `log1p(exp(x))` vs `log(exp(x) + 1)`
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_BF16 }) |t| {
            for ([_]i32{ 0, 1 }) |v| {
                var op: mod.c.enum_ggml_unary_op = 0;
                while (op < mod.c.GGML_UNARY_OP_COUNT) : (op += 1) {
                    if (op == mod.c.GGML_UNARY_OP_XIELU) continue; // need extra params, separate test
                    try list.add(mod.cases.TestUnary, .{ op, t, .{ 128, 2, 2, 2 }, v });
                    try list.add(mod.cases.TestUnary, .{ op, t, .{ 5, 7, 11, 13 }, v });
                }
            }
        }

        // fused relu + sqr (squared ReLU)
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |t| {
            try list.add(mod.cases.TestReluSqr, .{ t, .{ 128, 2, 2, 2 } });
            try list.add(mod.cases.TestReluSqr, .{ t, .{ 5, 7, 11, 13 } });
        }

        // fused unary + mul (gated activations that are not expressed as GGML_OP_GLU)
        for ([_]mod.c.enum_ggml_unary_op{ mod.c.GGML_UNARY_OP_GELU, mod.c.GGML_UNARY_OP_SILU, mod.c.GGML_UNARY_OP_RELU, mod.c.GGML_UNARY_OP_SIGMOID, mod.c.GGML_UNARY_OP_SOFTPLUS }) |op| {
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_BF16 }) |t| {
                for ([_]bool{ false, true }) |swap| {
                    try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, swap });
                }
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 5, 7, 11, 13 } });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "pad_unary" });
                // a view only stays out from between the two ops when the unary result is second
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, true, "pad_other" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, true, "halves" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "packed", "consumer" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "bcast" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "rep_ne0" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "view_mid" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "gate" });
                // must not fuse
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "strided_dim1" });
                try list.add(mod.cases.TestUnaryMul, .{ op, t, .{ 128, 2, 2, 2 }, false, "packed", "reuse" });
            }
        }

        // fused add + unary + mul: the delta-net alpha gate, bias and scale broadcast over dim 0
        for ([_]mod.c.enum_ggml_unary_op{ mod.c.GGML_UNARY_OP_SILU, mod.c.GGML_UNARY_OP_SIGMOID, mod.c.GGML_UNARY_OP_SOFTPLUS }) |op| {
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 7, 1, 1 } });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 512, 1, 1 } });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 5, 7, 11, 13 } });
            // one token: no broadcast left, and the unary result may be either MUL operand
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 1, 1, 1 }, false, "same_shape" });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 1, 1, 1 }, true, "same_shape" });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 7, 1, 1 }, false, "bcast", "consumer" });
            // must not fuse
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 7, 1, 1 }, false, "same_shape" });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 7, 1, 1 }, false, "rep_ne0" });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F32, .{ 32, 7, 1, 1 }, false, "bcast", "reuse" });
            try list.add(mod.cases.TestAddUnaryMul, .{ op, mod.c.GGML_TYPE_F16, .{ 32, 7, 1, 1 } });
        }
        // a unary op with no fused kernel must fall back to the three-op chain
        try list.add(mod.cases.TestAddUnaryMul, .{ mod.c.GGML_UNARY_OP_GELU, mod.c.GGML_TYPE_F32, .{ 32, 7, 1, 1 } });

        // SNAKE activation fusion: x + sin(a*x)^2 * inv_b
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16 }) |t| {
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 5, 7, 1, 1 } }); // primes sub-block
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 33, 32, 1, 1 } }); // boundary
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 1025, 13, 1, 1 } }); // large prime, grid-stride
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 128, 16, 1, 1 } }); // power-of-two
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 256, 192, 1, 1 } }); // BigVGAN-ish
            // higher-rank shapes: matcher must reject fusion, fallback to naive chain
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 64, 32, 2, 1 } }); // ne[2] > 1
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 64, 32, 1, 2 } }); // ne[3] > 1
            try list.add(mod.cases.TestSnakeFuse, .{ t, .{ 64, 32, 2, 3 } }); // ne[2] > 1 and ne[3] > 1
        }

        try list.add(mod.cases.TestDsv4HcComb, .{ 1, 1 });
        try list.add(mod.cases.TestDsv4HcComb, .{ 17, 4 });
        try list.add(mod.cases.TestDsv4HcComb, .{ 257, 8 });
        try list.add(mod.cases.TestDsv4HcComb, .{ 17, 20 });
        // production n_iter (DeepSeek-V4 uses 20) across batch sizes that cross
        // subgroup and workgroup boundaries; 1 = single-token decode
        for ([_]i64{ 1, 256, 336, 512, 513, 1024, 2048 }) |n_tokens| {
            try list.add(mod.cases.TestDsv4HcComb, .{ n_tokens, 20 });
        }

        try list.add(mod.cases.TestDsv4HcPre, .{ 1, 4, 1 });
        try list.add(mod.cases.TestDsv4HcPre, .{ 31, 4, 17 });
        try list.add(mod.cases.TestDsv4HcPre, .{ 128, 4, 257 });
        try list.add(mod.cases.TestDsv4HcPre, .{ 4096, 4, 21 });
        try list.add(mod.cases.TestDsv4HcPre, .{ 31, 4, 17, true });
        try list.add(mod.cases.TestDsv4HcPre, .{ 4096, 4, 21, true });
        for ([_]i64{ 1, 2, 3, 5, 8, 65 }) |n_hc| {
            try list.add(mod.cases.TestDsv4HcPre, .{ 128, n_hc, 17 });
            try list.add(mod.cases.TestDsv4HcPre, .{ 128, n_hc, 17, true });
        }

        try list.add(mod.cases.TestDsv4HcPost, .{ 1, 1 });
        try list.add(mod.cases.TestDsv4HcPost, .{ 31, 17 });
        try list.add(mod.cases.TestDsv4HcPost, .{ 128, 257 });
        try list.add(mod.cases.TestDsv4HcPost, .{ 4096, 21 });
        try list.add(mod.cases.TestDsv4HcPost, .{ 31, 17, true });
        try list.add(mod.cases.TestDsv4HcPost, .{ 4096, 21, true });
        try list.add(mod.cases.TestDsv4HcPost, .{ 31, 17, true, true });
        try list.add(mod.cases.TestDsv4HcPost, .{ 2560, 21, true, true });
        try list.add(mod.cases.TestDsv4HcPost, .{ 31, 17, false, true });
        try list.add(mod.cases.TestDsv4HcPost, .{ 4096, 1 }); // DeepSeek-V4-Flash decode
        try list.add(mod.cases.TestDsv4HcPost, .{ 2560, 1, true, true }); // Qwen4-Exp decode

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |t| {
            // blocks with extra cells and scores (k-pool)
            try list.add(mod.cases.TestTopKMask, .{t});
            try list.add(mod.cases.TestTopKMask, .{ t, 1001, 4, 70, 16, 3, true, 7 });
            try list.add(mod.cases.TestTopKMask, .{ t, 256, 4, 64, 64, 0, true, 33 });
            try list.add(mod.cases.TestTopKMask, .{ t, 512, 8, 16, 32, 7, true, 5, 1, 4 });
            try list.add(mod.cases.TestTopKMask, .{ t, 4096, 4, 1024, 512, 3, true, 1 });
            try list.add(mod.cases.TestTopKMask, .{ t, 4096, 4, 1024, 512, 3, true, 64, 1, 2 });
            // cells without scores (DSA)
            try list.add(mod.cases.TestTopKMask, .{ t, 256, 0, 0, 16, 0, false, 32 });
            try list.add(mod.cases.TestTopKMask, .{ t, 1001, 0, 0, 64, 0, false, 7, 1, 3 });
            try list.add(mod.cases.TestTopKMask, .{ t, 4096, 0, 0, 2048, 0, false, 5 });
            // one row, no blocks, fewer than 128 picks: no read past the pick list
            try list.add(mod.cases.TestTopKMask, .{ t, 1001, 0, 0, 16, 3, false, 1 });
            try list.add(mod.cases.TestTopKMask, .{ t, 256, 0, 0, 40, 0, true, 1 });
            // one row (single-row kernel) with each optional input combination
            try list.add(mod.cases.TestTopKMask, .{ t, 512, 0, 0, 16, 0, false, 1 });
            try list.add(mod.cases.TestTopKMask, .{ t, 1024, 4, 200, 32, 0, false, 1 });
            try list.add(mod.cases.TestTopKMask, .{ t, 1024, 4, 200, 32, 5, false, 1 });
            // cells with scores and extra cells
            try list.add(mod.cases.TestTopKMask, .{ t, 512, 0, 0, 32, 5, true, 9 });
            // blocks picked per head, the mask broadcast over the heads
            try list.add(mod.cases.TestTopKMask, .{ t, 512, 16, 32, 4, 0, true, 6, 4, 1 });
            try list.add(mod.cases.TestTopKMask, .{ t, 1024, 8, 128, 16, 8, false, 3, 2, 2 });
            // one mask row for all rows and streams (QSA)
            try list.add(mod.cases.TestTopKMask, .{ t, 1001, 4, 70, 16, 3, true, 7, 1, 1, false, true });
            try list.add(mod.cases.TestTopKMask, .{ t, 512, 8, 16, 32, 7, true, 5, 1, 4, false, true });
        }

        // glu ops
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_BF16 }) |t| {
            for ([_]i32{ 0, 1 }) |v| {
                var op: mod.c.enum_ggml_glu_op = 0;
                while (op < mod.c.GGML_GLU_OP_COUNT) : (op += 1) {
                    if (op == mod.c.GGML_GLU_OP_SWIGLU_OAI or op == mod.c.GGML_GLU_OP_SWIGLU_CLAMP) continue;

                    for ([_]bool{ false, true }) |swapped| {
                        try list.add(mod.cases.TestGlu, .{ op, t, .{ 128, 2, 2, 2 }, v, swapped });
                        try list.add(mod.cases.TestGlu, .{ op, t, .{ 5, 7, 11, 13 }, v, swapped });
                    }

                    try list.add(mod.cases.TestGluSplit, .{ op, t, .{ 128, 2, 2, 2 }, v });
                    try list.add(mod.cases.TestGluSplit, .{ op, t, .{ 5, 7, 11, 13 }, v });
                }
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |t| {
            for ([_]i32{ 0, 1 }) |v| {
                for ([_]f32{ 0.5, 1.702 }) |alpha| {
                    for ([_]f32{ 2.0, 7.0 }) |limit| {
                        try list.add(mod.cases.TestSwigluOai, .{ t, .{ 128, 2, 2, 2 }, v, alpha, limit });
                    }
                }
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |t| {
            for ([_]i32{ 0, 1 }) |v| {
                for ([_]f32{ 2.0, 10.0 }) |limit| {
                    try list.add(mod.cases.TestSwigluClamp, .{ t, .{ 128, 2, 2, 2 }, v, limit });
                }
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_Q4_0 }) |t| {
            try list.add(mod.cases.TestGetRows, .{ t, 300 * 256, 5, 4, 1, 2, false });
            try list.add(mod.cases.TestGetRows, .{ t, 256, 80000, 70000, 2, 1, false });
            try list.add(mod.cases.TestGetRows, .{ t, 256, 5, 4, 700, 100, false });
        }

        try list.add(mod.cases.TestGetRows, .{ mod.c.GGML_TYPE_F32, 1, 8, 2, 1, 1, false });
        for ([_]mod.c.enum_ggml_type{
            mod.c.GGML_TYPE_Q4_0,    mod.c.GGML_TYPE_Q8_0,  mod.c.GGML_TYPE_Q2_K, mod.c.GGML_TYPE_Q3_K, mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q5_K, mod.c.GGML_TYPE_Q6_K,
            mod.c.GGML_TYPE_IQ3_XXS, mod.c.GGML_TYPE_IQ3_S,
        }) |t| {
            try list.add(mod.cases.TestGetRowsAfterMulMat, .{t});
        }
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_IQ3_XXS, mod.c.GGML_TYPE_IQ3_S }) |t| {
            try list.add(mod.cases.TestMulMatReorderThenBigBatch, .{t});
        }
        for (mod.TypeLists.all_types) |t| {
            for ([_]i32{ 1, 7 }) |b| {
                for ([_]bool{ false, true }) |v| {
                    for ([_]bool{ false, true }) |vs0| {
                        try list.add(mod.cases.TestGetRows, .{ t, 256, 5, 4, b, 1, v, vs0 });
                    }
                }
            }
        }
        for ([_]i32{ 1, 7 }) |b| {
            for ([_]bool{ false, true }) |v| {
                for ([_]bool{ false, true }) |vs0| {
                    try list.add(mod.cases.TestGetRows, .{ mod.c.GGML_TYPE_I32, 256, 5, 4, b, 1, v, vs0 });
                }
            }
        }
        try list.add(mod.cases.TestGetRows, .{ mod.c.GGML_TYPE_F32, 256, 8, 2, 1, 1, false, true, 3 });

        try list.add(mod.cases.TestGetRowsBack, .{ mod.c.GGML_TYPE_F32, 1, 8, 2, 1, false });
        try list.add(mod.cases.TestGetRowsBack, .{ mod.c.GGML_TYPE_F32, 1, 70000, 4, 1, false }); // row count > CUDA grid-y limit (65535)
        for (mod.TypeLists.all_types) |t| {
            for ([_]bool{ false, true }) |v| {
                try list.add(mod.cases.TestGetRowsBack, .{ t, 256, 5, 4, 1, v });
            }
        }
        for ([_]bool{ false, true }) |v| {
            try list.add(mod.cases.TestGetRowsBack, .{ mod.c.GGML_TYPE_I32, 256, 5, 4, 1, v });
        }

        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_I64, .{ 1, 8, 1, 3 }, .{ 1, 1 }, 2, false });
        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_I32, .{ 1, 8, 1, 3 }, .{ 1, 1 }, 2, false });
        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_I32, .{ 256, 5, 1, 3 }, .{ 1, 1 }, 1, false });
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |src_type| {
            for (mod.TypeLists.all_types) |t| {
                const blck = mod.c.ggml_blck_size(t);
                for ([_]i64{ 1, 7 }) |b| {
                    for ([_]bool{ false, true }) |v| {
                        try list.add(mod.cases.TestSetRows, .{ src_type, t, mod.c.GGML_TYPE_I64, .{ 256, 5, b, 3 }, .{ 1, 1 }, 1, v });
                        try list.add(mod.cases.TestSetRows, .{ src_type, t, mod.c.GGML_TYPE_I64, .{ 256, 11, 1, b }, .{ 2, 3 }, 7, v });

                        try list.add(mod.cases.TestSetRows, .{ src_type, t, mod.c.GGML_TYPE_I64, .{ 3 * blck, 3, b, 1 }, .{ 2, 3 }, 2, v });

                        if (blck == 1) {
                            try list.add(mod.cases.TestSetRows, .{ src_type, t, mod.c.GGML_TYPE_I64, .{ 31, 3, b, 1 }, .{ 2, 3 }, 2, v });
                            try list.add(mod.cases.TestSetRows, .{ src_type, t, mod.c.GGML_TYPE_I64, .{ 33, 5, 1, b }, .{ 2, 3 }, 1, v });
                        }
                    }
                }
            }
        }
        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_I64, .{ 1, 8, 1, 3 }, .{ 1, 1 }, 2, false });
        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_I32, .{ 1, 8, 1, 3 }, .{ 1, 1 }, 2, false });
        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_I64, .{ 1, 8, 1, 3 }, .{ 1, 1 }, 2, true });
        try list.add(mod.cases.TestSetRows, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_I32, .{ 1, 8, 1, 3 }, .{ 1, 1 }, 2, true });

        for ([_]i32{ mod.c.GGML_ROPE_TYPE_NORMAL, mod.c.GGML_ROPE_TYPE_NEOX, mod.c.GGML_ROPE_TYPE_MROPE, mod.c.GGML_ROPE_TYPE_VISION, mod.c.GGML_ROPE_TYPE_IMROPE }) |mode| {
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |t| {
                for ([_]i64{ 1, 8, 512 }) |ne2| {
                    try list.add(mod.cases.TestRopeSetRows, .{ t, mod.c.GGML_TYPE_I64, .{ 128, 32, ne2, 1 }, mode });
                    try list.add(mod.cases.TestRopeSetRows, .{ t, mod.c.GGML_TYPE_I64, .{ 128, 32, ne2, 3 }, mode });
                }
            }
        }
        try list.add(mod.cases.TestRopeSetRows, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_I32, .{ 128, 32, 8, 1 }, mod.c.GGML_ROPE_TYPE_IMROPE });
        // partial rotation
        for ([_]i32{ mod.c.GGML_ROPE_TYPE_NORMAL, mod.c.GGML_ROPE_TYPE_NEOX, mod.c.GGML_ROPE_TYPE_MROPE, mod.c.GGML_ROPE_TYPE_IMROPE }) |mode| {
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |t| {
                try list.add(mod.cases.TestRopeSetRows, .{ t, mod.c.GGML_TYPE_I64, .{ 256, 4, 8, 1 }, mode, 64 });
            }
        }

        for ([_]mod.c.enum_ggml_type{mod.c.GGML_TYPE_F32}) |type_input| {
            for ([_]mod.c.enum_ggml_op_pool{ mod.c.GGML_OP_POOL_AVG, mod.c.GGML_OP_POOL_MAX }) |pool_type| {
                for ([_]i32{ 1, 3 }) |k0| {
                    for ([_]i32{ 1, 3 }) |k1| {
                        for ([_]i32{ 1, 2 }) |s0| {
                            for ([_]i32{ 1, 2 }) |s1| {
                                for ([_]i32{ 0, 1 }) |p0| {
                                    for ([_]i32{ 0, 1 }) |p1| {
                                        try list.add(mod.cases.TestPool2d, .{ pool_type, type_input, .{ 10, 10, 3, 1 }, k0, k1, s0, s1, p0, p1 });
                                    }
                                }
                            }
                        }
                    }
                }
                // MSA indexer block-max shape: wide 1-D window along ne0, no padding, non-divisible width
                for ([_]i64{ 4096, 4160 }) |iw| {
                    for ([_]i32{ 32, 64, 128 }) |blk| {
                        try list.add(mod.cases.TestPool2d, .{ pool_type, type_input, .{ iw, 2, 1, 1 }, blk, 1, blk, 1, 0, 0 });
                    }
                }
                try list.add(mod.cases.TestPool2d, .{ pool_type, type_input, .{ 259, 5, 3, 2 }, 128, 3, 64, 2, 64, 1 });
                try list.add(mod.cases.TestPool2d, .{ pool_type, type_input, .{ 193, 2, 1, 1 }, 128, 1, 128, 1, 128, 2 });
            }
        }

        for ([_]mod.c.enum_ggml_type{mod.c.GGML_TYPE_F32}) |type_input| {
            for ([_]mod.c.enum_ggml_op_pool{ mod.c.GGML_OP_POOL_AVG, mod.c.GGML_OP_POOL_MAX }) |pool_type| {
                for ([_]i32{ 1, 2, 3 }) |k0| {
                    for ([_]i32{ 1, 2, 3 }) |s0| {
                        for ([_]i32{ 0, 1, 2, 3 }) |p0| {
                            try list.add(mod.cases.TestPool1d, .{ pool_type, type_input, .{ 10, 3, 2, 1 }, k0, s0, p0 });
                            try list.add(mod.cases.TestPool1d, .{ pool_type, type_input, .{ 11, 1, 3, 2 }, k0, s0, p0 });
                            try list.add(mod.cases.TestPool1d, .{ pool_type, type_input, .{ 128, 2, 1, 3 }, k0, s0, p0 });
                        }
                    }
                }
            }
        }

        // (upstream `#if 0`: >4GB im2col destination, too slow to run by default.)

        // im2col 1D
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 3000, 128, 1, 1 }, .{ 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false });
        // deepseek4v / deepseek41v aligner: 3x3 stride-3 unfold (F.unfold == im2col)
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 48, 33, 1024, 1 }, .{ 3, 3, 1024, 1 }, 3, 3, 0, 0, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, .{ 3000, 128, 1, 1 }, .{ 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 3000, 128, 1, 1 }, .{ 3, 128, 1280, 1 }, 1, 0, 1, 0, 1, 0, false });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 3000, 384, 1, 1 }, .{ 3, 384, 384, 1 }, 1, 0, 1, 0, 1, 0, false });
        // 1D convs of an audio decoder: long sequences, dilated taps, pointwise kernels
        for ([_]i64{ 1, 3, 9 }) |d0| {
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 2000 + 6 * d0, 96, 1, 1 }, .{ 7, 96, 96, 1 }, 1, 0, 0, 0, d0, 0, false });
        }
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 1500, 192, 1, 1 }, .{ 1, 192, 192, 1 }, 1, 0, 0, 0, 1, 0, false });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, .{ 1001, 64, 1, 1 }, .{ 5, 64, 64, 1 }, 2, 0, 2, 0, 1, 0, false });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 777, 48, 2, 1 }, .{ 3, 48, 48, 1 }, 1, 0, 4, 0, 3, 0, false });
        // several rows of a few output-col blocks each: a thread's range of blocks can start mid-row
        for ([_]i64{ 5, 7 }) |n| {
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 600, 8, n, 1 }, .{ 3, 8, 8, 1 }, 1, 0, 1, 0, 1, 0, false });
        }
        // 2D with a kernel one row tall: with p1 > 0 some output rows fall outside the input height
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 40, 6, 8, 1 }, .{ 3, 1, 8, 4 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 40, 6, 8, 1 }, .{ 3, 1, 8, 4 }, 1, 1, 1, 0, 1, 1, true });
        for ([_]i32{ 1, 3 }) |s0| {
            for ([_]i32{ 0, 3 }) |p0| {
                for ([_]i32{ 1, 3 }) |d0| {
                    try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 20, 2, 2, 1 }, .{ 3, 2, 2, 1 }, s0, 0, p0, 0, d0, 0, false });
                }
            }
        }

        // im2col 2D
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32 });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        for ([_]i32{ 1, 3 }) |s0| {
            for ([_]i32{ 1, 3 }) |s1| {
                for ([_]i32{ 0, 3 }) |p0| {
                    for ([_]i32{ 0, 3 }) |p1| {
                        for ([_]i32{ 1, 3 }) |d0| {
                            for ([_]i32{ 1, 3 }) |d1| {
                                try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 20, 20, 2, 2 }, .{ 3, 3, 2, 2 }, s0, s1, p0, p1, d0, d1, true });
                            }
                        }
                    }
                }
            }
        }

        // extra tests for im2col 2D
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 1, 32 }, .{ 3, 3, 1, 32 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 2, 32 }, .{ 3, 3, 2, 32 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 1, 1024 }, .{ 3, 3, 1, 1024 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 2, 1024 }, .{ 3, 3, 2, 1024 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 1, 2048 }, .{ 3, 3, 1, 2048 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 2, 2048 }, .{ 3, 3, 2, 2048 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 1, 2560 }, .{ 3, 3, 1, 2560 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 12, 12, 2, 2560 }, .{ 3, 3, 2, 2560 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 5, 5, 1, 32 }, .{ 3, 4, 1, 32 }, 1, 1, 0, 0, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 2, 2, 1536, 729 }, .{ 2, 2, 1536, 4096 }, 1, 1, 0, 0, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 128, 128, 1, 2 }, .{ 32, 33, 1, 2 }, 1, 1, 1, 1, 1, 1, true });
        try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 128, 128, 2, 1 }, .{ 33, 34, 2, 1 }, 1, 1, 1, 1, 1, 1, true });
        // non-overlapping (stride == kernel, no pad/dilation) with IC*KH well above 256, e.g. SD1.5 1x1 convs
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |dst_type| {
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, dst_type, .{ 64, 64, 320, 1 }, .{ 1, 1, 320, 1 }, 1, 1, 0, 0, 1, 1, true });
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, dst_type, .{ 16, 16, 1280, 2 }, .{ 1, 1, 1280, 1 }, 1, 1, 0, 0, 1, 1, true });
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, dst_type, .{ 8, 8, 2560, 1 }, .{ 1, 1, 2560, 1 }, 1, 1, 0, 0, 1, 1, true });
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, dst_type, .{ 32, 32, 192, 1 }, .{ 2, 2, 192, 1 }, 2, 2, 0, 0, 1, 1, true });
            try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, dst_type, .{ 64, 512, 1, 1 }, .{ 2, 512, 1, 1 }, 2, 0, 0, 0, 1, 0, false });
        }

        // im2col 3D
        try list.add(mod.cases.TestIm2col3d, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32 });
        try list.add(mod.cases.TestIm2col3d, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 });
        try list.add(mod.cases.TestIm2col3d, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        for ([_]i32{ 1, 3 }) |s0| {
            for ([_]i32{ 1, 3 }) |s1| {
                for ([_]i32{ 1, 3 }) |s2| {
                    for ([_]i32{ 0, 3 }) |p0| {
                        for ([_]i32{ 0, 3 }) |p1| {
                            for ([_]i32{ 0, 3 }) |p2| {
                                for ([_]i32{ 1, 3 }) |d0| {
                                    for ([_]i32{ 1, 3 }) |d1| {
                                        for ([_]i32{ 1, 3 }) |d2| {
                                            for ([_]i64{ 1, 3 }) |ic| {
                                                for ([_]bool{ false, true }) |v| {
                                                    try list.add(mod.cases.TestIm2col3d, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 20, 20, 10, 3 }, .{ 3, 3, 3, 3 }, ic, s0, s1, s2, p0, p1, p2, d0, d1, d2, v });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Conv_2D test cases: upstream's DETAILED_TESTS block is not compiled by default.

        // CONV_2D:
        //const s0 = 3;
        const s1: i64 = 5;
        const p0: i64 = 5;
        //const p1 = 2;
        const d0: i64 = 2;
        const d1: i64 = 4;

        for ([_]i64{ 1, 3 }) |s0| {
            for ([_]i64{ 2, 5 }) |p1| {
                for ([_]i64{ 1, 25 }) |cin| {
                    for ([_]i64{ 1, 12 }) |cout| {
                        for ([_]i64{ 1, 2, 3, 11 }) |kh| {
                            for ([_]i64{ 1, 2, 3, 11 }) |kw| {
                                for ([_]i64{ 1, 133 }) |h| {
                                    for ([_]i64{ 1, 141 }) |w| {
                                        if (calcConvOutputSize(w, kw, s0, p0, d0) > 0 and
                                            calcConvOutputSize(h, kh, s1, p1, d1) > 0)
                                        {
                                            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
                                                try list.add(mod.cases.TestConv2d, .{ .{ w, h, cin, 2 }, .{ kw, kh, cin, cout }, kernel_type, s0, s1, p0, p1, d0, d1, false }); // cwhn = false
                                                try list.add(mod.cases.TestConv2d, .{ .{ w, h, cin, 2 }, .{ kw, kh, cin, cout }, kernel_type, s0, s1, p0, p1, d0, d1, true }); // cwhn = true
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
            try list.add(mod.cases.TestConv2d, .{ .{ 256, 256, 192, 1 }, .{ 3, 3, 192, 96 }, kernel_type, 1, 1, 1, 1, 1, 1, false }); // cwhn = false
            try list.add(mod.cases.TestConv2d, .{ .{ 256, 256, 192, 1 }, .{ 3, 3, 192, 96 }, kernel_type, 1, 1, 1, 1, 1, 1, true }); // cwhn = true
        }
        try list.add(mod.cases.TestConv2d, .{ .{ 19, 17, 8, 2 }, .{ 3, 3, 8, 65 }, mod.c.GGML_TYPE_F16, 1, 1, 1, 1, 1, 1 });
        try list.add(mod.cases.TestConv2d, .{ .{ 19, 17, 16, 3 }, .{ 3, 3, 16, 33 }, mod.c.GGML_TYPE_F16, 2, 3, 4, 2, 2, 1 });
        try list.add(mod.cases.TestConv2d, .{ .{ 13, 11, 16, 3 }, .{ 1, 1, 16, 33 }, mod.c.GGML_TYPE_F16, 1, 1, 0, 0, 1, 1 });
        try list.add(mod.cases.TestConv2d, .{ .{ 19, 17, 8, 2 }, .{ 3, 3, 8, 17 }, mod.c.GGML_TYPE_F16, 1, 1, 1, 1, 1, 1, false, 1 });

        // (upstream keeps two commented-out im2col cases with large input W and H, for sycl.)

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16 }) |kernel_type| {
            try list.add(mod.cases.TestConv1dDw, .{ .{ 64, 16, 1, 1 }, .{ 3, 1, 16, 1 }, kernel_type, 1, 0, 1 });
            try list.add(mod.cases.TestConv1dDw, .{ .{ 64, 16, 1, 1 }, .{ 7, 1, 16, 1 }, kernel_type, 1, 3, 1 });
            try list.add(mod.cases.TestConv1dDw, .{ .{ 97, 33, 1, 1 }, .{ 5, 1, 33, 1 }, kernel_type, 2, 2, 2 });
        }

        try list.add(mod.cases.TestConv2dDw, .{ .{ 17, 34, 9, 1 }, .{ 3, 3, 1, 9 }, mod.c.GGML_TYPE_F32, 1, 0, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 17, 34, 9, 1 }, .{ 3, 3, 1, 9 }, mod.c.GGML_TYPE_F32, 1, 0, 1, true });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 32, 8, 64, 1 }, .{ 3, 3, 1, 64 }, mod.c.GGML_TYPE_F32, 2, 1, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 32, 8, 64, 1 }, .{ 3, 3, 1, 64 }, mod.c.GGML_TYPE_F32, 2, 1, 1, true });

        try list.add(mod.cases.TestConv2dDw, .{ .{ 17, 34, 9, 1 }, .{ 3, 3, 1, 9 }, mod.c.GGML_TYPE_F16, 1, 0, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 17, 34, 9, 1 }, .{ 3, 3, 1, 9 }, mod.c.GGML_TYPE_F16, 1, 0, 1, true });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 32, 8, 64, 1 }, .{ 3, 3, 1, 64 }, mod.c.GGML_TYPE_F16, 2, 1, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 32, 8, 64, 1 }, .{ 3, 3, 1, 64 }, mod.c.GGML_TYPE_F16, 2, 1, 1, true });

        // CONV_3D (upstream's `calc_conv_output_size_3d` is the same formula as `calcConvOutputSize`)
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
            try list.add(mod.cases.TestConv3d, .{ 1, 4, 6, 78, 54, 512, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 2, 16, 4, 64, 64, 256, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
            for ([_]i64{ 1, 2 }) |n| {
                for ([_]i64{ 1, 3 }) |ic| {
                    for ([_]i64{ 1, 4 }) |oc| {
                        for ([_]i64{ 1, 2 }) |cs0| {
                            for ([_]i64{ 0, 1 }) |cp1| {
                                for ([_]i64{ 1, 2 }) |cd2| {
                                    const iw: i64 = 20;
                                    const ih: i64 = 22;
                                    const id: i64 = 18;
                                    const kw: i64 = 3;
                                    const kh: i64 = 3;
                                    const kd: i64 = 3;
                                    const cs1 = cs0;
                                    const cs2 = cs0;
                                    const cp0 = cp1;
                                    const cp2 = cp1;
                                    const cd0 = cd2;
                                    const cd1 = cd2;

                                    if (calcConvOutputSize(iw, kw, cs0, cp0, cd0) <= 0 or
                                        calcConvOutputSize(ih, kh, cs1, cp1, cd1) <= 0 or
                                        calcConvOutputSize(id, kd, cs2, cp2, cd2) <= 0)
                                    {
                                        continue;
                                    }
                                    try list.add(mod.cases.TestConv3d, .{ n, ic, id, ih, iw, oc, kd, kh, kw, cs0, cs1, cs2, cp0, cp1, cp2, cd0, cd1, cd2, kernel_type });

                                    // Asymmetric kernel and params
                                    const asym_kw: i64 = 5;
                                    const asym_kh: i64 = 1;
                                    const asym_kd: i64 = 3;
                                    const asym_s0: i64 = 2;
                                    const asym_s1: i64 = 1;
                                    const asym_s2: i64 = 1;
                                    const asym_p0: i64 = 2;
                                    const asym_p1: i64 = 0;
                                    const asym_p2: i64 = 1;
                                    const asym_d0: i64 = 1;
                                    const asym_d1: i64 = 1;
                                    const asym_d2: i64 = 2;

                                    if (calcConvOutputSize(iw, asym_kw, asym_s0, asym_p0, asym_d0) <= 0 or
                                        calcConvOutputSize(ih, asym_kh, asym_s1, asym_p1, asym_d1) <= 0 or
                                        calcConvOutputSize(id, asym_kd, asym_s2, asym_p2, asym_d2) <= 0)
                                    {
                                        continue;
                                    }
                                    try list.add(mod.cases.TestConv3d, .{ n, ic, id, ih, iw, oc, asym_kd, asym_kh, asym_kw, asym_s0, asym_s1, asym_s2, asym_p0, asym_p1, asym_p2, asym_d0, asym_d1, asym_d2, kernel_type });
                                }
                            }
                        }
                    }
                }
            }
            // Case with kernel size 1
            try list.add(mod.cases.TestConv3d, .{ 1, 4, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 2, 8, 5, 11, 9, 65, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 2, 5, 7, 9, 13, 17, 2, 3, 4, 2, 1, 3, 3, 2, 2, 2, 1, 2, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 3, 16, 3, 7, 9, 33, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 2, 8, 7, 5, 9, 33, 3, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 2, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 2, 3, 1, 2, 1, 7, 1, 1, 1, 1, 1, 1, 3, 4, 2, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 2, 8, 5, 7, 9, 17, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, kernel_type, 1 });
            try list.add(mod.cases.TestConv3d, .{ 1, 1, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 1, 1, 2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
            try list.add(mod.cases.TestConv3d, .{ 1, 1, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type });
        }

        for ([_]i64{ 1, 9 }) |cout| {
            for ([_]i64{ 1, 7 }) |cin| {
                for ([_]i64{ 1, 3, 1337 }) |k| {
                    for ([_]i64{ 1, 2, 13 }) |l| {
                        for ([_]i32{ 1, 2, 3 }) |s0| {
                            try list.add(mod.cases.TestConvTranspose1d, .{ .{ l, cin, 1, 1 }, .{ k, cout, cin, 1 }, s0, 0, 1 });
                        }
                    }
                }
            }
        }

        try list.add(mod.cases.TestConvTranspose1d, .{});
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 3, 2, 1, 1 }, .{ 2, 3, 2, 1 }, 3, 0, 1 });
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 3, 2, 1, 1 }, .{ 2, 3, 2, 1 }, 2, 0, 1 });
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 3, 2, 1, 1 }, .{ 2, 3, 2, 1 }, 1, 0, 1 });
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 3, 2, 1, 1 }, .{ 3, 2, 2, 1 }, 2, 0, 1 });
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 3, 2, 1, 1 }, .{ 3, 2, 2, 1 }, 1, 0, 1 });
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 3, 2, 1, 1 }, .{ 3, 1, 2, 1 }, 1, 0, 1 });
        try list.add(mod.cases.TestConvTranspose1d, .{ .{ 2, 1, 1, 1 }, .{ 3, 1, 1, 1 }, 1, 0, 1 });

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16 }) |t| {
            // ConvTranspose1d expressed as mul_mat + col2im (DAC decoder upsampling)
            try list.add(mod.cases.TestCol2im1d, .{ t, 16, 32, 197, 8, 0 }); // kernel = 2*stride
            try list.add(mod.cases.TestCol2im1d, .{ t, 4, 3, 7, 2, 0 });
            try list.add(mod.cases.TestCol2im1d, .{ t, 1, 5, 13, 1, 0 }); // stride 1, no overlap
            try list.add(mod.cases.TestCol2im1d, .{ t, 6, 4, 11, 3, 1 }); // with cropping
            try list.add(mod.cases.TestCol2im1d, .{ t, 2, 3, 9, 3, 0 }); // kernel < stride, gap positions are zeroed
            try list.add(mod.cases.TestCol2im1d, .{ t, 5, 4, 11, 2, 0 }); // kernel not a multiple of stride, alternating overlap
            try list.add(mod.cases.TestCol2im1d, .{ t, 8, 4, 13, 4, 2 }); // padding = stride/2 (DAC causal cropping)
            try list.add(mod.cases.TestCol2im1d, .{ t, 4, 3, 1, 2, 0 }); // single column, pure kernel unfold
            try list.add(mod.cases.TestCol2im1d, .{ t, 16, 1, 197, 8, 0 }); // OC = 1, mono output stage
            try list.add(mod.cases.TestCol2im1d, .{ t, 1, 5, 13, 3, 0 }); // K = 1 with stride > 1, sparse scatter
            try list.add(mod.cases.TestCol2im1d, .{ t, 8, 2, 3, 2, 5 }); // cropping eats most of the signal, T_out = 2
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 3, 2, 3, 1 }, .{ 2, 2, 1, 3 }, 1, kernel_type });
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 10, 10, 9, 1 }, .{ 3, 3, 1, 9 }, 2, kernel_type });
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 129, 63, 35, 1 }, .{ 3, 3, 48, 35 }, 1, kernel_type });
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 10, 10, 9, 2 }, .{ 3, 3, 1, 9 }, 2, kernel_type }); // for multiple batches
        }

        try list.add(mod.cases.TestCountEqual, .{ mod.c.GGML_TYPE_F32, .{ 4, 500, 1, 1 } });
        try list.add(mod.cases.TestCountEqual, .{ mod.c.GGML_TYPE_F32, .{ 4, 5000, 1, 1 } });

        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 32, 1, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 32, 513, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 100, 10, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 1024, 10, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 1024, 12, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 2000, 10, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 5438, 3, 1, 1 } });

        for ([_]i64{ 1, 3 }) |ne3| { // CUDA backward pass only supports ne3 == 1
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_F32, .{ 10, 5, 4, ne3 }, .{ 1, 1, 1, 1 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_F32, .{ 10, 5, 4, ne3 }, .{ 2, 1, 1, 1 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_F32, .{ 10, 5, 4, ne3 }, .{ 1, 2, 1, 1 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_F32, .{ 10, 5, 4, ne3 }, .{ 1, 1, 2, 1 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_F32, .{ 10, 5, 4, ne3 }, .{ 1, 1, 1, 2 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_F16, .{ 10, 5, 4, ne3 }, .{ 2, 1, 1, 1 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_I32, .{ 10, 5, 4, ne3 }, .{ 2, 1, 1, 1 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_I16, .{ 10, 5, 4, ne3 }, .{ 1, 1, 1, 2 } });
            try list.add(mod.cases.TestRepeat, .{ mod.c.GGML_TYPE_BF16, .{ 10, 5, 4, ne3 }, .{ 2, 1, 1, 1 } });
        }

        for ([_]bool{ false, true }) |view| {
            try list.add(mod.cases.TestRepeatBack, .{ mod.c.GGML_TYPE_F32, .{ 8, 6, 4, 2 }, .{ 1, 1, 1, 1 }, view });
            try list.add(mod.cases.TestRepeatBack, .{ mod.c.GGML_TYPE_F32, .{ 8, 6, 4, 2 }, .{ 2, 1, 1, 1 }, view });
            try list.add(mod.cases.TestRepeatBack, .{ mod.c.GGML_TYPE_F32, .{ 8, 6, 4, 2 }, .{ 1, 2, 1, 1 }, view });
            try list.add(mod.cases.TestRepeatBack, .{ mod.c.GGML_TYPE_F32, .{ 8, 6, 4, 2 }, .{ 1, 1, 2, 1 }, view });
            try list.add(mod.cases.TestRepeatBack, .{ mod.c.GGML_TYPE_F32, .{ 8, 6, 4, 2 }, .{ 1, 1, 1, 2 }, view });
        }

        try list.add(mod.cases.TestDup, .{mod.c.GGML_TYPE_F32});
        try list.add(mod.cases.TestDup, .{mod.c.GGML_TYPE_F16});
        try list.add(mod.cases.TestDup, .{mod.c.GGML_TYPE_I32});
        try list.add(mod.cases.TestDup, .{mod.c.GGML_TYPE_I16});
        try list.add(mod.cases.TestDup, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 5, 1 }, .{ 0, 2, 1, 3 } });
        try list.add(mod.cases.TestDup, .{ mod.c.GGML_TYPE_F16, .{ 10, 10, 5, 1 }, .{ 0, 2, 1, 3 } }); // dup by rows
        try list.add(mod.cases.TestDup, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 5, 1 }, .{ 1, 0, 2, 3 } });
        try list.add(mod.cases.TestDup, .{ mod.c.GGML_TYPE_F16, .{ 10, 10, 5, 1 }, .{ 1, 0, 2, 3 } }); // dup dst not-contiguous
        try list.add(mod.cases.TestDup, .{ mod.c.GGML_TYPE_I16, .{ 10, 8, 3, 1 }, .{ 0, 2, 1, 3 } });
        try list.add(mod.cases.TestDup, .{ mod.c.GGML_TYPE_I16, .{ 10, 8, 3, 1 }, .{ 1, 2, 0, 3 } });

        {
            var dim: i32 = 1;
            while (dim < mod.c.GGML_MAX_DIMS) : (dim += 1) {
                try list.add(mod.cases.TestSet, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 6, 5, 4, 3 }, dim, false });
                try list.add(mod.cases.TestSet, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 6, 5, 4, 3 }, dim, true });
            }
        }

        {
            var dim: i32 = 1;
            while (dim < mod.c.GGML_MAX_DIMS) : (dim += 1) {
                try list.add(mod.cases.TestSet, .{ mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_I32, .{ 6, 5, 4, 3 }, dim, false });
                try list.add(mod.cases.TestSet, .{ mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_I32, .{ 6, 5, 4, 3 }, dim, true });
            }
        }

        // same-type copy
        for (mod.TypeLists.all_types) |t| {
            const nk = mod.c.ggml_blck_size(t);

            var k: i64 = 1;
            while (k < 4) : (k += 1) {
                try list.add(mod.cases.TestCpy, .{ t, t, .{ k * nk, 2, 3, 4 } });
                try list.add(mod.cases.TestCpy, .{ t, t, .{ k * nk, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } });
                try list.add(mod.cases.TestCpy, .{ t, t, .{ k * nk, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 0, 3, 1, 2 }, .{ 0, 2, 1, 3 } });
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_F32 }) |type_src| {
            for (mod.TypeLists.all_types) |type_dst| {
                try list.add(mod.cases.TestCpy, .{ type_src, type_dst, .{ 256, 4, 4, 4 } });
                try list.add(mod.cases.TestCpy, .{ type_src, type_dst, .{ 256, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } }); // cpy by rows
            }
        }
        for (mod.TypeLists.all_types) |type_src| {
            for ([_]mod.c.enum_ggml_type{mod.c.GGML_TYPE_F32}) |type_dst| {
                try list.add(mod.cases.TestCpy, .{ type_src, type_dst, .{ 256, 4, 4, 4 } });
                try list.add(mod.cases.TestCpy, .{ type_src, type_dst, .{ 256, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } }); // cpy by rows
            }
        }
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |type_src| {
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |type_dst| {
                try list.add(mod.cases.TestCpy, .{ type_src, type_dst, .{ 256, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 } }); // cpy not-contiguous
            }
        }
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16 }) |type_src| {
            try list.add(mod.cases.TestCpy, .{ type_src, mod.c.GGML_TYPE_F8_E4M3, .{ 32, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, false, .{ 64, 2, 3, 4 } });
            try list.add(mod.cases.TestCpy, .{ type_src, mod.c.GGML_TYPE_F8_E4M3, .{ 32, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 }, .{ 1, 0, 2, 3 } });
            try list.add(mod.cases.TestCpy, .{ type_src, mod.c.GGML_TYPE_F8_E4M3, .{ 32, 2, 3, 4 }, .{ 16, 4, 3, 4 }, .{ 1, 0, 2, 3 }, .{ 0, 2, 1, 3 } });
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16 }) |type_dst| {
            try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F8_E4M3, type_dst, .{ 33, 2, 3, 4 } });
            try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F8_E4M3, type_dst, .{ 32, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, false, .{ 64, 2, 3, 4 } });
            try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F8_E4M3, type_dst, .{ 32, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 }, .{ 1, 0, 2, 3 } });
            try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F8_E4M3, type_dst, .{ 32, 2, 3, 4 }, .{ 16, 4, 3, 4 }, .{ 1, 0, 2, 3 }, .{ 0, 2, 1, 3 } });
        }

        // quant block count not a multiple of the kernel block size
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_Q4_0, .{ 96, 1, 1, 1 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_F32, .{ 96, 1, 1, 1 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_I32, .{ 256, 2, 3, 4 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_I32, .{ 256, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_F32, .{ 256, 2, 3, 4 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_F32, .{ 256, 2, 3, 4 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 256, 4, 3, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 256, 4, 3, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 256, 4, 3, 3 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_BF16, .{ 256, 4, 3, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 256, 4, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 256, 4, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_BF16, .{ 256, 4, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_I32, .{ 256, 4, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_I32, .{ 256, 1, 4, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 2, 0, 3 }, .{ 0, 0, 0, 0 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 256, 1, 4, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 2, 0, 3 }, .{ 0, 0, 0, 0 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 2, 2097121, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 1, 64, 48, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } }); // ne[0] == 1 permute
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 1, 64, 48, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } }); // ne[0] == 1 permute
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 2, 2, 524281, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 128, 2, 3, 1 }, .{ 128, 2, 3, 1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, false, .{ 128, 4, 3, 1 } }); // strided dst
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 128, 2, 3, 1 }, .{ 128, 2, 3, 1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, false, .{ 128, 4, 3, 1 } }); // strided dst
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 64, 16, 1, 1 }, .{ 1, 64, 16, 1 }, .{ 1, 2, 0, 3 }, .{ 0, 0, 0, 0 }, false, .{ 32, 64, 16, 1 } }); // permuted src, strided dst
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 64, 16, 1, 1 }, .{ 1, 64, 16, 1 }, .{ 1, 2, 0, 3 }, .{ 0, 0, 0, 0 }, false, .{ 32, 64, 16, 1 } }); // permuted src, strided dst

        // CPY - different src/dst shapes (reshaping via CPY)
        // Use permutations of {3, 5, 7, 32}. Total elements: 3*5*7*32 = 3360.
        // Each src permutation is tested against canonical sorted and reverse dst (skip self).
        {
            var dims = [4]i64{ 3, 5, 7, 32 };
            std.mem.sort(i64, &dims, {}, std.sort.asc(i64));
            const canonical = dims;
            const reversed = [4]i64{ 32, 7, 5, 3 };
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |t| {
                var cur = dims;
                while (true) {
                    if (!std.mem.eql(i64, &cur, &canonical)) {
                        try list.add(mod.cases.TestCpy, .{ t, t, cur, canonical });
                    }
                    if (!std.mem.eql(i64, &cur, &reversed)) {
                        try list.add(mod.cases.TestCpy, .{ t, t, cur, reversed });
                    }
                    if (cur[0] == 32 and t == mod.c.GGML_TYPE_F32) {
                        if (canonical[0] == 32) {
                            try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_0, cur, canonical });
                        }
                        if (reversed[0] == 32) {
                            try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_0, cur, reversed });
                        }
                    }
                    nextPermutation(&cur);
                    if (std.mem.eql(i64, &cur, &canonical)) break;
                }
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16 }) |type_dst| {
            for ([_]bool{ true, false }) |use_view_slice| {
                for ([_][4]i64{ .{ 2, 1, 1, 1 }, .{ 2, 1, 3, 5 }, .{ 2, 3, 5, 7 }, .{ 1, 4, 4, 1 }, .{ 1, 8, 17, 1 }, .{ 10, 10, 10, 1 } }) |ne| {
                    if (use_view_slice and (type_dst == mod.c.GGML_TYPE_F16 or type_dst == mod.c.GGML_TYPE_BF16)) {
                        continue; // TODO: add after WebGPU is fixed
                    }
                    try list.add(mod.cases.TestCont, .{ type_dst, ne, use_view_slice });
                }
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |type_dst| {
            for ([_][4]i64{
                .{ 10, 10, 10, 1 },   .{ 33, 5, 7, 1 },     .{ 64, 3, 65, 1 },    .{ 2, 3, 5, 7 },
                // large, tile-aligned and tile-unaligned, matching the perf cases
                .{ 1024, 64, 64, 1 }, .{ 2304, 64, 64, 1 }, .{ 1000, 33, 65, 1 },
            }) |ne| {
                for ([_][4]i64{
                    .{ 2, 1, 0, 3 }, // 0<->2 swap
                    .{ 1, 2, 0, 3 }, // 3-cycle
                    .{ 0, 2, 1, 3 },
                }) |perm| {
                    try list.add(mod.cases.TestCont, .{ type_dst, ne, false, perm });
                }
            }
        }

        // source strides over 16 MB (24-bit DMA descriptor limit)
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ 512, 8193, 2, 1 }, false, .{ 2, 1, 0, 3 } });
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ 256, 16385, 2, 1 }, false, .{ 0, 2, 1, 3 } });

        // channel swap with sizes not multiples of 32/64 tile
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |type_dst| {
            try list.add(mod.cases.TestCont, .{ type_dst, .{ 33, 17, 5, 19 }, false, .{ 1, 2, 3, 0 } });
        }

        // small channel dims and extent-1 dims
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ 8, 104, 60, 13 }, false, .{ 1, 2, 3, 0 } });
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F16, .{ 8, 104, 60, 13 }, false, .{ 1, 2, 3, 0 } });
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ 16, 1, 50, 7 }, false, .{ 1, 2, 3, 0 } });

        // full reversal: contiguous src dim lands on dst dim 3
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ 33, 5, 7, 9 }, false, .{ 3, 2, 1, 0 } });
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ 33, 64, 32, 16 }, false, .{ 3, 2, 1, 0 } });

        // permutes that move dense blocks (ne[0] stays dim 0)
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |type_dst| {
            try list.add(mod.cases.TestCont, .{ type_dst, .{ 4, 65, 9, 2 }, false, .{ 0, 2, 1, 3 } });
        }
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F16, .{ 64, 9, 7, 1 }, false, .{ 0, 2, 1, 3 } });

        // transpose with i32 and CPY with reshape
        try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_I32, .{ 33, 17, 5, 19 }, false, .{ 1, 2, 3, 0 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 33, 17, 5, 1 }, .{ 17, 33, 5, 1 }, .{ 0, 0, 0, 0 }, .{ 1, 0, 2, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 8, 52, 39, 1 }, .{ 8, 39, 52, 1 }, .{ 0, 0, 0, 0 }, .{ 0, 2, 1, 3 } });

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_BF16 }) |t| {
            for ([_]bool{ false, true }) |perm1| {
                try addTestBinBcast(list, t, .{ 1, 1, 8, 1 }, .{ 1, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 1, 1, 1, 1 }, .{ 32, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 1, 1, 320, 320 }, .{ 1, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 1, 1 }, .{ 1, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 1 }, .{ 1, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 1, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 2, 1, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 1, 2, 1, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 1, 1, 2, 1 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 1, 1, 1, 2 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 1, 1, 2, 2 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 1, 2, 2, 2 }, perm1, false);
                try addTestBinBcast(list, t, .{ 10, 5, 4, 3 }, .{ 2, 2, 2, 2 }, perm1, false);
            }

            // src_overlap
            try addTestBinBcast(list, t, .{ 10, 5, 4, 6 }, .{ 1, 1, 1, 1 }, false, true);
            try addTestBinBcast(list, t, .{ 10, 5, 4, 5 }, .{ 1, 1, 1, 1 }, false, true);
            try addTestBinBcast(list, t, .{ 1, 1, 120, 120 }, .{ 1, 1, 1, 1 }, false, true);
            try addTestBinBcast(list, t, .{ 1, 1, 4, 320 }, .{ 1, 1, 1, 1 }, false, true);

            // test case for k_bin_bcast_unravel in CUDA backend
            try addTestBinBcast(list, t, .{ 1, 1, 65536, 1 }, .{ 256, 1, 1, 1 }, false, false);

            // stable diffusion
            try addTestBinBcast(list, t, .{ 1280, 1, 1, 1 }, .{ 1, 1, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1280, 1, 1, 1 }, .{ 1, 16, 16, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1280, 16, 16, 1 }, .{ 1, 1, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1280, 1, 1, 1 }, .{ 1, 256, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1, 1, 1280, 1 }, .{ 16, 16, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 16, 16, 1280, 1 }, .{ 1, 1, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1, 1, 1920, 1 }, .{ 16, 16, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1, 1, 2560, 1 }, .{ 16, 16, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1, 1, 1280, 1 }, .{ 32, 32, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1, 1, 1920, 1 }, .{ 32, 32, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 1, 1, 640, 1 }, .{ 32, 32, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 5120, 1, 1, 1 }, .{ 1, 256, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 640, 1, 1, 1 }, .{ 1, 1, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 64, 262144, 1, 1 }, .{ 1, 1, 1, 1 }, false, false);
            try addTestBinBcast(list, t, .{ 128, 1, 8, 1 }, .{ 1, 4, 1, 1 }, false, false);
            //try addTestBinBcast(list, t, .{ 3, 3, 2560, 1280 }, .{ 1, 1, 1, 1 }, false, false);
            //try addTestBinBcast(list, t, .{ 3, 3, 2560, 1280 }, .{ 2, 1, 1, 1 }, false, false);
        }

        for ([_]mod.cases.BinOp{ &mod.c.ggml_add, &mod.c.ggml_add_inplace }) |op| {
            for ([_]i64{ 1, 7, 33 }) |n| {
                try list.add(mod.cases.TestBinBcast, .{ op, mod.c.GGML_TYPE_F8_E4M3, .{ n, 5, 2, 1 }, .{ 1, 1, 1, 1 }, 1, false, false, mod.c.GGML_TYPE_F32 });
            }
        }

        for ([_]mod.cases.BinOp{ &mod.c.ggml_add, &mod.c.ggml_mul }) |op| {
            try list.add(mod.cases.TestBinBcast, .{ op, mod.c.GGML_TYPE_F32, .{ 1, 2, 2, 2 }, .{ 17, 1, 1, 1 }, 1, false, false, mod.c.GGML_TYPE_COUNT, true });
        }
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 2, 2, 2, 2 }, .{ 17, 1, 1, 1 }, 1, false, false, mod.c.GGML_TYPE_COUNT, true });

        // single inplace tests, especially important for WebGPU backend since kernels for inplace vs. not are different
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add_inplace, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 16 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_mul_inplace, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 16 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_sub_inplace, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 16 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_div_inplace, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 16 });

        // BF16 activations with F32 weights
        for ([_]mod.cases.BinOp{ &mod.c.ggml_add, &mod.c.ggml_sub, &mod.c.ggml_mul, &mod.c.ggml_div }) |op| {
            try list.add(mod.cases.TestBinBcast, .{ op, mod.c.GGML_TYPE_BF16, .{ 10, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 1, false, false, mod.c.GGML_TYPE_F32 });
            try list.add(mod.cases.TestBinBcast, .{ op, mod.c.GGML_TYPE_BF16, .{ 10, 5, 4, 3 }, .{ 2, 2, 2, 2 }, 1, false, false, mod.c.GGML_TYPE_F32 });
            try list.add(mod.cases.TestBinBcast, .{ op, mod.c.GGML_TYPE_BF16, .{ 5120, 1, 1, 1 }, .{ 1, 32, 1, 1 }, 1, false, false, mod.c.GGML_TYPE_F32 });
        }

        // fusion
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 10, 5, 4, 3 }, .{ 2, 1, 1, 1 }, 2 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F16, .{ 10, 5, 4, 3 }, .{ 2, 1, 1, 1 }, 2 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_BF16, .{ 10, 5, 4, 3 }, .{ 2, 1, 1, 1 }, 2 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_BF16, .{ 16, 5, 4, 3 }, .{ 1, 2, 1, 1 }, 3 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 2, true });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 2, 1, 1 }, 3 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 10, 5, 4, 3 }, .{ 1, 1, 2, 1 }, 4 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 2 }, 5 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 10, 5, 4, 3 }, .{ 1, 1, 2, 2 }, 6 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 10, 5, 4, 3 }, .{ 1, 2, 2, 2 }, 7 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 2, 2, 2, 2 }, 8 });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 16, 5, 4, 3 }, .{ 1, 1, 1, 1 }, 16 });

        try list.add(mod.cases.TestScale, .{});
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 10, 10 }, 2.0, 1.0 });
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 10, 10 }, 2.0, 1.0, true }); // inplace test
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_F32, .{ 100, 10, 10, 10 }, 2.0, 1.0 });
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_BF16, .{ 10, 10, 10, 10 }, 2.0, 1.0 });
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_BF16, .{ 10, 10, 10, 10 }, 2.0, 1.0, true }); // inplace test
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_BF16, .{ 100, 10, 10, 10 }, 0.5, 0.0 });
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 10, 10 }, 2.0, 0.0, false, 1 }); // view test
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 10, 10 }, 2.0, 1.0, false, 1 }); // view test
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 10, 10 }, 2.0, 1.0, true, 1 }); // view test, inplace
        try list.add(mod.cases.TestScale, .{ mod.c.GGML_TYPE_BF16, .{ 10, 10, 10, 10 }, 2.0, 1.0, true, 1 }); // view test, inplace
        try list.add(mod.cases.TestSoftcap, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 10, 10 }, 50.0 });
        try list.add(mod.cases.TestSiluBack, .{});

        for ([_]f32{ 0.0, 1e-6, 1e-4, 1e-1, 10.0 }) |eps| {
            for ([_]i64{ 64, 1025 }) |n| {
                for ([_]bool{ false, true }) |v| {
                    try list.add(mod.cases.TestNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, v, eps });
                    try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, v, eps });
                }
                try list.add(mod.cases.TestNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, false, eps, true });
                try list.add(mod.cases.TestNormScale, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, false, 1.5 });
                try list.add(mod.cases.TestNormScale, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, true, 1.5 });
                try list.add(mod.cases.TestRmsNormBack, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps });
                try list.add(mod.cases.TestL2Norm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, false });
                try list.add(mod.cases.TestL2Norm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, true });
                try list.add(mod.cases.TestL2Norm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, false, true });
                // sibling batching: strided (production shape) and contiguous, 2 and 4 wide
                try list.add(mod.cases.TestL2NormBatch, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, 2, eps, true });
                try list.add(mod.cases.TestL2NormBatch, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, 4, eps, true });
                try list.add(mod.cases.TestL2NormBatch, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, 4, eps, false });
            }
            // row lengths that are not a multiple of 32, for the scalar (33) and float4 (132, 260) paths
            for ([_]i64{ 33, 132, 260 }) |n| {
                for ([_]bool{ false, true }) |v| {
                    try list.add(mod.cases.TestNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, v, eps });
                    try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, v, eps });
                }
            }
        }

        // in-place tests
        try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ 64, 5, 4, 3 }, false, 1e-6, true });
        // shapes below exceed the CUDA gridDim.y/gridDim.z limit of 65535 (#27901)
        try list.add(mod.cases.TestNorm, .{ mod.c.GGML_TYPE_F32, .{ 4, 1, 65536, 1 }, false, 1e-6 });
        try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ 4, 1, 65536, 1 }, false, 1e-6, false });
        try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ 4, 1, 1, 65536 }, false, 1e-6, false });
        try list.add(mod.cases.TestL2Norm, .{ mod.c.GGML_TYPE_F32, .{ 4, 1, 65536, 1 }, 1e-12, false, false });
        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 4, 1, 65536, 1 }, 1e-6, false, false });

        for ([_]i64{ 1, 127, 129, 511, 513, 2816, 4096, 16385 }) |n| {
            try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 3, 2, 1 } });
            try list.add(mod.cases.TestRmsNormMulRope, .{ .{ n, 3, 2, 1 }, 1e-6, false, false, true, mod.c.GGML_ROPE_TYPE_NORMAL, true, false });
            try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ n, 3, 2, 1 }, 1e-6, false, false, false, false, true });
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |set_rows_type| {
            try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 256, 1, 1, 1 }, 1e-6, false, true, false, mod.c.GGML_ROPE_TYPE_NORMAL, false, false, set_rows_type });
            try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 4, 3, 1 }, 1e-6, false, true, false, mod.c.GGML_ROPE_TYPE_NORMAL, false, false, set_rows_type });
            for ([_]i32{ mod.c.GGML_ROPE_TYPE_NORMAL, mod.c.GGML_ROPE_TYPE_NEOX }) |mode| {
                for ([_]i64{ 64, 256, 512, 1024 }) |n| {
                    try list.add(mod.cases.TestRmsNormMulRope, .{ .{ n, 4, 3, 1 }, 1e-6, false, true, true, mode, true, true, set_rows_type });
                }
            }
        }

        for ([_]f32{ 0.0, 1e-6, 1e-4, 1e-1, 1.0 }) |eps| {
            for ([_]i64{ 64, 1025 }) |n| {
                try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, false });
                try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, true });
                try list.add(mod.cases.TestNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, false });
                try list.add(mod.cases.TestNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, true });
                try list.add(mod.cases.TestAddRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, false });
                try list.add(mod.cases.TestAddRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, eps, true });
            }
        }
        for ([_]i64{ 1, 511, 1025, 8192, 33 * 512 }) |n| {
            for ([_]bool{ false, true }) |multi_add| {
                try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ n, 1, 1, 1 }, 1e-6, false, multi_add });
            }
            try list.add(mod.cases.TestAddRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ n, 1, 1, 1 }, 1e-6, false });
        }
        for ([_]i64{ 64, 1025 }) |n| {
            try list.add(mod.cases.TestAddAdd, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, false, false });
            try list.add(mod.cases.TestAddAdd, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, true, false });
            try list.add(mod.cases.TestAddAdd, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, false, true });
            try list.add(mod.cases.TestAddAdd, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ n, 5, 4, 3 }, false, false });
            try list.add(mod.cases.TestAddAdd, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, false, false });
            try list.add(mod.cases.TestAddAdd, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, .{ n, 5, 4, 3 }, true, false });
        }

        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 1536, 1, 1, 1 }, 1e-6, false, false, true });
        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 256, 4, 1, 1 }, 1e-6, false, false, true });
        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 256, 4, 3, 2 }, 1e-6, false, false, true });
        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 256, 4, 3, 2 }, 1e-6, false, false, true, false, true });
        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 1536, 1, 1, 1 }, 1e-6, false, false, false, true });
        try list.add(mod.cases.TestRmsNormMulAdd, .{ mod.c.GGML_TYPE_F32, .{ 256, 4, 1, 1 }, 1e-6, false, false, false, true });

        try list.add(mod.cases.TestRmsNormMulRope, .{.{ 128, 4, 7, 2 }});
        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 4, 7, 2 }, 1e-6, false, true });

        for ([_]bool{ false, true }) |multi_add| {
            for ([_]bool{ false, true }) |set_rows| {
                for ([_]bool{ false, true }) |broadcast| {
                    for ([_]i32{ mod.c.GGML_ROPE_TYPE_NORMAL, mod.c.GGML_ROPE_TYPE_NEOX, mod.c.GGML_ROPE_TYPE_IMROPE }) |rope| {
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 768, 1, 1, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 768, 3, 1, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 768, 3, 5, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 32, 2, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 4, 2, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 32, 50, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 4, 50, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 8192, 2, 2, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 8192, 2, 2, 1 }, 1e-6, multi_add, set_rows, broadcast, rope });
                    }
                }
            }
        }
        // ne[2] > 65535 exceeds the CUDA gridDim.y limit (#27901)
        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 4, 1, 65536, 1 }, 1e-6, false, false, false, mod.c.GGML_ROPE_TYPE_NORMAL });
        try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 4, 1, 65536, 1 }, 1e-6, false, true, false, mod.c.GGML_ROPE_TYPE_NEOX });
        for ([_]i64{ 3, 4, 7, 9 }) |d_conv| {
            for ([_]i64{ 1024, 1536, 2048 }) |d_inner| {
                try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ d_conv, d_inner, 1, 1 }, .{ d_conv, d_inner, 1, 1 } });
                try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ 2 * d_conv, d_inner, 1, 1 }, .{ d_conv, d_inner, 1, 1 } });
                try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ d_conv, d_inner, 4, 1 }, .{ d_conv, d_inner, 1, 1 } });
                // long token (n_t > 32, exercises the long_token kernel path)
                try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ d_conv - 1 + 64, d_inner, 1, 1 }, .{ d_conv, d_inner, 1, 1 } });
                try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ d_conv - 1 + 64, d_inner, 4, 1 }, .{ d_conv, d_inner, 1, 1 } });
            }
        }

        // fused ssm_conv + (optional) bias_add + silu. The bias-only graph (no silu) is intentionally
        // not tested since there's no fusion for that pattern in ggml_cuda_can_fuse.
        for ([_]i64{ 3, 4, 7, 9 }) |d_conv| {
            for ([_]i64{ 1024, 1536, 2048 }) |d_inner| {
                for ([_]bool{ false, true }) |fuse_bias| {
                    // short token path (n_t <= 32)
                    try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ d_conv, d_inner, 1, 1 }, .{ d_conv, d_inner, 1, 1 }, fuse_bias });
                    try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ 2 * d_conv, d_inner, 1, 1 }, .{ d_conv, d_inner, 1, 1 }, fuse_bias });
                    try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ d_conv, d_inner, 4, 1 }, .{ d_conv, d_inner, 1, 1 }, fuse_bias });
                    // long token path (n_t > 32)
                    try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ d_conv - 1 + 64, d_inner, 1, 1 }, .{ d_conv, d_inner, 1, 1 }, fuse_bias });
                    try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ d_conv - 1 + 64, d_inner, 4, 1 }, .{ d_conv, d_inner, 1, 1 }, fuse_bias });
                }
            }
        }

        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 16, 1, 1024, 1, 32, 4 }); // Mamba-1
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 96, 64, 128, 8, 1, 1 }); // Nemotron-3-Puzzle decode (scan path, unused warp in last block)
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 96, 64, 16, 8, 300, 2, false, 1, true }); // d_state=96 SSD multi-chunk (partial 2nd chunk, 2 seqs)
        try list.add(mod.cases.TestSsmScanRollback, .{ mod.c.GGML_TYPE_F32, 96, 64, 16, 8, 8, 2, 3 }); // d_state=96 rollback snapshots match prefix states
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 32, 4 }); // Mamba-2
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 256, 64, 8, 2, 32, 4 }); // Falcon-H1
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 128, 4, 4, 16, 2, true }); // x/B/C overlap
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 80, 128, 1, 256, 1 }); // Nemotron-9B SSD path
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 80, 128, 1, 512, 1 }); // Nemotron-9B SSD multi-chunk (2 aligned chunks)
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 80, 8, 300, 2 }); // Mamba-2 SSD multi-chunk (partial 2nd chunk, 2 seqs)
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 4, 2, false, 4 }); // Mamba-2 rollback snapshots
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 8, 2, false, 3 }); // Mamba-2 rollback overflow
        try list.add(mod.cases.TestSsmScanRollback, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 8, 2, 3 }); // rollback snapshots match prefix states
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 64, 4 }); // Metal SSD one chunk MMA only, no seq tail
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 65, 2 }); // SSD one chunk + 1-token sequential tail
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 128, 2 }); // SSD multi-chunk, no tail (chunk-to-chunk state handoff)
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 128, 2, false, 1, true }); // SSD multi-chunk, carried state not negligible

        // ssm_scan + cache cpy fusion
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 4, 1, 4 });
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 1, 1, 4 }); // n_seq_tokens < K
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 8, 1, 3 }); // n_seq_tokens > K
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 96, 64, 16, 2, 4, 1, 4 });
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 256, 64, 8, 2, 4, 1, 4 });
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 1, 1, 1 }); // K == 1, final state only
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 4, 1, 1 });
        try list.add(mod.cases.TestSsmScanCacheFusion, .{ mod.c.GGML_TYPE_F32, 128, 64, 16, 2, 300, 1, 1 }); // K == 1, SSD path over two chunks

        try list.add(mod.cases.TestRwkvWkv6, .{ mod.c.GGML_TYPE_F32, 32, 64, 1, 1 });
        try list.add(mod.cases.TestRwkvWkv6, .{ mod.c.GGML_TYPE_F32, 32, 64, 32, 1 });
        try list.add(mod.cases.TestRwkvWkv6, .{ mod.c.GGML_TYPE_F32, 32, 64, 32, 4 });
        try list.add(mod.cases.TestRwkvWkv6, .{ mod.c.GGML_TYPE_F32, 32, 64, 128, 4 });

        try list.add(mod.cases.TestRwkvWkv7, .{ mod.c.GGML_TYPE_F32, 32, 64, 1, 1 });
        try list.add(mod.cases.TestRwkvWkv7, .{ mod.c.GGML_TYPE_F32, 32, 64, 1, 4 });
        try list.add(mod.cases.TestRwkvWkv7, .{ mod.c.GGML_TYPE_F32, 32, 64, 32, 1 });
        try list.add(mod.cases.TestRwkvWkv7, .{ mod.c.GGML_TYPE_F32, 32, 64, 32, 4 });
        try list.add(mod.cases.TestRwkvWkv7, .{ mod.c.GGML_TYPE_F32, 32, 64, 128, 4 });

        try list.add(mod.cases.TestGla, .{ mod.c.GGML_TYPE_F32, 32, 64, 1, 1 });
        try list.add(mod.cases.TestGla, .{ mod.c.GGML_TYPE_F32, 32, 64, 32, 1 });
        try list.add(mod.cases.TestGla, .{ mod.c.GGML_TYPE_F32, 32, 64, 32, 4 });
        try list.add(mod.cases.TestGla, .{ mod.c.GGML_TYPE_F32, 32, 64, 128, 4 });

        // FWHT tests
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 128, 1, 128 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 64, 1, 64 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 256, 1, 256 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 512, 1, 512 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 128, 32, 128 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 128, 4, 128, .{ 2, 3 } });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 256, 512, 256 }); // many rows
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 64, 1, 64 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 128, 1, 128 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 256, 1, 256 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 512, 1, 512 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 128, 32, 128 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 256, 512, 256 }); // many rows
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 32, 1, 32 }); // too small (N<64)
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 16384, 1, 16384 }); // too big (N>8192)
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 64, 1, 64 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 128, 1, 128 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 256, 1, 256 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 512, 1, 512 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 128, 32, 128 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 128, 4, 128, .{ 2, 3 } });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 256, 512, 256 }); // many rows
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 1024, 1, 1024 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 2048, 1, 2048 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 4096, 1, 4096 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 8192, 1, 8192 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 1024, 7, 1024 }); // many rows
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 1024, 1, 1024 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 2048, 1, 2048 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 4096, 1, 4096 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 8192, 1, 8192 });
        try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, 1024, 7, 1024 }); // many rows

        // FP4 activation precision (default = native W4A4, src1 GGML_PREC_Q8 = W4A8)
        try list.add(mod.cases.TestMulMatW4a8, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 32, 1, 256 });
        try list.add(mod.cases.TestMulMatW4a8, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatW4a8, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 64, 16, 512 });
        try list.add(mod.cases.TestMulMatW4a4, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 32, 1, 256 });
        try list.add(mod.cases.TestMulMatW4a4, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatIdW4a8, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 8, 2, false, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatIdW4a8, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 4, 2, true, 64, 16, 256 });
        try list.add(mod.cases.TestMulMatIdW4a4, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 8, 2, false, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatIdW4a4, .{ mod.c.GGML_TYPE_NVFP4, mod.c.GGML_TYPE_F32, 4, 2, true, 64, 16, 256 });
        try list.add(mod.cases.TestMulMatW4a8, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatW4a8, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 64, 16, 512 });
        try list.add(mod.cases.TestMulMatW4a4, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatIdW4a8, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 8, 2, false, 32, 32, 256 });
        try list.add(mod.cases.TestMulMatIdW4a4, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 8, 2, false, 32, 32, 256 });

        // (upstream `#if 0`: > 2^31-element dst and > 4GB A matrix mul_mat / mul_mat_id cases, too large to run by default.)

        for (mod.TypeLists.all_types) |type_a| {
            var i: i64 = 1;
            while (i < 10) : (i += 1) {
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 16, i, 1 * 256, .{ 1, 1 }, .{ 1, 1 } });
                // (upstream keeps six more commented-out shapes here.)
            }
            // mat-vec shaders split k across lanes and loop over the blocks in strides. k must be
            // long enough that the loop wraps, else the stride is never exercised
            try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 16, 1, 16 * 256, .{ 1, 1 }, .{ 1, 1 } });
            try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 16, 8, 16 * 256, .{ 1, 1 }, .{ 1, 1 } });
        }

        // Multi-column MMVQ coverage for the Q4_K weight-reuse path and a Q5_K control.
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q5_K }) |type_a| {
            var n: i64 = 1;
            while (n <= 8) : (n += 1) {
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 4096, n, 1024, .{ 1, 1 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 1023, n, 4096, .{ 1, 1 }, .{ 1, 1 } });
            }
        }

        // The SYCL backend picks between one and two output rows per subgroup by row count when there
        // are two destination columns (Q4_K_MMVQ_ROW_PAIR_MIN_NROWS in ggml-sycl/mmvq.cpp). Cover both
        // sides of that boundary, including an odd row count above it for the row-pair tail.
        for ([_]i64{ 6271, 6272, 6273 }) |m| {
            try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_F32, m, 2, 1024, .{ 1, 1 }, .{ 1, 1 } });
        }

        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_F32, 2880, 32, 2880, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_F32, 2880, 32, 2880, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 2880, 32, 2880, .{ 1, 1 }, .{ 1, 1 } });

        // m == 1, with n on both sides of MMVF_MAX_BATCH_SIZE (8): mmvf below, operand swap above
        for ([_]i64{ 1, 7, 8, 9, 16, 127, 128, 511, 512 }) |n| {
            try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 1, n, 2048, .{ 1, 1 }, .{ 1, 1 } });
        }
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, 1, 512, 2048, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, 1, 512, 2048, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, 1, 509, 2051, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, 1, 509, 2051, .{ 1, 1 }, .{ 1, 1 } });

        // testing very large N
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, 16, 2105344, 32, .{ 1, 1 }, .{ 1, 1 } });

        // (upstream `#if 0`: N = 8400000 shapes.)

        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 31, 509, 2051, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 32, 509, 2112, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_F32, 32, 509, 2112, .{ 1, 1 }, .{ 1, 1 } });

        // few src1 rows (speculative verify): odd m, a single K block, long K, every tile width, broadcast batches
        for ([_]i64{ 2, 3, 5, 8, 9, 13, 16 }) |n| {
            for ([_][2]i64{ .{ 40, 32 }, .{ 100, 96 }, .{ 1000, 5120 }, .{ 3000, 1024 }, .{ 6144, 5120 }, .{ 17408, 512 } }) |mk| {
                try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_F32, mk[0], n, mk[1], .{ 1, 1 }, .{ 1, 1 } });
            }
            try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_F32, 64, n, 256, .{ 3, 2 }, .{ 2, 1 } });
            for ([_]mod.c.enum_ggml_type{
                mod.c.GGML_TYPE_F32,    mod.c.GGML_TYPE_F16,     mod.c.GGML_TYPE_BF16,  mod.c.GGML_TYPE_Q4_1,  mod.c.GGML_TYPE_Q5_0,    mod.c.GGML_TYPE_Q5_1,
                mod.c.GGML_TYPE_Q8_0,   mod.c.GGML_TYPE_Q1_0,    mod.c.GGML_TYPE_Q2_0,  mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_Q2_K,    mod.c.GGML_TYPE_Q3_K,
                mod.c.GGML_TYPE_Q4_K,   mod.c.GGML_TYPE_Q5_K,    mod.c.GGML_TYPE_Q6_K,  mod.c.GGML_TYPE_TQ2_0, mod.c.GGML_TYPE_IQ2_XXS, mod.c.GGML_TYPE_IQ2_XS,
                mod.c.GGML_TYPE_IQ2_S,  mod.c.GGML_TYPE_IQ3_XXS, mod.c.GGML_TYPE_IQ3_S, mod.c.GGML_TYPE_IQ1_S, mod.c.GGML_TYPE_IQ1_M,   mod.c.GGML_TYPE_IQ4_NL,
                mod.c.GGML_TYPE_IQ4_XS,
            }) |type_a| {
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 48, n, 2560, .{ 1, 1 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 1000, n, 1024, .{ 1, 1 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 3000, n, 512, .{ 1, 1 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 4100, n, 256, .{ 1, 1 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 64, n, 256, .{ 3, 2 }, .{ 2, 1 } });
                // a K that is not a multiple of the 64-weight chunk takes the other kernels
                if (@mod(96, mod.c.ggml_blck_size(type_a)) == 0) {
                    try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 100, n, 96, .{ 1, 1 }, .{ 1, 1 } });
                }
            }
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q5_K, mod.c.GGML_TYPE_F16 }) |type_a| {
                for ([_]@FieldType(mod.cases.TestMulMatAdd, "mode"){ .mmRes, .resMm, .row, .resInplace, .mmMm }) |mode| {
                    try list.add(mod.cases.TestMulMatAdd, .{ type_a, 1000, n, 1024, mode });
                }
                try list.add(mod.cases.TestMulMatAdd, .{ type_a, 2048, n, 2048, .bInplace });
            }
            // the widest tiles with two src1 tiles, and src1 rows padded in memory
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q5_K, mod.c.GGML_TYPE_Q6_K }) |type_a| {
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 8192, n, 512, .{ 1, 1 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 1000, n, 1024, .{ 1, 1 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 1280 });
            }
            // a src1 with a nonzero mean, for the zero points and mins of quantized src0 types
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_1, mod.c.GGML_TYPE_Q5_0, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q2_K, mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q5_K }) |type_a| {
                try list.add(mod.cases.TestMulMatPos, .{ type_a, mod.c.GGML_TYPE_F32, 1000, n, 1024, .{ 1, 1 }, .{ 1, 1 } });
            }
        }

        // (upstream `#if 0`: OpenCL Q8_0 paths.)

        for (mod.TypeLists.base_types) |type_a| {
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |type_b| {
                var ks_buf = [2]i64{ 256, 4 };
                const ks: []const i64 = if (mod.c.ggml_blck_size(type_a) == 1) ks_buf[0..2] else ks_buf[0..1];
                for (ks) |k| {
                    // test cases without permutation
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 1, 1 }, .{ 1, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 1, 1 }, .{ 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 1, 1 }, .{ 1, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 3, 1 }, .{ 1, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 3, 1 }, .{ 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 3, 2 }, .{ 1, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 3, 2 }, .{ 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 3, 2 }, .{ 1, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 3, 2 }, .{ 2, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 4, k, .{ 3, 2 }, .{ 2, 2 } });

                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 1, 1 }, .{ 1, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 1, 1 }, .{ 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 1, 1 }, .{ 1, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 3, 1 }, .{ 1, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 3, 1 }, .{ 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 3, 2 }, .{ 1, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 3, 2 }, .{ 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 3, 2 }, .{ 1, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 3, 2 }, .{ 2, 2 } });

                    // test cases with permutation
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 2, 1, 3 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 1, 3, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 3, 2, 1 } });

                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 4, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 3, 2, 1 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 2, 1, 3 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 1, 3, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 3, 2, 1 } });

                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 2, 1, 3 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 1, 3, 2 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, k, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 3, 2, 1 } });

                    // test cases with permutation and a broadcast of a in dim 3 (e.g. a weight against a per-stream attention output)
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, k, .{ 1, 1 }, .{ 1, 2 }, .{ 0, 2, 1, 3 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, k, .{ 1, 1 }, .{ 1, 2 }, .{ 0, 2, 1, 3 } });
                    try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, k, .{ 1, 1 }, .{ 4, 2 }, .{ 0, 2, 1, 3 } });
                }

                // test cases with large ne00/ne10 to cover stream-k fixup
                try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 1, 1024, .{ 3, 2 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, 1024, .{ 3, 2 }, .{ 1, 1 } });
                try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 16, 1024, .{ 3, 2 }, .{ 1, 1 } });

                // test cases with large batch size
                try list.add(mod.cases.TestMulMat, .{ type_a, type_b, 16, 8, 256, .{ 1536, 1 }, .{ 1, 1 } });
            }
        }
    }

    // Upstream's `calc_conv_output_size` (and `calc_conv_output_size_3d`, the same formula):
    // the output length of a convolution, with C++ integer division.
    fn calcConvOutputSize(ins: i64, ks: i64, s: i64, p: i64, d: i64) i64 {
        return @divTrunc(ins + 2 * p - d * (ks - 1) - 1, s) + 1;
    }

    // Upstream's `add_test_bin_bcast`: one `TestBinBcast` per add, sub, mul and div.
    fn addTestBinBcast(list: *mod.CaseList, t: mod.c.enum_ggml_type, ne: [4]i64, nr: [4]i32, perm1: bool, src_overlap: bool) !void {
        for ([_]mod.cases.BinOp{ &mod.c.ggml_add, &mod.c.ggml_sub, &mod.c.ggml_mul, &mod.c.ggml_div }) |op| {
            try list.add(mod.cases.TestBinBcast, .{ op, t, ne, nr, 1, perm1, src_overlap });
        }
    }

    // `std::next_permutation` on four elements; wraps the last permutation to the first.
    fn nextPermutation(a: *[4]i64) void {
        var i: usize = a.len - 1;
        while (i > 0 and a[i - 1] >= a[i]) i -= 1;
        if (i == 0) {
            std.mem.reverse(i64, a);
            return;
        }
        var j: usize = a.len - 1;
        while (a[j] <= a[i - 1]) j -= 1;
        std.mem.swap(i64, &a[i - 1], &a[j]);
        std.mem.reverse(i64, a[i..]);
    }
};
