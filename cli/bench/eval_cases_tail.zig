//! Port of `make_test_cases_eval`, lines 11501-12772 (tests/test-backend-ops.cpp @cd959bae9).

const mod = @import("module.zig");

/// The tail half of the eval case list.
pub const EvalCasesTail = struct {
    const Self = @This();

    /// Adds this half's cases to `list`, in upstream's order.
    ///
    /// Upstream's `rng` (`std::default_random_engine rng(0)`) only feeds code
    /// compiled out in this half (the `#else` of `#if 1` and an `#if 0`
    /// soft_max sweep), so no PRNG is needed here.
    ///
    /// - `list`: the destination.
    pub fn build(list: *mod.CaseList) !void {
        try Self.addMulMat(list);
        try Self.addMulMatId(list);
        try Self.addOutProdAddId(list);
        try Self.addUnary(list);
        try Self.addSoftMax(list);
        try Self.addRope(list);
        try Self.addConcat(list);
        try Self.addSortTopK(list);
        try Self.addMisc(list);
        try Self.addFlashAttn(list);
        try Self.addFusions(list);
        try Self.addDeltaNetIndexer(list);
    }

    // The tail of the mul_mat section, from the BF16 permutations on.
    fn addMulMat(list: *mod.CaseList) !void {
        const T = mod.cases.TestMulMat;
        const type_bf16 = mod.c.GGML_TYPE_BF16;
        const type_f16 = mod.c.GGML_TYPE_F16;
        const type_f32 = mod.c.GGML_TYPE_F32;

        // BF16 is absent from base_types: add the 3 standard non-contig permutations explicitly
        for ([_]i64{ 1, 8, 16 }) |n| {
            try list.add(T, .{ type_bf16, type_f32, 16, n, 256, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 2, 1, 3 } });
            try list.add(T, .{ type_bf16, type_f32, 16, n, 256, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 1, 3, 2 } });
            try list.add(T, .{ type_bf16, type_f32, 16, n, 256, .{ 2, 3 }, .{ 1, 1 }, .{ 0, 3, 2, 1 } });
        }

        // BF16 in src1, as ggml_conv_1d_dw emits it for a BF16 kernel
        try list.add(T, .{ type_f32, type_bf16, 16, 1, 256, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f32, type_bf16, 16, 1, 256, .{ 3, 2 }, .{ 2, 2 } });
        try list.add(T, .{ type_f32, type_bf16, 16, 8, 256, .{ 1, 1 }, .{ 1, 1 } });

        // token-tile boundary coverage. With n_used == n_mats every token routes to every expert,
        // so each expert receives exactly n rows. mul_mm_id is used from 32 tokens up: n = 32..49
        // leave a last tile of 32, 1, 15, 16 and 17 rows; smaller n cover the mat-vec path.
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_IQ2_XS, type_f16 }) |type_a| {
            for ([_]i64{ 1, 15, 16, 17, 31, 32, 33, 47, 48, 49 }) |n| {
                try list.add(mod.cases.TestMulMatId, .{ type_a, type_f32, 4, 4, false, 512, n, 256 });
            }
            // experts that receive no rows at all
            try list.add(mod.cases.TestMulMatId, .{ type_a, type_f32, 8, 1, false, 512, 1, 256 });
        }

        for (mod.TypeLists.other_types) |type_a| {
            if (mod.c.ggml_blck_size(type_a) != 256) {
                try list.add(T, .{ type_a, type_f32, 16, 1, mod.c.ggml_blck_size(type_a), .{ 1, 1 }, .{ 1, 1 } });
            }
            try list.add(T, .{ type_a, type_f32, 16, 1, 256, .{ 1, 1 }, .{ 1, 1 } });
        }

        // Test IQP panel path for all grid IQ types
        for (iq_grid_types) |type_a| {
            try list.add(T, .{ type_a, type_f32, 16, 10, 256, .{ 1, 1 }, .{ 1, 1 } });
        }
        // (upstream's `#else` branch, a random m/n/k sweep driven by `rng`, is compiled out)

        try list.add(T, .{ type_f16, type_f32, 64, 2, 128, .{ 8, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f16, type_f32, 83, 2, 128, .{ 8, 1 }, .{ 4, 1 } });
        try list.add(T, .{ type_f16, type_f32, 64, 2, 64, .{ 8, 1 }, .{ 4, 1 } });
        try list.add(T, .{ type_f16, type_f32, 83, 2, 64, .{ 8, 1 }, .{ 4, 1 } });
        try list.add(T, .{ type_f16, type_f32, 64, 45, 128, .{ 8, 1 }, .{ 4, 1 } });
        try list.add(T, .{ type_f16, type_f32, 128, 45, 64, .{ 8, 1 }, .{ 4, 1 } });
        try list.add(T, .{ type_f16, type_f32, 1056, 1, 193, .{ 1, 1 }, .{ 4, 1 }, .{ 0, 2, 1, 3 } });
        try list.add(T, .{ type_f16, type_f32, 1056, 1, 67, .{ 1, 1 }, .{ 4, 1 }, .{ 0, 2, 1, 3 } });
        try list.add(T, .{ type_f32, type_f32, 16, 32, 32, .{ 1, 1 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 64, 3 });
        try list.add(T, .{ type_f32, type_f32, 64, 77, 77, .{ 12, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f32, type_f32, 32, 4, 96, .{ 3, 2 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 0, 1, true });
        // the first rows of a KV cache: a is a view whose batches are strided by the cache length
        try list.add(T, .{ type_f32, type_f32, 8, 1, 64, .{ 8, 1 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 0, 1, false, 5120 });
        try list.add(T, .{ type_f16, type_f32, 8, 16, 64, .{ 8, 1 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 0, 1, false, 5120 });
        // the batches of a are padded, nb[2] is not a multiple of nb[1]
        try list.add(T, .{ type_f32, type_f32, 8, 1, 64, .{ 8, 1 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 0, 1, false, 16, 16 });
        try list.add(T, .{ type_f16, type_f32, 8, 16, 64, .{ 8, 1 }, .{ 1, 1 }, .{ 0, 1, 2, 3 }, 0, 1, false, 16, 16 });
        // as is a view whose experts are strided by more rows than it uses
        try list.add(mod.cases.TestMulMatId, .{ type_f32, type_f32, 4, 2, false, 8, 1, 64, 64 });
        try list.add(mod.cases.TestMulMatId, .{ type_f16, type_f32, 4, 2, false, 8, 16, 64, 64 });

        try list.add(T, .{ mod.c.GGML_TYPE_Q4_0, type_f32, 576, 512, 576, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(T, .{ mod.c.GGML_TYPE_Q4_0, type_f32, 1, 2048, 8192, .{ 1, 1 }, .{ 1, 1 } });
        for (mod.TypeLists.all_types) |type_a| {
            try list.add(T, .{ type_a, type_f32, 1, 64, 256, .{ 1, 1 }, .{ 1, 1 } });
        }

        try list.add(T, .{ mod.c.GGML_TYPE_Q8_0, type_f32, 6, 4096, 5120, .{ 1, 1 }, .{ 1, 1 } });

        // SIMD K boundaries, including tails shorter than one vector.
        for ([_]mod.c.enum_ggml_type{ type_f32, type_f16, type_bf16 }) |t| {
            for ([_]i64{ 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48 }) |k| {
                try list.add(T, .{ t, t, 64, 17, k, .{ 1, 1 }, .{ 1, 1 } });
                if (t != type_f32) {
                    try list.add(T, .{ t, type_f32, 64, 17, k, .{ 1, 1 }, .{ 1, 1 } });
                }
            }
        }
        try list.add(T, .{ type_f16, type_f16, 64, 32, 65, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f16, type_f16, 64, 32, 80, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f16, type_f32, 64, 32, 80, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f32, type_f32, 64, 32, 80, .{ 1, 1 }, .{ 1, 1 } });
        try list.add(T, .{ type_f16, type_f16, 64, 32, 588, .{ 1, 1 }, .{ 1, 1 } }); // 14*14*3, e.g. conv_2d im2col
        try list.add(T, .{ type_f16, type_f16, 64, 32, 80, .{ 4, 1 }, .{ 1, 1 } });

        // (upstream's `#if 0` mat-mat path sweep for Metal is compiled out)

        for ([_]i64{ 1, 3 }) |bs2| {
            for ([_]i64{ 1, 2, 4, 8 }) |bs| {
                for ([_]i64{ 1, 4 }) |nr| {
                    var m: i64 = 0;
                    while (m < 2) : (m += 1) {
                        var k: i64 = 0;
                        while (k < 2) : (k += 1) {
                            for ([_]mod.c.enum_ggml_type{ type_f16, type_bf16, type_f32 }) |t| {
                                try list.add(T, .{ t, type_f32, 1056 + m, 1, 128 + k, .{ bs, bs2 }, .{ nr, 1 }, .{ 0, 2, 1, 3 } });
                                try list.add(T, .{ t, type_f32, 128 + m, 1, 1056 + k, .{ bs, bs2 }, .{ nr, 1 }, .{ 0, 1, 2, 3 }, 2 * 1056 + k });
                            }
                        }
                    }
                }
            }
        }

        // sycl's large-k type_f16 -> f32 convert case (512x262144x9216) stays commented out upstream
    }

    // The grid IQ types of the IQP panel path cases.
    const iq_grid_types = [_]mod.c.enum_ggml_type{
        mod.c.GGML_TYPE_IQ2_XXS, mod.c.GGML_TYPE_IQ2_XS, mod.c.GGML_TYPE_IQ2_S, mod.c.GGML_TYPE_IQ3_XXS,
        mod.c.GGML_TYPE_IQ3_S,   mod.c.GGML_TYPE_IQ1_S,  mod.c.GGML_TYPE_IQ1_M, mod.c.GGML_TYPE_IQ4_XS,
    };

    // The mul_mat_id section.
    fn addMulMatId(list: *mod.CaseList) !void {
        const T = mod.cases.TestMulMatId;
        const type_f16 = mod.c.GGML_TYPE_F16;
        const type_f32 = mod.c.GGML_TYPE_F32;
        const q4_0 = mod.c.GGML_TYPE_Q4_0;
        const iq3_s = mod.c.GGML_TYPE_IQ3_S;

        // test large experts*tokens
        for ([_]bool{ false, true }) |b| {
            try list.add(T, .{ type_f16, type_f32, 16, 16, b, 32, 1024, 16 });
            try list.add(T, .{ type_f16, type_f32, 2, 2, b, 32, 8192, 64 });
            try list.add(T, .{ type_f16, type_f32, 16, 16, b, 50, 200, 64 });
            try list.add(T, .{ mod.c.GGML_TYPE_BF16, type_f32, 16, 16, b, 32, 1024, 16 });
            try list.add(T, .{ mod.c.GGML_TYPE_BF16, type_f32, 16, 16, b, 50, 200, 64 });
        }

        // For issue 27873
        try list.add(T, .{ mod.c.GGML_TYPE_IQ2_XXS, type_f32, 1, 1, false, 1, 8192, 4096 });

        for ([_]i64{ 1, 63, 65 }) |k| {
            try list.add(T, .{ type_f16, type_f32, 1, 1, false, 8, 16, k });
        }
        try list.add(mod.cases.TestMulMatIdFusion, .{ type_f16, type_f32, 16, 16, false, 32, 32, 32, 3 });
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q8_0 }) |t| {
            for ([_]i64{ 1, 2, 3, 4, 8, 9 }) |n| {
                try list.add(mod.cases.TestMulMatIdShared, .{ t, n });
            }
        }

        // gpt-oss issue with Vulkan mmq_id
        try list.add(T, .{ mod.c.GGML_TYPE_MXFP4, type_f32, 32, 2, false, 2880, 32, 2880 });
        // more than 256 experts (hoisted row-id path): 512 as in Qwen3.8-Flash-Next,
        // and 1024 at the LLAMA_MAX_EXPERTS limit
        for ([_]i64{ 1, 5, 64, 300 }) |n| {
            try list.add(T, .{ iq3_s, type_f32, 512, 10, false, 128, n, 512 });
            try list.add(T, .{ q4_0, type_f32, 512, 10, false, 256, n, 128 });
            try list.add(T, .{ iq3_s, type_f32, 1024, 10, false, 128, n, 512 });
            try list.add(T, .{ q4_0, type_f32, 1024, 10, false, 256, n, 128 });
        }
        try list.add(T, .{ q4_0, type_f32, 32, 2, false, 2880, 32, 2880 });

        // multiple blocks per row: exercises the block-stride loop and the
        // per-expert base offset, which k == 256 alone leaves untested
        try list.add(T, .{ mod.c.GGML_TYPE_TQ1_0, type_f32, 28, 10, false, 1024, 1, 4096 });
        try list.add(T, .{ mod.c.GGML_TYPE_TQ1_0, type_f32, 128, 8, false, 1024, 1, 2048 });

        // (upstream's `#if 0` repeated-expert-id cases, test_mul_mat_id_dup, are disabled
        // until CUDA support is added [TAG_MUL_MAT_ID_DUPLICATE_ID])

        for (mod.TypeLists.all_types) |type_a| {
            try list.add(T, .{ type_a, type_f32, 4, 2, false, 64, 16, 3 * mod.c.ggml_blck_size(type_a) });
        }

        // Test IQP panel path for all grid IQ types
        for (iq_grid_types) |type_a| {
            try list.add(T, .{ type_a, type_f32, 4, 4, false, 16, 10, 256 });
        }

        for (mod.TypeLists.base_types) |type_a| {
            for ([_]i32{ 4, 8 }) |n_mats| {
                for ([_]i32{ 1, 2, 4 }) |n_used| {
                    for ([_]bool{ false, true }) |b| {
                        for ([_]i64{ 1, 4, 5, 17, 32, 129 }) |n| {
                            try list.add(T, .{ type_a, type_f32, n_mats, n_used, b, 512, n, 256 });
                        }
                    }
                }
            }
        }

        for (mod.TypeLists.other_types) |type_a| {
            for ([_]i64{ 1, 32 }) |n| {
                try list.add(T, .{ type_a, type_f32, 4, 2, false, 512, n, 256 });
            }
        }

        // skipped slots (negative ids)
        for ([_]mod.c.enum_ggml_type{ type_f32, type_f16, mod.c.GGML_TYPE_BF16, q4_0, mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_MXFP4 }) |type_a| {
            for ([_]i32{ 1, 2, 4 }) |n_used| {
                for ([_]bool{ false, true }) |b| {
                    for ([_]i64{ 1, 4, 33, 129 }) |n| {
                        try list.add(T, .{ type_a, type_f32, 8, n_used, b, 512, n, 256, 0, false, true });
                    }
                }
            }
        }
        // n_used == n_mats, so every expert can be skipped
        try list.add(T, .{ type_f16, type_f32, 4, 4, false, 512, 64, 256, 0, false, true });
        try list.add(T, .{ type_f16, type_f32, 1, 1, false, 8, 16, 1, 0, false, true });
        try list.add(mod.cases.TestMulMatIdFusion, .{ type_f16, type_f32, 16, 16, false, 32, 32, 32, 3, false, true });

        for ([_]i64{ 1, 4, 512 }) |bs| {
            for ([_]mod.c.enum_ggml_type{ type_f32, type_f16, q4_0, mod.c.GGML_TYPE_Q4_K }) |type_a| {
                // test with mul after (ffn_moe_weighted)
                try list.add(mod.cases.TestMulMatIdFusion, .{ type_a, type_f32, 128, 8, false, 768, bs, 2048, 1, true });
            }
        }
    }

    // out_prod and add_id.
    fn addOutProdAddId(list: *mod.CaseList) !void {
        const type_f32 = mod.c.GGML_TYPE_F32;

        for (mod.TypeLists.base_types) |type_a| {
            for ([_]mod.c.enum_ggml_type{ type_f32, mod.c.GGML_TYPE_F16 }) |type_b| {
                for ([_]i64{ 1, 16 }) |n| {
                    for ([_]i64{ 1, 16 }) |k| {
                        for ([_]i64{ 1, 3 }) |bs2| {
                            for ([_]i64{ 1, 3 }) |bs3| {
                                for ([_]i64{ 1, 2 }) |nr2| {
                                    for ([_]i64{ 1, 2 }) |nr3| {
                                        try list.add(mod.cases.TestOutProd, .{ type_a, type_b, 256, n, k, .{ bs2, bs3 }, .{ nr2, nr3 } });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ne2 sweep to cover the cublasSgemmStridedBatched path (dps2 == 1, ne2 > 1)
        for ([_]i64{ 1, 8, 16, 32 }) |ne2| {
            try list.add(mod.cases.TestOutProd, .{ type_f32, type_f32, 256, 16, 16, .{ ne2, 1 }, .{ 1, 1 } });
        }

        // nr2 sweep to cover the cublasSgemmBatched pointer-array path (dps2 > 1)
        for ([_]i64{ 8, 16, 32 }) |nr2| {
            try list.add(mod.cases.TestOutProd, .{ type_f32, type_f32, 256, 16, 16, .{ 1, 1 }, .{ nr2, 1 } });
        }

        // add_id
        for ([_]i64{ 4, 8 }) |n_mats| {
            for ([_]i64{ 1, 2, 4 }) |n_used| {
                for ([_]i64{ 32, 129 }) |n_embd| {
                    for ([_]i64{ 1, 32, 129 }) |n_token| {
                        try list.add(mod.cases.TestAddId, .{ type_f32, type_f32, n_embd, n_mats, n_used, n_token });
                        try list.add(mod.cases.TestAddId, .{ type_f32, type_f32, n_embd, n_mats, n_used, n_token, true });
                    }
                }
            }
        }
    }

    // The F16/F32 unary ops, and diag_mask_inf.
    fn addUnary(list: *mod.CaseList) !void {
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32 }) |t| {
            try list.add(mod.cases.TestSqr, .{t});
            try list.add(mod.cases.TestSqrt, .{t});
            try list.add(mod.cases.TestLog, .{t});
            try list.add(mod.cases.TestSin, .{t});
            try list.add(mod.cases.TestCos, .{t});
            try list.add(mod.cases.TestClamp, .{t});
            try list.add(mod.cases.TestClamp, .{ t, .{ 10, 5, 4, 3 }, -0.5, 0.5, 1 });
            try list.add(mod.cases.TestLeakyRelu, .{t});
            try list.add(mod.cases.TestFloor, .{t});
            try list.add(mod.cases.TestCeil, .{t});
            try list.add(mod.cases.TestRound, .{t});
            try list.add(mod.cases.TestTrunc, .{t});
            inline for (.{
                mod.cases.TestSqr,  mod.cases.TestSqrt,  mod.cases.TestLog,       mod.cases.TestSin,
                mod.cases.TestCos,  mod.cases.TestClamp, mod.cases.TestLeakyRelu, mod.cases.TestFloor,
                mod.cases.TestCeil, mod.cases.TestRound, mod.cases.TestTrunc,
            }) |U| {
                try list.add(U, .{ t, .{ 7, 1, 5, 3 } });
                try list.add(U, .{ t, .{ 1024, 1024, 1, 1 } });
            }
        }

        try list.add(mod.cases.TestDiagMaskInf, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 1, 1 }, 5 });
        try list.add(mod.cases.TestDiagMaskInf, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 3, 1 }, 5 });
        try list.add(mod.cases.TestDiagMaskInf, .{ mod.c.GGML_TYPE_F32, .{ 10, 10, 3, 2 }, 5 });
    }

    // soft_max and soft_max_back.
    fn addSoftMax(list: *mod.CaseList) !void {
        const T = mod.cases.TestSoftMax;
        const type_f32 = mod.c.GGML_TYPE_F32;
        const type_f16 = mod.c.GGML_TYPE_F16;

        // (upstream's `#if 0` random soft_max sweep driven by `rng` is compiled out)
        for ([_]bool{ false, true }) |mask| {
            for ([_]bool{ false, true }) |sinks| {
                for ([_]f32{ 0.0, 8.0 }) |max_bias| {
                    if (!mask and max_bias > 0.0) continue;
                    for ([_]f32{ 1.0, 0.1 }) |scale| {
                        for ([_]i64{ 16, 1024 }) |ne0| {
                            for ([_]i64{ 16, 1024 }) |ne1| {
                                if (mask) {
                                    for ([_]mod.c.enum_ggml_type{ type_f32, type_f16 }) |m_prec| {
                                        try list.add(T, .{ type_f32, .{ ne0, ne1, 1, 1 }, mask, sinks, m_prec, .{ 1, 1 }, scale, max_bias });
                                        try list.add(T, .{ type_f32, .{ ne0 - 1, ne1 - 1, 1, 1 }, mask, sinks, m_prec, .{ 1, 1 }, scale, max_bias });

                                        if (ne0 <= 32 and ne1 <= 32) {
                                            try list.add(T, .{ type_f32, .{ ne0, ne1, 1, 3 }, mask, sinks, m_prec, .{ 3, 1 }, scale, max_bias });
                                            try list.add(T, .{ type_f32, .{ ne0 - 1, ne1 - 1, 1, 1 }, mask, sinks, m_prec, .{ 2, 3 }, scale, max_bias });
                                        }
                                    }
                                } else {
                                    // The precision of mask here doesn't matter as boolean mask is false
                                    try list.add(T, .{ type_f32, .{ ne0, ne1, 1, 1 }, mask, sinks, type_f32, .{ 1, 1 }, scale, max_bias });
                                    try list.add(T, .{ type_f32, .{ ne0 - 1, ne1 - 1, 1, 1 }, mask, sinks, type_f32, .{ 1, 1 }, scale, max_bias });
                                }
                            }
                        }
                    }
                }
                // inplace tests
                try list.add(T, .{ type_f32, .{ 16, 2, 32, 1 }, mask, sinks, type_f32, .{ 1, 1 }, 0.1, 0.0, true });
                try list.add(T, .{ type_f32, .{ 16, 2, 32, 1 }, mask, sinks, type_f16, .{ 1, 1 }, 0.1, 0.0, true });
            }
        }
        try list.add(T, .{ type_f32, .{ 16, 2, 32, 1 }, true, true, type_f32, .{ 1, 1 }, 0.1, 0.0 });
        try list.add(T, .{ type_f32, .{ 16, 2, 32, 1 }, true, false, type_f16, .{ 1, 1 }, 0.1, 0.0 });
        try list.add(T, .{ type_f32, .{ 16, 2, 32, 1 }, false, true, type_f32, .{ 1, 1 }, 0.1, 0.0 });
        try list.add(T, .{ type_f32, .{ 32, 2, 32, 1 }, true, true, type_f32, .{ 1, 1 }, 0.1, 0.0 });
        try list.add(T, .{ type_f32, .{ 32, 2, 32, 1 }, true, false, type_f16, .{ 1, 1 }, 0.1, 0.0 });
        try list.add(T, .{ type_f32, .{ 32, 2, 32, 1 }, true, true, type_f32, .{ 1, 1 }, 0.1, 8.0 });
        try list.add(T, .{ type_f32, .{ 32, 2, 32, 1 }, true, true, type_f16, .{ 1, 1 }, 0.1, 8.0 });

        try list.add(T, .{ type_f32, .{ 200001, 2, 3, 1 }, true, true, type_f32, .{ 1, 1 }, 0.1, 8.0 });
        try list.add(T, .{ type_f32, .{ 200001, 2, 3, 1 }, true, true, type_f16, .{ 1, 1 }, 0.1, 8.0 });
        try list.add(T, .{ type_f32, .{ 200000, 1, 1, 1 }, false, false, type_f32, .{ 1, 1 }, 1.0, 0.0 });
        try list.add(T, .{ type_f32, .{ 200000, 4, 1, 1 }, false, false, type_f32, .{ 1, 1 }, 1.0, 0.0 });
        try list.add(T, .{ type_f32, .{ 4, 1, 1, 1 }, false, false, type_f32, .{ 1, 1 }, 1.0, 0.0 });
        try list.add(T, .{ type_f32, .{ 4, 1023, 1, 1 }, false, false, type_f32, .{ 1, 1 }, 1.0, 0.0 });
        try list.add(T, .{ type_f32, .{ 643251, 3, 1, 1 }, false, false, type_f32, .{ 1, 1 }, 1.0, 0.0 });

        for ([_]f32{ 0.0, 8.0 }) |max_bias| {
            for ([_]f32{ 1.0, 0.1 }) |scale| {
                for ([_]i64{ 16, 1024 }) |ne0| {
                    for ([_]i64{ 16, 1024 }) |ne1| {
                        try list.add(mod.cases.TestSoftMaxBack, .{ type_f32, .{ ne0, ne1, 1, 1 }, scale, max_bias });
                        try list.add(mod.cases.TestSoftMaxBack, .{ type_f32, .{ ne0 - 1, ne1 - 1, 1, 1 }, scale, max_bias });
                        try list.add(mod.cases.TestSoftMaxBack, .{ type_f32, .{ ne0, ne1, 2, 3 }, scale, max_bias });
                    }
                }
            }
        }
    }

    // rope.
    fn addRope(list: *mod.CaseList) !void {
        const T = mod.cases.TestRope;
        const normal = mod.c.GGML_ROPE_TYPE_NORMAL;
        const neox = mod.c.GGML_ROPE_TYPE_NEOX;
        const mrope = mod.c.GGML_ROPE_TYPE_MROPE;
        const imrope = mod.c.GGML_ROPE_TYPE_IMROPE;
        const vision = mod.c.GGML_ROPE_TYPE_VISION;
        const type_f32 = mod.c.GGML_TYPE_F32;
        const rope_modes = [_]i32{ normal, neox, mrope, imrope, vision };

        for ([_]bool{ true, false }) |fw| { // fw == forward
            var all = true;

            for ([_]f32{ 1.0, 1.4245 }) |fs| {
                for ([_]f32{ 0.0, 0.7465 }) |ef| {
                    for ([_]f32{ 1.0, 1.4245 }) |af| {
                        for ([_]mod.c.enum_ggml_type{ type_f32, mod.c.GGML_TYPE_F16 }) |t| {
                            for ([_]bool{ false, true }) |ff| { // freq_factors
                                for ([_]i32{ 0, 1 }) |v| {
                                    try list.add(T, .{ t, .{ 128, 32, 2, 1 }, 128, normal, 512, fs, ef, af, ff, v, fw }); // llama 7B

                                    if (all) {
                                        try list.add(T, .{ t, .{ 128, 40, 2, 1 }, 128, normal, 512, fs, ef, af, ff, v, fw }); // llama 13B
                                        try list.add(T, .{ t, .{ 128, 52, 2, 1 }, 128, normal, 512, fs, ef, af, ff, v, fw }); // llama 30B
                                        try list.add(T, .{ t, .{ 128, 64, 2, 1 }, 128, normal, 512, fs, ef, af, ff, v, fw }); // llama 65B
                                        try list.add(T, .{ t, .{ 16, 16, 8192, 1 }, 16, normal, 512, fs, ef, af, ff, v, fw });
                                    }

                                    if (all) {
                                        try list.add(T, .{ t, .{ 64, 1, 2, 1 }, 64, neox, 512, fs, ef, af, ff, v, fw }); // neox (falcon 7B)
                                        try list.add(T, .{ t, .{ 64, 71, 2, 1 }, 64, neox, 512, fs, ef, af, ff, v, fw }); // neox (falcon 7B)
                                        try list.add(T, .{ t, .{ 64, 8, 2, 1 }, 64, neox, 512, fs, ef, af, ff, v, fw }); // neox (falcon 40B)

                                        try list.add(T, .{ t, .{ 80, 32, 2, 1 }, 20, normal, 512, fs, ef, af, ff, v, fw });
                                        try list.add(T, .{ t, .{ 80, 32, 2, 1 }, 32, normal, 512, fs, ef, af, ff, v, fw });
                                        try list.add(T, .{ t, .{ 80, 32, 4, 1 }, 32, normal, 512, fs, ef, af, ff, v, fw });

                                        try list.add(T, .{ t, .{ 80, 32, 2, 1 }, 20, neox, 512, fs, ef, af, ff, v, fw }); // neox (stablelm)
                                        try list.add(T, .{ t, .{ 80, 32, 2, 1 }, 32, neox, 512, fs, ef, af, ff, v, fw }); // neox (phi-2)
                                        try list.add(T, .{ t, .{ 80, 32, 4, 1 }, 32, neox, 512, fs, ef, af, ff, v, fw }); // neox (phi-2)
                                        try list.add(T, .{ t, .{ 16, 16, 8192, 1 }, 16, neox, 512, fs, ef, af, ff, v, fw });
                                    }

                                    if (all) {
                                        try list.add(T, .{ t, .{ 128, 12, 2, 1 }, 128, mrope, 512, fs, ef, af, ff, v, fw }); // rope_multi,m-rope (qwen2vl 2B)
                                        try list.add(T, .{ t, .{ 128, 28, 2, 1 }, 128, mrope, 512, fs, ef, af, ff, v, fw }); // rope_multi,m-rope (qwen2vl 7B)
                                        try list.add(T, .{ t, .{ 128, 12, 2, 1 }, 20, mrope, 512, fs, ef, af, ff, v, fw });
                                        try list.add(T, .{ t, .{ 128, 28, 2, 1 }, 32, mrope, 512, fs, ef, af, ff, v, fw });
                                        try list.add(T, .{ t, .{ 128, 12, 2, 1 }, 128, imrope, 512, fs, ef, af, ff, v, fw }); // rope_multi,imrope (qwen3vl 2B)
                                        try list.add(T, .{ t, .{ 128, 28, 2, 1 }, 128, imrope, 512, fs, ef, af, ff, v, fw }); // rope_multi,imrope (qwen3vl 7B)
                                        try list.add(T, .{ t, .{ 128, 12, 2, 1 }, 20, imrope, 512, fs, ef, af, ff, v, fw });
                                        try list.add(T, .{ t, .{ 128, 28, 2, 1 }, 32, imrope, 512, fs, ef, af, ff, v, fw });
                                        try list.add(T, .{ t, .{ 80, 16, 2, 1 }, 80, vision, 512, fs, ef, af, ff, v, fw }); // rope_multi,m-rope (qwen2vl ViT)
                                        try list.add(T, .{ t, .{ 128, 16, 2, 1 }, 128, imrope, 512, fs, ef, af, ff, v, fw }); // rope_multi,m-rope (qwen3vl)
                                        try list.add(T, .{ t, .{ 16, 16, 8192, 1 }, 16, imrope, 512, fs, ef, af, ff, v, fw });
                                    }

                                    try list.add(T, .{ t, .{ 64, 128, 2, 1 }, 64, neox, 512, fs, ef, af, ff, v, fw }); // neox (falcon 40B)
                                }

                                // build_rope_2d-style: ROPE on a non-contiguous view that starts at a
                                // non-zero offset along dim 0 (e.g. gemma4v vision second-half view).
                                for (rope_modes) |rmode| {
                                    try list.add(T, .{ t, .{ 36, 16, 2457, 1 }, 36, rmode, 512, fs, ef, af, ff, 2, fw });
                                }
                            }

                            all = false;
                        }
                    }
                }
            }
        }

        // single inplace test per type/mode/ff
        for ([_]mod.c.enum_ggml_type{ type_f32, mod.c.GGML_TYPE_F16 }) |t| {
            for (rope_modes) |mode| {
                for ([_]bool{ false, true }) |ff| {
                    try list.add(T, .{ t, .{ 128, 32, 2, 1 }, 128, mode, 512, 1.4245, 0.7465, 1.4245, ff, 0, true, true });
                    try list.add(T, .{ t, .{ 128, 32, 2, 1 }, 128, mode, 512, 1.4245, 0.7465, 1.4245, ff, 1, true, true });
                    try list.add(T, .{ t, .{ 128, 32, 2, 3 }, 128, mode, 512, 1.4245, 0.7465, 1.4245, ff, 1, true, true });
                }
            }
        }

        // rotated dims window at an offset (ggml_rope_set_offset), not supported for vision mode
        for ([_]mod.c.enum_ggml_type{ type_f32, mod.c.GGML_TYPE_F16 }) |t| {
            for ([_]bool{ true, false }) |fw| { // fw == forward
                for ([_]bool{ false, true }) |ff| {
                    try list.add(T, .{ t, .{ 128, 32, 2, 1 }, 32, normal, 512, 1.4245, 0.7465, 1.4245, ff, 0, fw, false, 32 });
                    try list.add(T, .{ t, .{ 128, 32, 2, 1 }, 32, neox, 512, 1.4245, 0.7465, 1.4245, ff, 0, fw, false, 32 });
                    try list.add(T, .{ t, .{ 128, 12, 2, 1 }, 24, mrope, 512, 1.4245, 0.7465, 1.4245, ff, 0, fw, false, 32 });
                    try list.add(T, .{ t, .{ 128, 12, 2, 1 }, 24, imrope, 512, 1.4245, 0.7465, 1.4245, ff, 0, fw, false, 32 });
                }
            }
            // inplace with an offset
            try list.add(T, .{ t, .{ 128, 32, 2, 1 }, 32, neox, 512, 1.4245, 0.7465, 1.4245, false, 0, true, true, 32 });
        }

        // Real-model RoPE: F32 forward, packed Q, 512-token prefill.
        try list.add(T, .{ type_f32, .{ 256, 8, 512, 1 }, 64, imrope, 512, 1.0, 0.0, 1.0, false, 0, true }); // qwen3.5 0.8B
        try list.add(T, .{ type_f32, .{ 256, 16, 512, 1 }, 64, imrope, 512, 1.0, 0.0, 1.0, false, 0, true }); // qwen3.5 4B
        try list.add(T, .{ type_f32, .{ 256, 8, 512, 1 }, 256, neox, 512, 1.0, 0.0, 1.0, false, 0, true }); // gemma4 E2B sliding
        try list.add(T, .{ type_f32, .{ 512, 8, 512, 1 }, 128, neox, 512, 1.0, 0.0, 1.0, true, 0, true }); // gemma4 E4B global
    }

    // concat.
    fn addConcat(list: *mod.CaseList) !void {
        const T = mod.cases.TestConcat;
        for ([_]i32{ 0, 1, 2, 3, 16 }) |v| {
            for ([_]i32{ 0, 1, 2, 3 }) |dim| {
                for ([_]mod.c.enum_ggml_type{
                    mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_I8,
                    mod.c.GGML_TYPE_I16, mod.c.GGML_TYPE_I32, mod.c.GGML_TYPE_I64,
                }) |t| {
                    try list.add(T, .{ t, .{ 11, 12, 13, 14 }, 7, dim, v });
                }
            }
        }

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_1, mod.c.GGML_TYPE_Q5_0, mod.c.GGML_TYPE_Q5_1, mod.c.GGML_TYPE_Q8_0 }) |type_a| {
            for ([_]i32{ 0, 4, 8, 12 }) |v| {
                for ([_]i32{ 0, 1, 2, 3 }) |dim| {
                    try list.add(T, .{ type_a, .{ 128, 12, 13, 14 }, @as(i64, if (dim == 0) 256 else 7), dim, v });
                }
            }
        }

        // few long rows, which backends may split across workgroups
        for ([_]i32{ 0, 1 }) |dim| {
            try list.add(T, .{ mod.c.GGML_TYPE_F32, .{ 98304, 8, 1, 1 }, @as(i64, if (dim == 0) 288 else 2), dim, 0 });
            try list.add(T, .{ mod.c.GGML_TYPE_F32, .{ 20000, 3, 2, 1 }, @as(i64, if (dim == 0) 333 else 1), dim, 1 });
            try list.add(T, .{ mod.c.GGML_TYPE_Q8_0, .{ 8192, 2, 1, 1 }, @as(i64, if (dim == 0) 4096 else 1), dim, 0 });
        }
    }

    // argsort, top_k and top_k fusions.
    fn addSortTopK(list: *mod.CaseList) !void {
        const A = mod.cases.TestArgsort;
        const K = mod.cases.TestTopK;
        const type_f32 = mod.c.GGML_TYPE_F32;

        for ([_]mod.c.enum_ggml_sort_order{ mod.c.GGML_SORT_ORDER_ASC, mod.c.GGML_SORT_ORDER_DESC }) |order| {
            var i: i64 = 4;
            while (i <= 1024 * 1024) : (i *= 2) {
                try list.add(A, .{ type_f32, .{ i - 1, 1, 1, 1 } });
                try list.add(A, .{ type_f32, .{ i, 1, 1, 1 } });
            }
            try list.add(A, .{ type_f32, .{ 16, 10, 10, 10 }, order });
            try list.add(A, .{ type_f32, .{ 60, 10, 10, 10 }, order }); // qwen
            try list.add(A, .{ type_f32, .{ 1023, 2, 1, 3 }, order });
            try list.add(A, .{ type_f32, .{ 1024, 2, 1, 3 }, order });
            try list.add(A, .{ type_f32, .{ 1025, 2, 1, 3 }, order });
            try list.add(A, .{ type_f32, .{ 1025, 256, 1, 1 }, order }); // test ceildiv in CUDA's CUB's DeviceSegmentedSort
            try list.add(A, .{ type_f32, .{ 2047, 2, 1, 3 }, order });
            try list.add(A, .{ type_f32, .{ 2048, 2, 1, 3 }, order });
            try list.add(A, .{ type_f32, .{ 2049, 2, 1, 3 }, order });
            try list.add(A, .{ type_f32, .{ 2, 8, 8192, 1 }, order }); // bailingmoe2 (group selection)
            try list.add(A, .{ type_f32, .{ 2048, 512, 1, 1 }, order }); // test CUDA dispatching to radix sort for nrows > = 1 in graph mode
        }

        var n: i64 = 1;
        while (n < 5) : (n += 1) {
            var k: i64 = 1;
            while (k <= n) : (k += 1) {
                try list.add(K, .{ type_f32, .{ n, 2, 1, 3 }, k, true });
            }
        }
        for ([_]i32{ 1, 10, 40 }) |k| {
            try list.add(mod.cases.TestTopKInf, .{ .{ 4096, 2, 1, 1 }, k });
            try list.add(mod.cases.TestTopKInf, .{ .{ 248320, 1, 1, 1 }, k });
        }

        var i: u6 = 0;
        while (i < 20) : (i += 1) {
            const p = @as(i64, 1) << i;
            for ([_]i64{ 1, 2, 3, 7, 15, 100, 500, 1023, 9999 }) |k| {
                if (k <= p) {
                    try list.add(K, .{ type_f32, .{ p, 1, 1, 1 }, k });
                    try list.add(K, .{ type_f32, .{ p + 11, 1, 2, 1 }, k });
                    try list.add(K, .{ type_f32, .{ p + 11, 1, 2, 1 }, k, true });
                }
            }
        }
        for ([_]i32{ 4, 8, 16, 32 }) |k| {
            for ([_]i64{ 1, 8, 16 }) |nrows| {
                try list.add(K, .{ type_f32, .{ 202048, nrows, 1, 1 }, k });
                try list.add(K, .{ type_f32, .{ 151936, nrows, 1, 1 }, k });
                try list.add(K, .{ type_f32, .{ 8192, nrows, 1, 1 }, k });
                try list.add(K, .{ type_f32, .{ 8193, nrows, 1, 1 }, k });
                try list.add(K, .{ type_f32, .{ 8192, nrows, 1, 1 }, k, true });
                try list.add(K, .{ type_f32, .{ 202048, nrows, 1, 1 }, k, true });
            }
        }

        for ([_]i32{ 1, 2, 3, 7, 15 }) |k| {
            try list.add(K, .{ type_f32, .{ 16, 10, 10, 10 }, k });
            try list.add(K, .{ type_f32, .{ 60, 10, 10, 10 }, k });
            try list.add(K, .{ type_f32, .{ 1023, 2, 1, 3 }, k });
            try list.add(K, .{ type_f32, .{ 1024, 2, 1, 3 }, k });
            try list.add(K, .{ type_f32, .{ 1025, 2, 1, 3 }, k });
            try list.add(K, .{ type_f32, .{ 16384, 1, 1, 1 }, k });
            try list.add(K, .{ type_f32, .{ 2047, 2, 1, 3 }, k });
            try list.add(K, .{ type_f32, .{ 2048, 2, 1, 3 }, k });
            try list.add(K, .{ type_f32, .{ 2049, 2, 1, 3 }, k });
        }

        // Large-k, including multi-row and ties (qwen4exp)
        try list.add(K, .{ type_f32, .{ 1024, 1, 1, 1 }, 1024 });
        try list.add(K, .{ type_f32, .{ 2048, 2, 1, 1 }, 1024 });
        try list.add(K, .{ type_f32, .{ 4096, 1, 1, 1 }, 2048 });
        try list.add(K, .{ type_f32, .{ 8192, 2, 1, 1 }, 2051 });
        try list.add(K, .{ type_f32, .{ 33024, 1, 1, 1 }, 2051 });
        try list.add(K, .{ type_f32, .{ 33024, 4, 1, 1 }, 2051 });
        try list.add(K, .{ type_f32, .{ 8192, 2, 1, 1 }, 2051, true });
        try list.add(K, .{ type_f32, .{ 33024, 4, 1, 1 }, 2051, true });

        // rows that CUDA processes in several chunks (bitonic, radix)
        try list.add(K, .{ type_f32, .{ 500, 40000, 1, 1 }, 16 });
        try list.add(K, .{ type_f32, .{ 1100, 33000, 1, 1 }, 16 });

        // qwen4exp QSA indexer top-k fusion (get_rows + type_f16 mask + top_k)
        try list.add(mod.cases.TestTopkQsa, .{ 512, 2048, 1, 1, 1500 });
        try list.add(mod.cases.TestTopkQsa, .{ 512, 2048, 2, 1, 1500 });
        try list.add(mod.cases.TestTopkQsa, .{ 256, 2048, 4, 2, 2000 });
        try list.add(mod.cases.TestTopkQsa, .{ 64, 256, 2, 1, 200 }); // small k: unfused fallback

        // (upstream's exhaustive top_k loop stays commented out)
    }

    // upscale/interpolate through pad_ext: the small single-op cases.
    fn addMisc(list: *mod.CaseList) !void {
        const type_f32 = mod.c.GGML_TYPE_F32;
        const type_f16 = mod.c.GGML_TYPE_F16;
        const type_bf16 = mod.c.GGML_TYPE_BF16;

        for ([_]u32{
            mod.c.GGML_SCALE_MODE_NEAREST,
            mod.c.GGML_SCALE_MODE_BILINEAR,
            mod.c.GGML_SCALE_MODE_BICUBIC,
            mod.c.GGML_SCALE_MODE_BILINEAR | mod.c.GGML_SCALE_FLAG_ANTIALIAS,
        }) |mode| {
            try list.add(mod.cases.TestUpscale, .{ type_f32, .{ 512, 512, 3, 2 }, 2, mode });
            try list.add(mod.cases.TestUpscale, .{ type_f32, .{ 512, 512, 3, 2 }, 2, mode, true });
            try list.add(mod.cases.TestInterpolate, .{ type_f32, .{ 2, 5, 7, 11 }, .{ 5, 7, 11, 13 }, mode });
            try list.add(mod.cases.TestInterpolate, .{ type_f32, .{ 5, 7, 11, 13 }, .{ 2, 5, 7, 11 }, mode });
        }
        for ([_]u32{ mod.c.GGML_SCALE_MODE_BILINEAR, mod.c.GGML_SCALE_MODE_BICUBIC }) |mode| {
            const m = mode | mod.c.GGML_SCALE_FLAG_ALIGN_CORNERS;
            try list.add(mod.cases.TestInterpolate, .{ type_f32, .{ 2, 5, 7, 11 }, .{ 5, 7, 11, 13 }, m });
            try list.add(mod.cases.TestInterpolate, .{ type_f32, .{ 1, 4, 3, 2 }, .{ 2, 8, 3, 2 }, m });
            try list.add(mod.cases.TestInterpolate, .{ type_f32, .{ 4, 1, 3, 2 }, .{ 1, 1, 3, 2 }, m });
            try list.add(mod.cases.TestInterpolate, .{ type_f32, .{ 64, 48, 48, 1 }, .{ 156, 104, 64, 1 }, m, .{ 2, 0, 1, 3 } });
        }

        const S = mod.cases.TestSum;
        const M = mod.cases.TestMean;
        const R = mod.cases.TestSumRows;
        try list.add(S, .{});
        try list.add(S, .{ type_f32, .{ 11, 5, 6, 3 }, .{ 0, 2, 1, 3 } }); // row-contiguous but non-contiguous
        try list.add(S, .{ type_f32, .{ 11, 5, 6, 3 }, .{ 0, 3, 2, 1 } });
        try list.add(S, .{ type_f32, .{ 11, 5, 6, 3 }, .{ 0, 1, 3, 2 } });
        try list.add(M, .{});
        try list.add(M, .{ type_f32, .{ 33, 1, 1, 1 } });
        try list.add(M, .{ type_f32, .{ 33, 256, 1, 1 } });
        try list.add(M, .{ type_f32, .{ 32769, 1, 1, 1 } });
        try list.add(M, .{ type_f32, .{ 32, 1, 1, 1 } });
        try list.add(M, .{ type_f32, .{ 32, 256, 1, 1 } });
        try list.add(M, .{ type_f32, .{ 32768, 1, 1, 1 } });
        try list.add(M, .{ type_f32, .{ 11, 5, 6, 3 }, true, false });
        try list.add(M, .{ type_f32, .{ 11, 5, 6, 3 }, false, true });
        try list.add(M, .{ type_f32, .{ 11, 5, 6, 3 }, true, true });
        try list.add(S, .{ type_f32, .{ 33, 1, 1, 1 } });
        try list.add(S, .{ type_f32, .{ 33, 1024, 1, 1 } });
        try list.add(S, .{ type_f32, .{ 33, 256, 1, 1 } });
        try list.add(S, .{ type_f32, .{ 33, 256, 1, 1 }, .{ 1, 0, 2, 3 } }); // sum dst not-contiguous
        try list.add(S, .{ type_f32, .{ 10, 5, 4, 3 }, .{ 1, 0, 2, 3 } }); // non-contiguous rows, 4D
        try list.add(S, .{ type_f32, .{ 11, 5, 6, 3 }, .{ 2, 0, 1, 3 } }); // rotated dims, nb[0] != sizeof(float)
        try list.add(S, .{ type_f32, .{ 11, 5, 6, 3 }, .{ 0, 0, 0, 0 }, true }); // sliced view (gaps)
        try list.add(S, .{ type_f32, .{ 11, 5, 6, 3 }, .{ 1, 0, 2, 3 }, true }); // sliced + non-contiguous rows
        try list.add(R, .{});
        try list.add(R, .{ type_f32, .{ 11, 5, 6, 3 }, true, false });
        try list.add(R, .{ type_f32, .{ 11, 5, 6, 3 }, false, true });
        try list.add(R, .{ type_f32, .{ 11, 5, 6, 3 }, true, true });
        try list.add(R, .{ type_f32, .{ 16, 5, 6, 3 }, true, false });
        try list.add(R, .{ type_f32, .{ 16, 5, 6, 3 }, false, true });
        try list.add(R, .{ type_f32, .{ 16, 5, 6, 3 }, true, true });
        try list.add(R, .{ type_f32, .{ 33, 1, 1, 1 } });
        try list.add(R, .{ type_f32, .{ 33, 1024, 1, 1 } });
        try list.add(R, .{ type_f32, .{ 33, 256, 1, 1 } });
        try list.add(mod.cases.TestGroupNorm, .{ type_f32, .{ 64, 64, 320, 1 } });
        try list.add(mod.cases.TestGroupNorm, .{ type_f32, .{ 9, 9, 1280, 1 } });
        try list.add(mod.cases.TestGroupNormMulAdd, .{ type_f32, .{ 64, 64, 320, 1 } });
        try list.add(mod.cases.TestGroupNormMulAdd, .{ type_f32, .{ 9, 9, 1280, 1 } });
        try list.add(mod.cases.TestAcc, .{ type_f32, .{ 256, 17, 1, 1 }, .{ 256, 16, 1, 1 }, -1 });
        try list.add(mod.cases.TestAcc, .{ type_f32, .{ 256, 17, 2, 3 }, .{ 256, 16, 2, 3 }, -1 });
        try list.add(mod.cases.TestAcc, .{ type_f32, .{ 256, 17, 2, 3 }, .{ 128, 16, 2, 3 }, -1 });
        try list.add(mod.cases.TestAcc, .{ type_f32, .{ 256, 17, 2, 3 }, .{ 256, 16, 2, 3 }, 1 });
        try list.add(mod.cases.TestAcc, .{ type_f32, .{ 256, 17, 2, 3 }, .{ 128, 16, 2, 3 }, 2 });
        try list.add(mod.cases.TestAcc, .{ type_f32, .{ 256, 17, 2, 3 }, .{ 64, 16, 2, 3 }, 3 });

        const P = mod.cases.TestPad;
        try list.add(P, .{});
        try list.add(P, .{ type_f32, .{ 33, 17, 2, 1 }, 4, 3, true }); // circular
        try list.add(mod.cases.TestPadExt, .{});
        try list.add(P, .{ type_f32, .{ 1024, 1, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 1024, 2, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 1024, 16, 1, 1 }, 0, 1, false });
        try list.add(P, .{ type_f32, .{ 1023, 1, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 1023, 8, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 1025, 1, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 1025, 8, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 2048, 1, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 2048, 4, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 2049, 1, 1, 1 }, 1, 0, false });
        try list.add(P, .{ type_f32, .{ 100, 1, 1, 1 }, 100, 0, false });
        try list.add(P, .{ type_f32, .{ 100, 1, 1, 1 }, 0, 100, false });
        try list.add(P, .{ type_f32, .{ 100, 100, 1, 1 }, 50, 50, false });
        // more than 65535 rows or slices, beyond the CUDA grid.y/grid.z limit
        try list.add(P, .{ type_f32, .{ 4, 70000, 1, 1 }, 1, 1, false });
        try list.add(P, .{ type_f32, .{ 4, 70000, 1, 1 }, 1, 1, true });
        try list.add(mod.cases.TestPadExt, .{ type_f32, .{ 4, 2, 300, 300 }, 1, 1, 0, 0, 0, 0, 0, 0, 0, false });

        try list.add(mod.cases.TestPadReflect1d, .{});
        try list.add(mod.cases.TestPadReflect1d, .{ type_f32, .{ 3000, 384, 4, 1 } });
        try list.add(mod.cases.TestRoll, .{});
        try list.add(mod.cases.TestRoll, .{ 3, -2, 1, -1, true });
        try list.add(mod.cases.TestArange, .{});
        try list.add(mod.cases.TestArange, .{ type_f32, 3.0, 3.5, 1.0 });
        try list.add(mod.cases.TestArange, .{ type_f32, -10.0, 10.0, 0.3 });
        for ([_]f32{ 255.0, 256.0, 257.0, 1023.0, 1024.0, 1025.0 }) |n| {
            try list.add(mod.cases.TestArange, .{ type_f32, -1.0, -1.0 + n * 0.25, 0.25 });
        }
        try list.add(mod.cases.TestArange, .{ type_f32, 0.0, 1048576.0, 1.0 });
        try list.add(mod.cases.TestTimestepEmbedding, .{});
        try list.add(mod.cases.TestLeakyRelu, .{});

        for ([_][4]i64{
            .{ 10, 5, 4, 3 },         .{ 127, 5, 4, 3 },        .{ 128, 5, 4, 3 },   .{ 128, 128, 4, 4 },
            .{ 255, 5, 4, 3 },        .{ 256, 5, 4, 3 },        .{ 511, 5, 4, 3 },   .{ 512, 5, 4, 3 },
            .{ 1023, 5, 4, 3 },       .{ 1024, 5, 4, 3 },       .{ 2047, 5, 4, 3 },  .{ 2048, 5, 4, 3 },
            .{ 201 * 1204, 1, 1, 1 }, .{ 312 * 1205, 1, 1, 1 }, .{ 20481, 4, 1, 1 },
        }) |ne| {
            try list.add(mod.cases.TestCumsum, .{ type_f32, ne });
        }

        try list.add(mod.cases.TestXielu, .{});
        try list.add(mod.cases.TestXielu, .{type_f16});
        try list.add(mod.cases.TestXielu, .{type_bf16});
        try list.add(mod.cases.TestXielu, .{ type_f32, .{ 512, 16, 1, 1 } });
        try list.add(mod.cases.TestXielu, .{ type_f16, .{ 512, 16, 1, 1 } });
        try list.add(mod.cases.TestXielu, .{ type_bf16, .{ 512, 16, 1, 1 } });

        try list.add(mod.cases.TestTri, .{mod.c.GGML_TRI_TYPE_LOWER});
        try list.add(mod.cases.TestTri, .{mod.c.GGML_TRI_TYPE_LOWER_DIAG});
        try list.add(mod.cases.TestTri, .{mod.c.GGML_TRI_TYPE_UPPER});
        try list.add(mod.cases.TestTri, .{mod.c.GGML_TRI_TYPE_UPPER_DIAG});

        try list.add(mod.cases.TestFill, .{0.0});
        try list.add(mod.cases.TestFill, .{ 2.0, type_f32, .{ 303, 207, 11, 3 } });
        try list.add(mod.cases.TestFill, .{ -152.0, type_f32, .{ 800, 600, 4, 4 } });
        try list.add(mod.cases.TestFill, .{ 3.5, type_f32, .{ 2048, 512, 2, 2 } });
        try list.add(mod.cases.TestFill, .{ 2.0, type_f16, .{ 303, 207, 11, 3 } });
        try list.add(mod.cases.TestFill, .{ -152.0, type_f16, .{ 800, 600, 4, 4 } });
        try list.add(mod.cases.TestFill, .{ 3.5, type_f16, .{ 2048, 512, 2, 2 } });

        try list.add(mod.cases.TestDiag, .{});
        try list.add(mod.cases.TestDiag, .{ type_f32, .{ 79, 1, 19, 13 } });
        try list.add(mod.cases.TestDiag, .{ type_f32, .{ 256, 1, 8, 16 } });

        try list.add(mod.cases.TestSolveTri, .{});
        for ([_][2][4]i64{
            .{ .{ 11, 11, 1, 1 }, .{ 5, 11, 1, 1 } },     .{ .{ 17, 17, 2, 4 }, .{ 9, 17, 2, 4 } },
            .{ .{ 30, 30, 7, 1 }, .{ 8, 30, 7, 1 } },     .{ .{ 42, 42, 5, 2 }, .{ 10, 42, 5, 2 } },
            .{ .{ 64, 64, 2, 2 }, .{ 10, 64, 2, 2 } },    .{ .{ 64, 64, 2, 2 }, .{ 64, 64, 2, 2 } },
            .{ .{ 79, 79, 5, 3 }, .{ 417, 79, 5, 3 } },   .{ .{ 128, 128, 4, 2 }, .{ 32, 128, 4, 2 } },
            .{ .{ 80, 80, 2, 8 }, .{ 80, 80, 2, 8 } },    .{ .{ 80, 80, 2, 8 }, .{ 79, 80, 2, 8 } },
            .{ .{ 80, 80, 2, 8 }, .{ 81, 80, 2, 8 } },    .{ .{ 80, 80, 8, 8 }, .{ 80, 80, 8, 8 } },
            .{ .{ 80, 80, 8, 8 }, .{ 79, 80, 8, 8 } },    .{ .{ 80, 80, 8, 8 }, .{ 81, 80, 8, 8 } },
            .{ .{ 84, 84, 4, 4 }, .{ 32, 84, 4, 4 } },    .{ .{ 95, 95, 8, 8 }, .{ 40, 95, 8, 8 } },
            .{ .{ 100, 100, 4, 4 }, .{ 41, 100, 4, 4 } }, .{ .{ 128, 128, 4, 4 }, .{ 31, 128, 4, 4 } },
            .{ .{ 128, 128, 4, 4 }, .{ 32, 128, 4, 4 } }, .{ .{ 128, 128, 3, 4 }, .{ 32, 128, 3, 4 } },
            .{ .{ 128, 128, 4, 1 }, .{ 32, 128, 4, 1 } }, .{ .{ 64, 64, 4, 4 }, .{ 200, 64, 4, 4 } },
            .{ .{ 64, 64, 4, 4 }, .{ 384, 64, 4, 4 } },
        }) |lr| {
            try list.add(mod.cases.TestSolveTri, .{ type_f32, lr[0], lr[1] });
        }

        for ([_]i32{ 0, 1, 2 }) |tfrm| {
            for ([_]bool{ false, true }) |circular| {
                try list.add(mod.cases.TestPadExt, .{ type_f32, .{ 512, 512, 1, 1 }, 0, 1, 0, 1, 0, 0, 0, 0, tfrm, circular });
                try list.add(mod.cases.TestPadExt, .{ type_f32, .{ 11, 22, 33, 44 }, 1, 2, 3, 4, 5, 6, 7, 8, tfrm, circular });
                try list.add(mod.cases.TestPadExt, .{ type_f32, .{ 11, 22, 33, 44 }, 0, 2, 0, 4, 0, 6, 0, 8, tfrm, circular });
            }
        }
    }

    // flash_attn_ext and its variants.
    fn addFlashAttn(list: *mod.CaseList) !void {
        const T = mod.cases.TestFlashAttnExt;
        const prec_f32 = mod.c.GGML_PREC_F32;
        const type_f32 = mod.c.GGML_TYPE_F32;
        const type_f16 = mod.c.GGML_TYPE_F16;
        const type_bf16 = mod.c.GGML_TYPE_BF16;
        const q8_0 = mod.c.GGML_TYPE_Q8_0;
        const q4_0 = mod.c.GGML_TYPE_Q4_0;
        const q1_0 = mod.c.GGML_TYPE_Q1_0;
        const q2_0 = mod.c.GGML_TYPE_Q2_0;
        const id = [4]i32{ 0, 1, 2, 3 };
        const perm = [4]i32{ 0, 2, 1, 3 };

        // prefill-shaped cases with long KV (nb >= 32, kv >= 1024): covers the
        // XMX/GEMM-accelerated SYCL FA path which only activates for these shapes.
        for ([_]i64{ 1024, 2048 }) |kv| {
            for ([_]i64{ 64, 128, 256 }) |hs| {
                for ([_]i64{ 32, 64 }) |nb| {
                    for ([_]mod.c.enum_ggml_type{ type_f16, q8_0, q4_0 }) |type_kv| {
                        try list.add(T, .{ hs, hs, 8, .{ 4, 1 }, kv, nb, true, false, 0, 0, prec_f32, type_kv, type_kv });
                    }
                }
            }
        }

        // KV-head counts that do not divide the device count
        for ([_]i64{ 128, 256, 512 }) |hs| {
            for ([_]i64{ 1, 2, 6 }) |nh| {
                for ([_]i64{ 4, 8 }) |nr2| {
                    for ([_]i64{ 75, 1024 }) |nb| {
                        try list.add(T, .{ hs, hs, nh, .{ nr2, 1 }, 1024, nb, true, false, 0, 0, prec_f32, type_f16, type_f16 });
                    }
                }
            }
        }

        for ([_]i64{ 40, 64, 72, 80, 96, 128, 192, 256, 320, 512, 576 }) |hsk| {
            for ([_]i64{ 40, 64, 72, 80, 96, 128, 192, 256, 512 }) |hsv| {
                if (hsk != 96 and hsk != 192 and hsk != 320 and hsk != 576 and hsk != hsv) continue;
                if (hsk == 96 and (hsv != 64 and hsv != 96)) continue; // MiniCPM3
                if (hsk == 192 and (hsv != 128 and hsv != 192)) continue;
                if (hsk == 576 and hsv != 512) continue; // DeepSeek MLA
                if (hsk == 320 and hsv != 256) continue; // Mistral4 MLA

                for ([_]bool{ true, false }) |mask| {
                    for ([_]bool{ true, false }) |sinks| {
                        for ([_]f32{ 0.0, 8.0 }) |max_bias| {
                            if (!mask and max_bias > 0.0) continue;
                            for ([_]f32{ 0.0, 10.0 }) |logit_softcap| {
                                if (hsk != 128 and logit_softcap != 0.0) continue;
                                for ([_]i64{ 1, 4 }) |nh| {
                                    if (nh == 1 and hsk != 320 and hsk != 576) continue;
                                    for ([_]i64{ 1, 3 }) |nr3| {
                                        if (hsk > 64 and nr3 > 1) continue; // skip broadcast for large head sizes
                                        for ([_]i64{ 1, 4, 8, 12, 16, 20, 32 }) |nr2| {
                                            if (nr2 == 8 and hsk != 192) continue;
                                            if (nr2 == 12 and hsk != 128) continue;
                                            if (nr2 == 16 and hsk != 192) continue;
                                            if (nr2 == 20 and (nh != 1 or hsk != 576)) continue;
                                            if (nr2 == 32 and (nh != 1 or hsk != 320)) continue;
                                            for ([_]i64{ 113, 512, 1024 }) |kv| {
                                                if (nr2 != 1 and kv != 512) continue;
                                                for ([_]i64{ 1, 3, 32, 75 }) |nb| {
                                                    for ([_]mod.c.enum_ggml_prec{ prec_f32, mod.c.GGML_PREC_DEFAULT }) |prec| {
                                                        if (hsk != 128 and prec == mod.c.GGML_PREC_DEFAULT) continue;
                                                        for (fa_kv_types) |type_kv| {
                                                            if (type_kv != type_f16 and hsk != 64 and hsk != 72) continue;
                                                            // DeepSeek MLA: the V cache is a sub-view of the K cache
                                                            const v_is_view_of_k = hsk == 576;
                                                            try list.add(T, .{ hsk, hsv, nh, .{ nr2, nr3 }, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_kv, type_kv, id, true, v_is_view_of_k });
                                                            // run fewer test cases permuted
                                                            if (mask and max_bias == 0.0 and logit_softcap == 0 and kv == 512) {
                                                                try list.add(T, .{ hsk, hsv, nh, .{ nr2, nr3 }, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_kv, type_kv, perm, true, v_is_view_of_k });
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
                }
            }
        }

        // asymmetric head_dim (hsk != hsv) with one or both sides not 64-aligned
        try list.add(T, .{ 72, 64, 4, .{ 1, 1 }, 256, 2, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 64, 72, 4, .{ 1, 1 }, 256, 2, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 65, 67, 4, .{ 1, 1 }, 113, 75, true, true, 8.0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 65, 67, 4, .{ 1, 1 }, 17, 75, false, false, 0, 1.0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 65, 67, 4, .{ 1, 1 }, 113, 75, false, false, 0, 1.0, prec_f32, type_f16, type_f16 });

        // mixed quant and Q1_0 test cases
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 128, 2, true, false, 0, 0, prec_f32, q8_0, q4_0 });
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 128, 2, true, false, 0, 0, prec_f32, q4_0, type_f16 });
        try list.add(T, .{ 72, 72, 4, .{ 1, 1 }, 96, 2, true, false, 0, 0, prec_f32, q4_0, q8_0 });
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 96, 2, true, false, 0, 0, prec_f32, type_f16, type_f32 });
        try list.add(T, .{ 128, 128, 4, .{ 1, 1 }, 256, 1, false, false, 0, 0, prec_f32, type_f16, q4_0 });
        try list.add(T, .{ 128, 128, 4, .{ 1, 1 }, 96, 2, true, false, 0, 0, prec_f32, q1_0, q1_0 });
        try list.add(T, .{ 128, 64, 4, .{ 1, 1 }, 128, 2, true, false, 0, 0, prec_f32, q1_0, q4_0 });
        try list.add(T, .{ 64, 128, 4, .{ 1, 1 }, 128, 2, true, false, 0, 0, prec_f32, q4_0, q1_0 });
        try list.add(T, .{ 128, 64, 4, .{ 1, 1 }, 64, 2, true, false, 0, 0, prec_f32, q1_0, type_f16 });
        try list.add(T, .{ 128, 128, 4, .{ 1, 1 }, 96, 2, true, false, 0, 0, prec_f32, q2_0, q2_0 });
        try list.add(T, .{ 128, 64, 4, .{ 1, 1 }, 128, 2, true, false, 0, 0, prec_f32, q2_0, q4_0 });
        try list.add(T, .{ 64, 128, 4, .{ 1, 1 }, 128, 2, true, false, 0, 0, prec_f32, q4_0, q2_0 });
        try list.add(T, .{ 128, 64, 4, .{ 1, 1 }, 64, 2, true, false, 0, 0, prec_f32, q2_0, type_f16 });

        // q8_0 KV cases: decode and prompt batches, KV pad, permuted KV, feature flags, and long context
        try list.add(T, .{ 256, 256, 2, .{ 16, 1 }, 113, 1, true, false, 0, 0, prec_f32, q8_0, q8_0 });
        try list.add(T, .{ 256, 256, 2, .{ 16, 1 }, 1024, 1, true, false, 0, 0, prec_f32, q8_0, q8_0 });
        try list.add(T, .{ 256, 256, 2, .{ 16, 1 }, 1024, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, perm });
        try list.add(T, .{ 256, 256, 2, .{ 16, 2 }, 1025, 1, true, true, 8, 30, prec_f32, q8_0, q8_0 });
        try list.add(T, .{ 256, 256, 2, .{ 16, 1 }, 1025, 64, true, false, 0, 0, prec_f32, q8_0, q8_0, perm });
        try list.add(T, .{ 256, 256, 2, .{ 16, 1 }, 16384, 1, true, false, 0, 0, prec_f32, q8_0, q8_0 });

        // MLA shape: the V cache is a sub-view of the K cache, with quantized KV
        try list.add(T, .{ 576, 512, 1, .{ 8, 1 }, 113, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, true });
        try list.add(T, .{ 576, 512, 1, .{ 8, 1 }, 1024, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, true });
        try list.add(T, .{ 576, 512, 1, .{ 8, 1 }, 1024, 64, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, true });
        try list.add(T, .{ 576, 512, 1, .{ 8, 1 }, 4096, 24, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, true });
        try list.add(T, .{ 512, 512, 1, .{ 8, 1 }, 4096, 24, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, false });

        // Sparse mask hint: supported decode/prefill layouts and dense fallbacks.
        try list.add(T, .{ 512, 512, 1, .{ 8, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 512, 512, 1, .{ 8, 2 }, 4096, 3, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 768 });
        try list.add(T, .{ 576, 512, 1, .{ 16, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true, 512 });
        try list.add(T, .{ 576, 512, 1, .{ 16, 2 }, 4096, 2, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true, 768 });
        try list.add(T, .{ 512, 512, 1, .{ 8, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 2304 });
        try list.add(T, .{ 256, 256, 1, .{ 8, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });

        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 4096, 4, true, false, 8.0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });

        // flash attention over a list of K/V rows per slice (unified KV cache)
        const R = mod.cases.TestFlashAttnExtKvRows;
        for ([_]i64{ 1, 3, 8, 64, 512 }) |nb| {
            for ([_]i64{ 1, 4 }) |n_slices| {
                try list.add(R, .{ 256, 2, 8, 4096, 1024, nb, n_slices });
                try list.add(R, .{ 128, 8, 4, 4096, 512, nb, n_slices });
                try list.add(R, .{ 128, 4, 1, 1000, 256, nb, n_slices });
                try list.add(R, .{ 64, 4, 2, 3000, 768, nb, n_slices, q8_0 });
                try list.add(R, .{ 512, 2, 8, 4096, 1024, nb, n_slices });
                try list.add(R, .{ 256, 8, 2, 4096, 1280, nb, n_slices });
            }
        }

        // sparse mask with large batch size
        try list.add(T, .{ 512, 512, 1, .{ 8, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 512, 512, 1, .{ 8, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 2048 });
        try list.add(T, .{ 576, 512, 1, .{ 16, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true, 512 });
        try list.add(T, .{ 576, 512, 1, .{ 16, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true, 2048 });
        try list.add(T, .{ 256, 256, 1, .{ 8, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 2048 });

        // sparse attn (qwen4 shape - gqa 12)
        try list.add(T, .{ 256, 256, 2, .{ 12, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 256, 256, 2, .{ 12, 1 }, 8192, 64, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });
        try list.add(T, .{ 256, 256, 1, .{ 12, 2 }, 8192, 67, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });

        // sparse mask + quantized cache
        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 4096, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, false, 512 });
        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 4096, 64, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, false, 512 });

        // Qwen QSA: 256/256, gqa 12, budget 2048.
        try list.add(T, .{ 256, 256, 2, .{ 12, 1 }, 8192, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 2048 });
        // quantized cache, deep enough to take the sparse path
        try list.add(T, .{ 256, 256, 2, .{ 12, 1 }, 32768, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, false, 2048 });
        try list.add(T, .{ 256, 256, 2, .{ 12, 1 }, 32768, 1, true, false, 0, 0, prec_f32, q4_0, q4_0, id, true, false, 2048 });
        // single head with quantized K (MMQ on Vulkan)
        try list.add(T, .{ 128, 128, 1, .{ 1, 1 }, 8192, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, false, 512 });
        try list.add(T, .{ 128, 128, 1, .{ 1, 1 }, 8192, 1, true, false, 0, 0, prec_f32, mod.c.GGML_TYPE_Q4_1, mod.c.GGML_TYPE_Q4_1, id, true, false, 512 });
        // KV not a multiple of the compaction workgroup size.
        try list.add(T, .{ 128, 128, 1, .{ 8, 1 }, 5003, 4, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, false, 512 });

        // Sparse gather: native block sizes, padded slots, and head/batch strides.
        for (fa_kv_types) |t| {
            try list.add(T, .{ 128, 96, 2, .{ 8, 2 }, 5003, 3, true, false, 0, 0, prec_f32, t, t, id, true, false, 257 });
        }
        try list.add(T, .{ 128, 96, 2, .{ 8, 2 }, 5003, 3, true, false, 0, 0, prec_f32, q8_0, type_f16, perm, true, false, 257 });
        try list.add(T, .{ 128, 96, 2, .{ 8, 2 }, 5003, 3, true, false, 0, 0, prec_f32, type_f16, q4_0, perm, true, false, 257 });
        try list.add(T, .{ 128, 96, 2, .{ 8, 2 }, 5003, 3, true, false, 0, 0, prec_f32, q8_0, type_f32, id, true, false, 257 });
        try list.add(T, .{ 128, 96, 2, .{ 8, 2 }, 5003, 3, true, false, 0, 0, prec_f32, type_f32, q8_0, id, true, false, 257 });

        // more V-is-sub-view-of-K cases: other head shapes, and full views with equal head sizes
        try list.add(T, .{ 320, 256, 1, .{ 32, 1 }, 512, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true });
        try list.add(T, .{ 192, 128, 4, .{ 8, 1 }, 512, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true });
        try list.add(T, .{ 128, 128, 8, .{ 4, 1 }, 512, 8, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true });
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 512, 1, true, false, 0, 0, prec_f32, q8_0, q8_0, id, true, true });

        // large-KV F16 cases (Qwen3.6-27B geometry and a llama-class control): the upstream matrix
        // stops at kv=1024, blind to long-context FA bugs (e.g. the oneDNN SDPA ordering race on BMG).
        for ([_]i64{ 4096, 16384 }) |kv| {
            try list.add(T, .{ 256, 256, 4, .{ 6, 1 }, kv, 512, true, false, 0, 0, prec_f32, type_f16, type_f16 });
            try list.add(T, .{ 128, 128, 8, .{ 4, 1 }, kv, 512, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        }

        try list.add(T, .{ 256, 256, 4, .{ 6, 1 }, 512, 512, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 256, 256, 4, .{ 6, 1 }, 4096, 64, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 256, 256, 4, .{ 6, 1 }, 4096, 16, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 256, 256, 4, .{ 2, 1 }, 4096, 512, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 256, 256, 4, .{ 4, 1 }, 4096, 512, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 256, 256, 2, .{ 12, 1 }, 4096, 512, true, false, 0, 0, prec_f32, type_f16, type_f16 });

        // dense-allocated (non-view) quant K/V at batch >= 64, in cache and native layouts
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 512, 75, true, false, 0, 0, prec_f32, q8_0, q8_0, perm, false });
        try list.add(T, .{ 64, 64, 4, .{ 4, 1 }, 512, 75, true, false, 0, 0, prec_f32, q8_0, q8_0, perm, false });
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 1024, 75, true, false, 0, 0, prec_f32, q8_0, q8_0, perm, false });
        try list.add(T, .{ 64, 64, 4, .{ 1, 1 }, 512, 75, true, false, 0, 0, prec_f32, q8_0, q8_0, id, false });

        try list.add(T, .{ 128, 128, 4, .{ 8, 1 }, 1024, 320, true, false, 0, 0, prec_f32, type_f16, type_f16, perm, false });
        try list.add(T, .{ 96, 96, 8, .{ 4, 1 }, 512, 256, true, false, 0, 0, prec_f32, type_f16, type_f16, perm, false });
        try list.add(T, .{ 64, 64, 4, .{ 8, 1 }, 512, 259, true, false, 8.0, 0, prec_f32, type_f16, type_f16, perm, false });
        try list.add(T, .{ 128, 128, 4, .{ 8, 1 }, 512, 288, true, false, 0, 30.0, prec_f32, type_f16, type_f16, perm, false });
        try list.add(T, .{ 128, 128, 4, .{ 8, 1 }, 1024, 320, true, false, 0, 0, prec_f32, type_bf16, type_bf16, perm, false });
        try list.add(T, .{ 64, 64, 4, .{ 8, 1 }, 512, 259, true, false, 8.0, 0, prec_f32, type_bf16, type_bf16, perm, false });

        // FLASH_ATTN_EXT MMA: non-pow2 head size and MLA K/V view.
        try list.add(T, .{ 192, 128, 8, .{ 8, 1 }, 4096, 1, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 576, 512, 1, .{ 20, 1 }, 512, 1, true, false, 0, 0, prec_f32, type_f16, type_f16, id, true, true });

        // FLASH_ATTN_EXT MMA, swizzled K/V tiles, power-of-two stride: nbatch_K2 = 32, 64, 128, 256.
        try list.add(T, .{ 64, 64, 8, .{ 8, 1 }, 4096, 4, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 128, 128, 8, .{ 4, 1 }, 4096, 8, true, true, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 256, 256, 4, .{ 2, 1 }, 1024, 32, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(T, .{ 512, 512, 4, .{ 2, 1 }, 1024, 4, true, false, 0, 0, prec_f32, type_f16, type_f16 });

        // BF16 K/V for the other head sizes of the Metal tensor API kernel
        try list.add(T, .{ 128, 128, 4, .{ 12, 1 }, 512, 75, true, false, 0, 0, prec_f32, type_bf16, type_bf16 });
        try list.add(T, .{ 192, 128, 4, .{ 8, 1 }, 512, 75, true, false, 0, 0, prec_f32, type_bf16, type_bf16 });
        try list.add(T, .{ 512, 512, 4, .{ 4, 1 }, 512, 75, true, false, 0, 0, prec_f32, type_bf16, type_bf16 });
        try list.add(T, .{ 576, 512, 4, .{ 4, 1 }, 512, 75, true, false, 0, 0, prec_f32, type_bf16, type_bf16, id, true, true });

        // FLASH_ATTN_EXT: large logits
        const L = mod.cases.TestFlashAttnExtLargeLogits;
        try list.add(L, .{ 64, 64, 16, .{ 4, 1 }, 1024, 75, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(L, .{ 128, 128, 8, .{ 4, 1 }, 1024, 75, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(L, .{ 256, 256, 4, .{ 4, 1 }, 1024, 75, true, false, 0, 0, prec_f32, type_f16, type_f16 });
        try list.add(L, .{ 256, 256, 4, .{ 4, 1 }, 1024, 75, true, false, 0, 10.0, prec_f32, type_f16, type_f16 });
        try list.add(L, .{ 256, 256, 4, .{ 4, 1 }, 1024, 75, true, false, 0, 0, prec_f32, type_bf16, type_bf16 });

        try list.add(mod.cases.TestCrossEntropyLoss, .{ type_f32, .{ 10, 5, 4, 3 } });
        try list.add(mod.cases.TestCrossEntropyLoss, .{ type_f32, .{ 30000, 1, 1, 1 } });
        try list.add(mod.cases.TestCrossEntropyLossBack, .{ type_f32, .{ 10, 5, 4, 3 } });
        try list.add(mod.cases.TestCrossEntropyLossBack, .{ type_f32, .{ 30000, 1, 1, 1 } });

        try list.add(mod.cases.TestOptStepAdamw, .{ type_f32, .{ 10, 5, 4, 3 } });
        try list.add(mod.cases.TestOptStepSgd, .{ type_f32, .{ 10, 5, 4, 3 } });
    }

    // The K/V types of the flash_attn_ext matrix and sparse gather cases.
    const fa_kv_types = [_]mod.c.enum_ggml_type{
        mod.c.GGML_TYPE_F32,  mod.c.GGML_TYPE_F16,  mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_Q8_0,   mod.c.GGML_TYPE_Q5_1,
        mod.c.GGML_TYPE_Q5_0, mod.c.GGML_TYPE_Q4_1, mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_IQ4_NL,
    };

    // mul_mat_vec_fusion, topk_moe and moe_reduce.
    fn addFusions(list: *mod.CaseList) !void {
        const V = mod.cases.TestMulMatVecFusion;
        const swiglu = mod.c.GGML_GLU_OP_SWIGLU;
        const geglu = mod.c.GGML_GLU_OP_GEGLU;
        const swiglu_clamp = mod.c.GGML_GLU_OP_SWIGLU_CLAMP;

        for (mod.TypeLists.base_types) |t| {
            for ([_]bool{ false, true }) |with_gate| {
                for ([_]bool{ false, true }) |use_id| {
                    for ([_]bool{ false, true }) |b| {
                        if (!use_id and b) continue;
                        for ([_]bool{ false, true }) |with_bias| {
                            if (!with_gate and !with_bias) continue;
                            for ([_]mod.c.enum_ggml_glu_op{ swiglu, geglu, swiglu_clamp }) |glu_op| {
                                if (!with_bias and glu_op == mod.c.GGML_GLU_OP_SWIGLU_OAI) continue;
                                if (!with_gate and glu_op != swiglu) continue;
                                for ([_]bool{ false, true }) |with_lane_scale| {
                                    if (with_lane_scale and t != mod.c.GGML_TYPE_NVFP4 and t != mod.c.GGML_TYPE_F8_E4M3) continue;
                                    try list.add(V, .{ t, glu_op, 1, 32, 256, use_id, 16, 8, b, with_bias, with_gate, with_lane_scale });
                                    try list.add(V, .{ t, glu_op, 1, 32, 256, use_id, 16, 8, b, with_bias, with_gate, with_lane_scale, .{ 1, 1 } });
                                    // multi-token batches (spec decoding)
                                    for ([_]i64{ 2, 4, 8 }) |m_batch| {
                                        try list.add(V, .{ t, glu_op, m_batch, 32, 256, use_id, 16, 8, b, with_bias, with_gate, with_lane_scale, .{ 1, 1 } });
                                    }
                                    // the lane scale does a GET_ROWS on the ids, which does not accept -1
                                    if (use_id and !with_lane_scale) {
                                        try list.add(V, .{ t, glu_op, 8, 32, 256, use_id, 16, 8, b, with_bias, with_gate, with_lane_scale, .{ 1, 1 }, true });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for ([_]bool{ false, true }) |b| {
            try list.add(V, .{ mod.c.GGML_TYPE_IQ2_S, swiglu_clamp, 1, 32, 256, true, 16, 8, b, false, true, false });
        }

        // Gate and up on the SYCL XMX kernels: rows long enough for 2D block loads, a last tile of
        // 8 rows, and columns on both sides of the 8 and 16 column tiles
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_Q2_K, mod.c.GGML_TYPE_Q3_K, mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q5_K, mod.c.GGML_TYPE_Q6_K, mod.c.GGML_TYPE_Q8_0 }) |t| {
            for ([_]mod.c.enum_ggml_glu_op{ swiglu, geglu }) |glu_op| {
                for ([_]i64{ 1, 2, 3, 5, 8, 9, 16, 17 }) |m_batch| {
                    try list.add(V, .{ t, glu_op, m_batch, 1000, 1024, false, 16, 8, false, false, true, false, .{ 1, 1 } });
                }
            }
        }

        for ([_]i64{ 96, 896, 1024, 2880 }) |k| {
            for ([_]mod.c.enum_ggml_glu_op{ swiglu, geglu }) |glu_op| {
                try list.add(V, .{ mod.c.GGML_TYPE_Q8_0, glu_op, 1, 32, k, false, 16, 8, false, false, true, false, .{ 1, 1 } });
            }
        }

        // (upstream's fused row-pair coverage stays commented out: its max_nmse_err() is not
        // estimated correctly, causing sporadic false failures)

        // Both sides of the same row-count boundary as above, on the fused path.
        for ([_]i64{ 6271, 6272, 6273 }) |rows| {
            try list.add(V, .{ mod.c.GGML_TYPE_Q4_K, swiglu, 2, rows, 256, false, 16, 8, false, false, true, false, .{ 1, 1 } });
        }

        // Fused gate/up + GLU on the MMQ path, only used if its tiling is efficient: ~1280 tiles per case, one case per type and GLU op.
        try list.add(V, .{ mod.c.GGML_TYPE_Q4_K, swiglu, 17, 81920, 256, false, 1, 1, false, false, true, false, .{ 1, 1 } });
        try list.add(V, .{ mod.c.GGML_TYPE_Q5_K, geglu, 150, 40960, 256, false, 1, 1, false, false, true, false, .{ 1, 1 } });
        try list.add(V, .{ mod.c.GGML_TYPE_Q6_K, mod.c.GGML_GLU_OP_SWIGLU_OAI, 150, 40960, 256, false, 1, 1, false, false, true, false, .{ 1, 1 } });
        try list.add(V, .{ mod.c.GGML_TYPE_Q8_0, swiglu_clamp, 150, 6912, 256, false, 1, 1, false, false, true, false, .{ 2, 3 } });
        try list.add(V, .{ mod.c.GGML_TYPE_IQ4_XS, swiglu, 150, 40960, 512, false, 1, 1, false, false, true, false, .{ 1, 1 } });
        // The same with experts (~1280 tiles each): 17 tokens with broadcast input (one MMA tile per warp),
        // 2048 tokens without (two, several column tiles per expert).
        try list.add(V, .{ mod.c.GGML_TYPE_Q4_K, swiglu, 17, 5120, 256, true, 16, 8, true });
        try list.add(V, .{ mod.c.GGML_TYPE_Q8_0, swiglu, 2048, 128, 256, true, 40, 8, false });

        const M = mod.cases.TestTopkMoe;
        const Gating = @FieldType(M, "gating_func");
        for ([_]Gating{ .softmax, .sigmoid, .softmaxWeight, .sqrtSoftplus }) |gate| {
            for ([_]bool{ false, true }) |with_norm| {
                for ([_]bool{ false, true }) |bias_probs| {
                    for ([_]f32{ 0.0, 2.0 }) |scale_w| {
                        try list.add(M, .{ .{ 8, 22, 1, 1 }, 4, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 31, 22, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 32, 22, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 40, 22, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 71, 22, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 128, 1, 1, 1 }, 128, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 129, 1, 1, 1 }, 128, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 160, 4, 1, 1 }, 160, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 256, 22, 1, 1 }, 6, with_norm, bias_probs, gate, scale_w }); // Used by DeepSeek-V4
                        try list.add(M, .{ .{ 288, 22, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w }); // Used by StepFun 3.7
                        try list.add(M, .{ .{ 384, 1, 1, 1 }, 6, with_norm, bias_probs, gate, scale_w }); // Used by DeepSeek-V4.1
                        try list.add(M, .{ .{ 384, 8, 1, 1 }, 6, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 384, 22, 1, 1 }, 6, with_norm, bias_probs, gate, scale_w });
                        // rows at and just past the limit where one block still covers all rows
                        try list.add(M, .{ .{ 32, 8, 1, 1 }, 4, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 32, 8, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w });
                        try list.add(M, .{ .{ 32, 9, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w });

                        // grouped experts
                        try list.add(M, .{ .{ 128, 1, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w, 16 });
                        try list.add(M, .{ .{ 256, 22, 1, 1 }, 6, with_norm, bias_probs, gate, scale_w, 32 });
                        try list.add(M, .{ .{ 256, 7, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w, 8 }); // DeepSeek V3
                        try list.add(M, .{ .{ 128, 5, 1, 1 }, 6, with_norm, bias_probs, gate, scale_w, 8 });
                        try list.add(M, .{ .{ 512, 3, 1, 1 }, 8, with_norm, bias_probs, gate, scale_w, 4 });
                    }
                }
            }
        }

        // Cover the supported boundaries, common k = 8 shapes, interleaved views and adds, and k = 16 fallback.
        const R = mod.cases.TestMoeReduce;
        try list.add(R, .{ 63, 2, 17 });
        try list.add(R, .{ 2048, 8, 128 });
        try list.add(R, .{ 2048, 8, 128, false, true });
        try list.add(R, .{ 63, 12, 33, true, true, true });
        try list.add(R, .{ 2048, 15, 40, false, true });
        try list.add(R, .{ 2048, 16, 32, false, true });
    }

    // gated_delta_net and lightning_indexer.
    fn addDeltaNetIndexer(list: *mod.CaseList) !void {
        const G = mod.cases.TestGatedDeltaNet;
        const type_f32 = mod.c.GGML_TYPE_F32;

        try list.add(G, .{ type_f32, 32, 128, 1, 1 });
        try list.add(G, .{ type_f32, 32, 16, 1, 1 });
        try list.add(G, .{ type_f32, 32, 16, 1, 1, 1, true, true });
        try list.add(G, .{ type_f32, 32, 16, 1, 1, 1, false, true });
        try list.add(G, .{ type_f32, 16, 64, 1, 2 });
        try list.add(G, .{ type_f32, 4, 64, 4, 1 });
        try list.add(G, .{ type_f32, 4, 64, 4, 2 });
        try list.add(G, .{ type_f32, 8, 32, 4, 2, 2 });
        try list.add(G, .{ type_f32, 4, 64, 4, 2, 1, true });
        try list.add(G, .{ type_f32, 4, 64, 4, 1, 1, true });
        // KDA (vector gate)
        try list.add(G, .{ type_f32, 4, 64, 1, 1, 1, false, true });
        try list.add(G, .{ type_f32, 4, 64, 1, 2, 1, false, true });
        try list.add(G, .{ type_f32, 4, 16, 1, 2, 1, false, true });
        try list.add(G, .{ type_f32, 4, 32, 4, 1, 1, false, true });
        try list.add(G, .{ type_f32, 4, 64, 4, 2, 1, false, true });
        try list.add(G, .{ type_f32, 8, 32, 4, 2, 2, false, true });
        try list.add(G, .{ type_f32, 4, 64, 4, 2, 1, true, true });
        try list.add(G, .{ type_f32, 4, 16, 4, 2, 1, true, true });
        // chunked path: multi-chunk and non-multiple-of-chunk-size (chunk_size=64 GDN, 16 KDA)
        try list.add(G, .{ type_f32, 4, 64, 64, 1 });
        try list.add(G, .{ type_f32, 4, 64, 127, 1 });
        try list.add(G, .{ type_f32, 4, 64, 256, 1 });
        try list.add(G, .{ type_f32, 4, 64, 65, 1 });
        try list.add(G, .{ type_f32, 4, 64, 100, 1 });
        try list.add(G, .{ type_f32, 4, 64, 200, 1 });
        try list.add(G, .{ type_f32, 4, 64, 127, 2 });
        try list.add(G, .{ type_f32, 4, 64, 64, 1, 1, false, true });
        try list.add(G, .{ type_f32, 4, 64, 33, 1, 1, false, true });
        try list.add(G, .{ type_f32, 4, 64, 100, 1, 1, false, true });

        // MMA-eligible GDN shapes: whole heads, value slices, tails, and the HIP prefill threshold.
        try list.add(G, .{ type_f32, 16, 128, 65, 1, 1 });
        try list.add(G, .{ type_f32, 16, 128, 65, 3, 4 });
        try list.add(G, .{ type_f32, 16, 128, 64, 1, 4 });
        try list.add(G, .{ type_f32, 16, 128, 257, 1, 3 });
        try list.add(G, .{ type_f32, 16, 128, 512, 1, 2 });
        try list.add(G, .{ type_f32, 16, 128, 513, 1, 3 });
        try list.add(G, .{ type_f32, 16, 128, 513, 2, 2, true });
        try list.add(G, .{ type_f32, 16, 128, 2048, 1, 2 });
        // Keep the WebGPU dispatch below 65535 workgroups (32 per token for this shape).
        try list.add(G, .{ type_f32, 16, 128, 2047, 2, 3 });

        // K > 1: output keeps the last min(n_tokens, K) per-token snapshots, ordered most-recent-first
        // (slot 0 = final state, slot s = state s tokens back). Exact-match cases (K == n_seq_tokens):
        try list.add(G, .{ type_f32, 4, 16, 2, 1, 1, false, false, 2 });
        try list.add(G, .{ type_f32, 4, 32, 4, 1, 1, false, false, 4 });
        try list.add(G, .{ type_f32, 4, 64, 4, 2, 1, false, false, 4 });
        try list.add(G, .{ type_f32, 8, 128, 4, 1, 1, false, false, 4 });
        try list.add(G, .{ type_f32, 4, 64, 4, 2, 1, false, true, 4 });
        try list.add(G, .{ type_f32, 8, 32, 4, 2, 2, false, true, 4 });

        // gdn + cache cpy fusion (K > 1)
        const F = mod.cases.TestGatedDeltaNetCacheFusion;
        try list.add(F, .{ type_f32, 4, 32, 2, 1, 2 });
        try list.add(F, .{ type_f32, 4, 64, 4, 1, 2 });
        try list.add(F, .{ type_f32, 4, 32, 4, 1, 4 });
        try list.add(F, .{ type_f32, 8, 32, 4, 2, 4 });
        try list.add(F, .{ type_f32, 4, 32, 8, 1, 4 });

        // (upstream's `#if 0` test_llama / test_falcon cases are disabled to save execution time)

        // lightning_indexer
        const I = mod.cases.TestLightningIndexer;
        for ([_]i64{256}) |kv| {
            for ([_]i64{ 1, 512 }) |bs| {
                for ([_]i64{ 4, 32, 64 }) |nh| {
                    for ([_][2]i64{ .{ 1, 1 }, .{ 4, 4 }, .{ 4, 1 } }) |ns_nm| {
                        for (fa_kv_types) |type_k| {
                            try list.add(I, .{ 128, nh, kv, bs, ns_nm[0], ns_nm[1], type_k });
                        }
                    }
                }
            }
        }

        for ([_]i64{ 1, 7, 8, 63, 64, 65 }) |kv| {
            for (fa_kv_types[0..8]) |type_k| {
                try list.add(I, .{ 128, 64, kv, 32, 4, 1, type_k });
            }
        }
    }
};
