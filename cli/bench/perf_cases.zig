//! Port of `make_test_cases_perf` (tests/test-backend-ops.cpp @cd959bae9).

const mod = @import("module.zig");

/// The perf case list: the shapes `test-backend-ops perf` times.
pub const PerfCases = struct {
    const Self = @This();

    /// Adds every perf case to `list`, in upstream's order.
    ///
    /// - `list`: the destination.
    pub fn build(list: *mod.CaseList) !void {
        for ([_]f32{ 128.0, 4096.0, 32768.0, 1048576.0 }) |n| {
            try list.add(mod.cases.TestArange, .{ mod.c.GGML_TYPE_F32, 0.0, n, 1.0 });
        }

        // delta-net conv input: conv state followed by the transposed qkv of the ubatch
        try list.add(mod.cases.TestConcat, .{ mod.c.GGML_TYPE_F32, .{ 3, 8192, 1, 1 }, 512, 0, 16 });

        // DSV4_HC_PRE: n_tokens 1 is decode, 128 and 512 are prompt batches
        for ([_]i64{ 1, 128, 512 }) |n_tokens| {
            try list.add(mod.cases.TestDsv4HcPre, .{ 4096, 4, n_tokens }); // DeepSeek-V4-Flash
            try list.add(mod.cases.TestDsv4HcPre, .{ 4096, 4, n_tokens, true }); // synthetic, no gated model uses this width
            try list.add(mod.cases.TestDsv4HcPre, .{ 2560, 4, n_tokens, true }); // Qwen4-Exp (Qwen3.8-Flash-Next)
        }

        // DSV4_HC_POST: n_tokens 1 is decode, 128 and 512 are prompt batches
        for ([_]i64{ 1, 128, 512 }) |n_tokens| {
            try list.add(mod.cases.TestDsv4HcPost, .{ 4096, n_tokens }); // DeepSeek-V4-Flash
            try list.add(mod.cases.TestDsv4HcPost, .{ 7168, n_tokens }); // DeepSeek-V4-Pro
            try list.add(mod.cases.TestDsv4HcPost, .{ 2560, n_tokens, true, true }); // Qwen4-Exp, with the post gate chain
            try list.add(mod.cases.TestDsv4HcPost, .{ 2560, n_tokens, true }); // synthetic, Qwen4-Exp kernel without the gate chain
        }

        // Qwen3.8-Flash-Next pooled indexer prefill.
        try list.add(mod.cases.TestLightningIndexer, .{ 128, 4, 1088, 1024, 1, 1, mod.c.GGML_TYPE_F32 });
        try list.add(mod.cases.TestLightningIndexer, .{ 128, 4, 1088, 1024, 1, 1, mod.c.GGML_TYPE_Q4_0 });
        // SWIGLU at a 27B-class FFN width, fused [gate|up] vs split operands
        // note: same bytes either way, so a backend that indexes them differently shows it here
        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_BF16 }) |t| {
            for ([_]i64{ 512, 2048 }) |n_tokens| {
                try list.add(mod.cases.TestGlu, .{ mod.c.GGML_GLU_OP_SWIGLU, t, .{ 2 * 17408, n_tokens, 1, 1 }, 0, false });
                try list.add(mod.cases.TestGluSplit, .{ mod.c.GGML_GLU_OP_SWIGLU, t, .{ 17408, n_tokens, 1, 1 }, 0 });
            }
        }

        // CONT of a 0<->2 permute at DeepSeek-V4 lightning-indexer shapes:
        // indexer_kq is [n_kv, n_tokens, n_head=64] and gets ggml_cont(ggml_permute(.., 2,1,0,3)).
        for ([_]i64{ 1024, 1280, 2048, 2304 }) |n_kv| {
            try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ n_kv, 64, 64, 1 }, false, .{ 2, 1, 0, 3 } });
        }
        for ([_]i64{ 2048, 2304 }) |n_kv| {
            try list.add(mod.cases.TestCont, .{ mod.c.GGML_TYPE_F32, .{ n_kv, 512, 64, 1 }, false, .{ 2, 1, 0, 3 } });
        }

        // POOL_2D max over MiniMax-M3 indexer block scores:
        // sc is [n_ps, n_head=4, n_tokens] and gets ggml_pool_2d(.., MAX, blk=128, 1, blk=128, 1, 0, 0).
        for ([_]i64{ 8192, 32768 }) |n_ps| {
            for ([_]i64{ 1, 512 }) |n_tokens| {
                try list.add(mod.cases.TestPool2d, .{ mod.c.GGML_OP_POOL_MAX, mod.c.GGML_TYPE_F32, .{ n_ps, 4, n_tokens, 1 }, 128, 1, 128, 1, 0, 0 });
            }
        }

        // RMS_NORM and fused variants at decode and prefill shapes
        for ([_]i64{ 1, 512 }) |n_tokens| {
            try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ 4096, n_tokens, 1, 1 } });
            try list.add(mod.cases.TestRmsNorm, .{ mod.c.GGML_TYPE_F32, .{ 128, 32 * n_tokens, 1, 1 } });
            try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 4096, n_tokens, 1, 1 }, 1e-6, false, false, true, mod.c.GGML_ROPE_TYPE_NORMAL, true, false });
            try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 32, n_tokens, 1 }, 1e-6, false, false, true, mod.c.GGML_ROPE_TYPE_NEOX });
            try list.add(mod.cases.TestRmsNormMulRope, .{ .{ 128, 8, n_tokens, 1 }, 1e-6, false, true, true, mod.c.GGML_ROPE_TYPE_NEOX });
        }

        // POOL_1D max over MoE down-projection amax shapes:
        // amax is [1, n_expert, n_tokens] after reducing the whole intermediate dim.
        for ([_]i64{ 2048, 4096, 8192 }) |n_ff| {
            for ([_]i64{ 8, 16 }) |n_expert| {
                for ([_]i64{ 1, 512 }) |n_tokens| {
                    try list.add(mod.cases.TestPool1d, .{ mod.c.GGML_OP_POOL_MAX, mod.c.GGML_TYPE_F32, .{ n_ff, n_expert, n_tokens, 1 }, n_ff, n_ff, 0 });
                }
            }
        }

        // LEAKY_RELU at FFN activation width, for direct comparison with RELU
        for ([_]i64{ 512, 2048 }) |n_tokens| {
            try list.add(mod.cases.TestLeakyRelu, .{ mod.c.GGML_TYPE_F32, .{ 17408, n_tokens, 1, 1 }, 0.1 });
        }

        // Conv2d: K=CRS=NPQ=4096 matmul performance
        // Each row is {IWH, KWH, Cout, Cin, B}.
        const conv2d_cases = [_][5]i64{
            // K=CRS=NPQ=4096 conv2d matmul performance
            .{ 19, 4, 4096, 256, 16 },
            // K=128, CRS=128, NPQ=4096
            .{ 19, 4, 128, 8, 16 },
            // K=130, CRS=128, NPQ=4096
            .{ 19, 4, 130, 8, 16 },
            // Edge case: K x CRS is small
            .{ 19, 2, 4, 4, 16 },
            // A ConvNet's first layer
            .{ 224, 3, 8, 3, 1 },
            // A ConvNet's first layer with 2x2 convolution, and 1 channel
            .{ 224, 2, 8, 1, 1 },
            // A ConvNet's first layer with 2x2 convolution, and 1 channel, several images in the batch
            .{ 224, 2, 8, 1, 8 },
            // A middle layer of a ConvNet
            .{ 58, 3, 64, 32, 1 },
            // A middle layer of a ConvNet, several images in the batch
            .{ 58, 3, 64, 32, 8 },
            // A deep layer of a ConvNet, several images in the batch
            .{ 16, 3, 512, 128, 8 },
            // High resolution output (large NPQ)
            .{ 1536, 3, 64, 32, 1 },
        };
        const iwh_idx = 0;
        const kwh_idx = 1;
        const cout_idx = 2;
        const cin_idx = 3;
        const b_idx = 4;

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
            for (conv2d_cases) |act_case| {
                const ne_input = [4]i64{ act_case[iwh_idx], act_case[iwh_idx], act_case[cin_idx], act_case[b_idx] };
                const ne_kernel = [4]i64{ act_case[kwh_idx], act_case[kwh_idx], act_case[cin_idx], act_case[cout_idx] };
                // Direct CONV_2D
                try list.add(mod.cases.TestConv2d, .{ ne_input, ne_kernel, kernel_type, 1, 1, 0, 0, 1, 1, false }); // cwhn = false
                try list.add(mod.cases.TestConv2d, .{ ne_input, ne_kernel, kernel_type, 1, 1, 0, 0, 1, 1, true }); // cwhn = true
            }
        }

        // Each row is {N, IC, ID, IH, IW, OC, KD, KH, KW, s0, s1, s2, p0, p1, p2, d0, d1, d2}.
        const conv3d_cases = [_][18]i64{
            .{ 1, 320, 8, 38, 26, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            .{ 1, 1280, 8, 38, 26, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            .{ 1, 320, 8, 76, 52, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            .{ 1, 1280, 8, 76, 52, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            .{ 1, 320, 8, 152, 104, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            // Wan VAE pads spatial borders and temporal history before conv3d; operator padding is zero.
            .{ 1, 384, 3, 40, 28, 384, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 192, 4, 78, 54, 192, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 96, 6, 154, 106, 96, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 96, 6, 306, 210, 3, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 1024, 3, 40, 28, 1024, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 256, 6, 154, 106, 256, 3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 384, 3, 38, 26, 768, 3, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            .{ 1, 192, 2, 76, 52, 96, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1 },
            // Upstream's `#if 0` rows (too slow on some devices) are left out.
        };

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
            for (conv3d_cases) |c| {
                try list.add(mod.cases.TestConv3d, .{
                    c[0],  c[1],  c[2],  c[3],        c[4],
                    c[5],  c[6],  c[7],  c[8],        c[9],
                    c[10], c[11], c[12], c[13],       c[14],
                    c[15], c[16], c[17], kernel_type,
                });
            }
        }

        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 4096, 1, 1, 1 }, .{ 1, 1, 1, 1 } });
        try list.add(mod.cases.TestBinBcast, .{ &mod.c.ggml_add, mod.c.GGML_TYPE_F32, .{ 4096, 1, 1, 1 }, .{ 1, 512, 1, 1 } });

        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, .{ 512, 3072, 1, 1 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 8192, 512, 2, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 3072, 512, 2, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 2, 1, 3 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_Q4_0, .{ 8192, 512, 2, 1 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_F32, .{ 8192, 512, 2, 1 } });

        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 768 * 1024, 256, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 }, .{ 0, 0, 0, 0 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 768 * 1024, 256, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 }, .{ 0, 0, 0, 0 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 768, 1024, 256, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 }, .{ 0, 0, 0, 0 } });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_BF16, .{ 768, 1024, 256, 1 }, .{ -1, -1, -1, -1 }, .{ 1, 0, 2, 3 }, .{ 0, 0, 0, 0 } });

        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 768 * 1024, 256, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, .{ 768, 1024, 256, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 768 * 1024, 256, 1, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 768, 1024, 256, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });
        try list.add(mod.cases.TestCpy, .{ mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_BF16, .{ 768, 1024, 256, 1 }, .{ -1, -1, -1, -1 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, true });

        for ([_][4]i64{
            .{ 4096, 4096, 5, 1 },
            .{ 12888, 256, 5, 1 },
            .{ 77, 4096, 5, 1 },
            .{ 1024, 1024, 10, 1 },
            .{ 77, 1024, 10, 1 },
            .{ 256, 256, 20, 1 },
            .{ 64, 64, 20, 1 },
            .{ 77, 64, 20, 1 },
        }) |ne| {
            try list.add(mod.cases.TestSoftMax, .{ mod.c.GGML_TYPE_F32, ne, false, false, mod.c.GGML_TYPE_F32, .{ 1, 1 }, 1.0, 0.0 });
        }

        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 32, 10, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 1024, 10, 1, 1 } });
        try list.add(mod.cases.TestArgmax, .{ mod.c.GGML_TYPE_F32, .{ 32000, 512, 1, 1 } });

        try list.add(mod.cases.TestPadReflect1d, .{ mod.c.GGML_TYPE_F32, .{ 512, 34, 2, 1 } });
        try list.add(mod.cases.TestPadReflect1d, .{ mod.c.GGML_TYPE_F32, .{ 3000, 80, 1, 1 } });
        try list.add(mod.cases.TestPadReflect1d, .{ mod.c.GGML_TYPE_F32, .{ 3000, 80, 4, 1 } });
        try list.add(mod.cases.TestPadReflect1d, .{ mod.c.GGML_TYPE_F32, .{ 3000, 384, 1, 1 } });
        try list.add(mod.cases.TestPadReflect1d, .{ mod.c.GGML_TYPE_F32, .{ 3000, 384, 4, 1 } });

        // SNAKE activation fusion at BigVGAN scale (T=7680 = 24 kHz x 320 ms, C=192)
        try list.add(mod.cases.TestSnakeFuse, .{ mod.c.GGML_TYPE_F32, .{ 7680, 192, 1, 1 } });
        try list.add(mod.cases.TestSnakeFuse, .{ mod.c.GGML_TYPE_F16, .{ 7680, 192, 1, 1 } });
        try list.add(mod.cases.TestSnakeFuse, .{ mod.c.GGML_TYPE_BF16, .{ 7680, 192, 1, 1 } });

        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, 16416, 1, 128, .{ 8, 1 }, .{ 4, 1 }, .{ 0, 2, 1, 3 } });
        try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, 128, 1, 16416, .{ 8, 1 }, .{ 4, 1 }, .{ 0, 1, 2, 3 }, 2 * 16416 });

        // FWHT tests
        for ([_][2]i64{
            .{ 128, 1 },   .{ 64, 1 },     .{ 256, 1 },    .{ 128, 32 },
            .{ 64, 2048 }, .{ 128, 2048 }, .{ 256, 2048 }, .{ 512, 2048 },
        }) |mn| {
            try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, mn[0], mn[1], mn[0] });
        }
        for ([_][2]i64{
            .{ 128, 1 },     .{ 64, 1 },      .{ 256, 1 },     .{ 128, 32 },
            .{ 64, 2048 },   .{ 128, 2048 },  .{ 256, 2048 },  .{ 512, 2048 },
            .{ 1024, 2048 }, .{ 4096, 2048 }, .{ 8192, 2048 },
        }) |mn| {
            try list.add(mod.cases.TestMulMatHadamard, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mn[0], mn[1], mn[0] });
        }

        try list.add(mod.cases.TestSolveTri, .{ mod.c.GGML_TYPE_F32, .{ 64, 64, 4, 4 }, .{ 32, 64, 4, 4 } });
        try list.add(mod.cases.TestSolveTri, .{ mod.c.GGML_TYPE_F32, .{ 128, 128, 4, 2 }, .{ 32, 128, 4, 2 } });
        // qwen3next with CHUNK_SIZE 64
        try list.add(mod.cases.TestSolveTri, .{ mod.c.GGML_TYPE_F32, .{ 64, 64, 8, 32 }, .{ 64, 64, 8, 32 } });
        // qwen3next with CHUNK_SIZE 128
        try list.add(mod.cases.TestSolveTri, .{ mod.c.GGML_TYPE_F32, .{ 128, 128, 4, 32 }, .{ 128, 128, 4, 32 } });
        try list.add(mod.cases.TestSolveTri, .{ mod.c.GGML_TYPE_F32, .{ 256, 256, 4, 2 }, .{ 128, 256, 4, 2 } });

        try list.add(mod.cases.TestTri, .{ mod.c.GGML_TRI_TYPE_LOWER, mod.c.GGML_TYPE_F32, .{ 256, 256, 4, 4 } });
        try list.add(mod.cases.TestTri, .{ mod.c.GGML_TRI_TYPE_UPPER_DIAG, mod.c.GGML_TYPE_F32, .{ 1024, 1024, 8, 4 } });

        try list.add(mod.cases.TestCumsum, .{ mod.c.GGML_TYPE_F32, .{ 128, 128, 4, 4 } });
        try list.add(mod.cases.TestCumsum, .{ mod.c.GGML_TYPE_F32, .{ 2048, 16, 5, 4 } });
        try list.add(mod.cases.TestCumsum, .{ mod.c.GGML_TYPE_F32, .{ 20000, 10, 4, 1 } });

        for ([_]i64{ 1, 2, 3, 4, 5, 8, 512 }) |bs| {
            for (mod.TypeLists.all_types) |type_a| {
                try list.add(mod.cases.TestMulMat, .{ type_a, mod.c.GGML_TYPE_F32, 4096, bs, 14336, .{ 1, 1 }, .{ 1, 1 } });
            }
        }

        // Q4_K multi-column mat-vec
        for ([_]i64{ 4096, 6144, 6272, 14336 }) |m| {
            for ([_]i64{ 1, 2, 3, 4, 8 }) |bs| {
                try list.add(mod.cases.TestMulMat, .{ mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_F32, m, bs, 4096, .{ 1, 1 }, .{ 1, 1 } });
            }
        }

        const moe_types = [_]mod.c.enum_ggml_type{
            mod.c.GGML_TYPE_F32,  mod.c.GGML_TYPE_F16,  mod.c.GGML_TYPE_Q4_0,   mod.c.GGML_TYPE_Q8_0,
            mod.c.GGML_TYPE_Q4_K, mod.c.GGML_TYPE_Q6_K, mod.c.GGML_TYPE_IQ2_XS, mod.c.GGML_TYPE_IQ4_XS,
        };

        // qwen3-30b-a3b
        for ([_]i64{ 1, 4, 8, 32, 64, 128, 256, 512 }) |bs| {
            for (moe_types) |type_a| {
                try list.add(mod.cases.TestMulMatId, .{ type_a, mod.c.GGML_TYPE_F32, 128, 8, false, 768, bs, 2048 });
                try list.add(mod.cases.TestMulMatIdFusion, .{ type_a, mod.c.GGML_TYPE_F32, 128, 8, false, 768, bs, 2048, 1 });
            }
        }

        for ([_]i64{ 1, 4, 8, 32, 64, 128, 256, 512 }) |bs| {
            for (moe_types) |type_a| {
                try list.add(mod.cases.TestMulMatId, .{ type_a, mod.c.GGML_TYPE_F32, 32, 4, false, 1792, bs, 2048 });
                try list.add(mod.cases.TestMulMatIdFusion, .{ type_a, mod.c.GGML_TYPE_F32, 32, 4, false, 1792, bs, 2048, 1 });
            }
        }

        // gpt-oss-20b
        for ([_]i64{ 1, 4, 8, 512 }) |bs| {
            try list.add(mod.cases.TestMulMatId, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 32, 4, false, 2880, bs, 2880 });
            try list.add(mod.cases.TestMulMatIdFusion, .{ mod.c.GGML_TYPE_MXFP4, mod.c.GGML_TYPE_F32, 32, 4, false, 2880, bs, 2880, 1 });
        }

        for ([_]i64{ 3, 5 }) |k| {
            for ([_]i64{ 256, 2560 }) |ic| {
                for ([_]i64{ 32, 64, 256 }) |iw_ih| {
                    if (ic == 2560 and iw_ih == 256) {
                        // too big
                        continue;
                    }
                    try list.add(mod.cases.TestIm2col, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F32, .{ iw_ih, iw_ih, ic, 1 }, .{ k, k, ic, 1 }, 1, 1, 1, 1, 1, 1, true });
                }
            }
        }

        // Qwen3-VL-8B https://github.com/ggml-org/llama.cpp/issues/17012
        try list.add(mod.cases.TestFlashAttnExt, .{ 72, 72, 16, .{ 1, 1 }, 5776, 5776, false, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });

        // Sparse flash attention (n_kv_max hint) decode across KV depths.
        // Shapes: 576/512 DeepSeek MLA, 512/512 DeepSeek-V4/GLM-5.2, 256/256 gqa12 Qwen QSA.
        for ([_]i64{ 4096, 16384, 32768 }) |kv| {
            try list.add(mod.cases.TestFlashAttnExt, .{ 512, 512, 1, .{ 8, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, false, 512 });
            try list.add(mod.cases.TestFlashAttnExt, .{ 576, 512, 1, .{ 16, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, true, 512 });
            try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 12, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, false, 2048 });
            try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 12, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0, .{ 0, 1, 2, 3 }, true, false, 2048 });
            try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 12, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_0, .{ 0, 1, 2, 3 }, true, false, 2048 });
            try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 12, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_BF16, mod.c.GGML_TYPE_BF16, .{ 0, 1, 2, 3 }, true, false, 2048 });
            try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 12, 1 }, kv, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0, .{ 0, 1, 2, 3 }, true, false, 0 });
        }

        try list.add(mod.cases.TestFlashAttnExt, .{ 64, 64, 8, .{ 8, 1 }, 7680, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 64, 64, 8, .{ 8, 1 }, 7680, 4, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 64, 64, 8, .{ 8, 1 }, 7680, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 64, 64, 8, .{ 8, 1 }, 7680, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_Q4_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 64, 64, 8, .{ 8, 1 }, 7680, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 64, 64, 8, .{ 8, 1 }, 7680, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });

        // sparse decode at long context
        try list.add(mod.cases.TestFlashAttnExt, .{ 512, 512, 1, .{ 8, 1 }, 49152, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, false, 0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 512, 512, 1, .{ 8, 1 }, 49152, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, false, 2048 });
        // gemma-4-26b-a4b global-attn layers: head_count_kv=2, 16 query heads (gqa_ratio=8)
        try list.add(mod.cases.TestFlashAttnExt, .{ 512, 512, 2, .{ 8, 1 }, 49152, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, false, 0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 512, 512, 2, .{ 8, 1 }, 49152, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, false, 2048 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 576, 512, 1, .{ 16, 1 }, 49152, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, true, 0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 576, 512, 1, .{ 16, 1 }, 49152, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, true, 2048 });

        // GLM 4.7 Flash gqa20
        try list.add(mod.cases.TestFlashAttnExt, .{ 576, 512, 1, .{ 20, 1 }, 8192, 32, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, true });
        try list.add(mod.cases.TestFlashAttnExt, .{ 576, 512, 1, .{ 20, 1 }, 8192, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, true });

        // q8_0 KV cases with long context (decode and prompt)
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 128, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 512, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 1024, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 2048, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 4096, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 10000, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 20000, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 10000, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 20000, 1, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 10000, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 20000, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q8_0 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 10000, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
        try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 2, .{ 16, 1 }, 20000, 512, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });

        for ([_]i64{ 4096, 16384, 65536, 131072 }) |kv| {
            for ([_]i64{ 1, 512 }) |nb| {
                try list.add(mod.cases.TestFlashAttnExt, .{ 256, 256, 4, .{ 6, 1 }, kv, nb, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16 });
            }
        }

        for ([_]i64{ 4096, 8192, 16384, 32768, 65536 }) |kv| {
            for ([_]i64{ 64, 128, 256, 512, 576 }) |hs| {
                const hsv: i64 = if (hs == 576) 512 else hs;
                const v_view = hs == 576;
                for ([_]i64{ 1, 4, 8 }) |nr| {
                    for ([_]i64{ 1, 4096 }) |nb| {
                        try list.add(mod.cases.TestFlashAttnExt, .{ hs, hsv, 8, .{ nr, 1 }, kv, nb, true, false, 0, 0, mod.c.GGML_PREC_F32, mod.c.GGML_TYPE_F16, mod.c.GGML_TYPE_F16, .{ 0, 1, 2, 3 }, true, v_view });
                    }
                }
            }
        }

        for ([_]i64{ 8192, 16384, 32768, 65536, 131072, 262144, 524288 }) |col| {
            for ([_]i64{ 1, 4, 16 }) |rows| {
                try list.add(mod.cases.TestSoftMax, .{ mod.c.GGML_TYPE_F32, .{ col, rows, 1, 1 }, false, false, mod.c.GGML_TYPE_F32, .{ 1, 1 }, 1.0, 0.0 });
            }
        }

        try list.add(mod.cases.TestConv2dDw, .{ .{ 512, 512, 256, 1 }, .{ 3, 3, 1, 256 }, mod.c.GGML_TYPE_F32, 1, 1, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 512, 512, 256, 1 }, .{ 3, 3, 1, 256 }, mod.c.GGML_TYPE_F32, 1, 1, 1, true });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 112, 112, 32, 1 }, .{ 3, 3, 1, 32 }, mod.c.GGML_TYPE_F32, 1, 1, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 112, 112, 32, 1 }, .{ 3, 3, 1, 32 }, mod.c.GGML_TYPE_F32, 1, 1, 1, true });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 56, 56, 128, 1 }, .{ 5, 5, 1, 128 }, mod.c.GGML_TYPE_F32, 2, 2, 1, false });
        try list.add(mod.cases.TestConv2dDw, .{ .{ 56, 56, 128, 1 }, .{ 5, 5, 1, 128 }, mod.c.GGML_TYPE_F32, 2, 2, 1, true });

        for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |kernel_type| {
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 256, 256, 256, 1 }, .{ 3, 3, 16, 256 }, 1, kernel_type });
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 16, 16, 16, 1 }, .{ 3, 3, 8, 16 }, 1, kernel_type });
            try list.add(mod.cases.TestConvTranspose2d, .{ .{ 10, 10, 9, 1 }, .{ 3, 3, 1, 9 }, 2, kernel_type });
        }

        // Memory bound overlap-add of the GEMM + col2im_1d transposed conv path, real vocoder stage shapes
        try list.add(mod.cases.TestCol2im1d, .{ mod.c.GGML_TYPE_F32, 16, 512, 2048, 8, 0 });
        try list.add(mod.cases.TestCol2im1d, .{ mod.c.GGML_TYPE_F32, 4, 128, 65536, 2, 0 });
        try list.add(mod.cases.TestCol2im1d, .{ mod.c.GGML_TYPE_F16, 16, 512, 2048, 8, 0 });

        try list.add(mod.cases.TestMean, .{ mod.c.GGML_TYPE_F32, .{ 256, 256, 3, 1 } });

        for ([_]i64{ 1, 512 }) |n_token| {
            try list.add(mod.cases.TestAddId, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 2880, 128, 4, n_token });
            try list.add(mod.cases.TestAddId, .{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F32, 2880, 32, 4, n_token });
        }

        for ([_]bool{ true, false }) |fw| { // fw == forward
            for ([_]mod.c.enum_ggml_type{ mod.c.GGML_TYPE_F32, mod.c.GGML_TYPE_F16 }) |t| {
                for ([_]bool{ false, true }) |ff| { // freq_factors
                    // upstream iterates `float v`; the field is an int, so v is 0 or 1 either way
                    for ([_]i32{ 0, 1 }) |v| {
                        try list.add(mod.cases.TestRope, .{ t, .{ 128, 32, 512, 1 }, 128, mod.c.GGML_ROPE_TYPE_NORMAL, 512, 1.0, 0.0, 1.0, ff, v, fw }); // llama 7B
                        try list.add(mod.cases.TestRope, .{ t, .{ 128, 64, 512, 1 }, 128, mod.c.GGML_ROPE_TYPE_NORMAL, 512, 1.0, 0.0, 1.0, ff, v, fw }); // llama 65B
                        try list.add(mod.cases.TestRope, .{ t, .{ 80, 32, 512, 1 }, 20, mod.c.GGML_ROPE_TYPE_NEOX, 512, 1.0, 0.0, 1.0, ff, v, fw }); // neox (stablelm)
                        try list.add(mod.cases.TestRope, .{ t, .{ 64, 8, 512, 1 }, 64, mod.c.GGML_ROPE_TYPE_NEOX, 512, 1.0, 0.0, 1.0, ff, v, fw }); // neox (falcon 40B)
                        try list.add(mod.cases.TestRope, .{ t, .{ 128, 12, 512, 1 }, 128, mod.c.GGML_ROPE_TYPE_MROPE, 512, 1.0, 0.0, 1.0, ff, v, fw }); // rope_multi,m-rope (qwen2vl 2B)
                        try list.add(mod.cases.TestRope, .{ t, .{ 128, 12, 512, 1 }, 128, mod.c.GGML_ROPE_TYPE_IMROPE, 512, 1.0, 0.0, 1.0, ff, v, fw }); // rope_multi,imrope (qwen3vl 2B)
                        try list.add(mod.cases.TestRope, .{ t, .{ 80, 16, 2, 1 }, 80, mod.c.GGML_ROPE_TYPE_VISION, 512, 1.0, 0.0, 1.0, ff, v, fw }); // rope_multi,m-rope (qwen2vl ViT)
                    }
                }
            }
        }

        // Real-model RoPE: F32 forward, packed Q, 512-token prefill.
        try list.add(mod.cases.TestRope, .{ mod.c.GGML_TYPE_F32, .{ 256, 8, 512, 1 }, 64, mod.c.GGML_ROPE_TYPE_IMROPE, 512, 1.0, 0.0, 1.0, false, 0, true }); // qwen3.5 0.8B
        try list.add(mod.cases.TestRope, .{ mod.c.GGML_TYPE_F32, .{ 256, 16, 512, 1 }, 64, mod.c.GGML_ROPE_TYPE_IMROPE, 512, 1.0, 0.0, 1.0, false, 0, true }); // qwen3.5 4B
        try list.add(mod.cases.TestRope, .{ mod.c.GGML_TYPE_F32, .{ 256, 8, 512, 1 }, 256, mod.c.GGML_ROPE_TYPE_NEOX, 512, 1.0, 0.0, 1.0, false, 0, true }); // gemma4 E2B sliding
        try list.add(mod.cases.TestRope, .{ mod.c.GGML_TYPE_F32, .{ 512, 8, 512, 1 }, 128, mod.c.GGML_ROPE_TYPE_NEOX, 512, 1.0, 0.0, 1.0, true, 0, true }); // gemma4 E4B global

        const reduce_rows_cases = [_][4]i64{
            .{ 8192, 1, 1, 1 },
            .{ 8192, 8192, 1, 1 },
            .{ 128, 8192, 1, 1 },
        };

        for (reduce_rows_cases) |it| {
            try list.add(mod.cases.TestMean, .{ mod.c.GGML_TYPE_F32, it });
            try list.add(mod.cases.TestSumRows, .{ mod.c.GGML_TYPE_F32, it });
            try list.add(mod.cases.TestSum, .{ mod.c.GGML_TYPE_F32, it });
        }

        try list.add(mod.cases.TestArgsort, .{ mod.c.GGML_TYPE_F32, .{ 65000, 16, 1, 1 } });
        try list.add(mod.cases.TestArgsort, .{ mod.c.GGML_TYPE_F32, .{ 200000, 1, 1, 1 } });
        try list.add(mod.cases.TestArgsort, .{ mod.c.GGML_TYPE_F32, .{ 200000, 16, 1, 1 } });

        try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ 2, 1, 1, 1 }, 1 });
        // widths around the tiling threshold
        for ([_]i64{ 4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072 }) |cols| {
            for ([_]i64{ 1, 16 }) |nrows| {
                try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ cols, nrows, 1, 1 }, 16 });
            }
        }
        for ([_]i64{ 1, 4, 8, 10, 16, 32, 40, 400 }) |k| {
            for ([_]i64{ 1, 16 }) |nrows| {
                for ([_]i64{ k, 1000, 65000, 200000 }) |cols| {
                    try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ cols, nrows, 1, 1 }, k });
                }
            }
        }

        // qwen4exp sparse-attention indexer: nrows = n_tokens/n_stream, so tg gives nrows==1.
        // Sweep nrows to expose how much of the device a single row leaves idle.
        for ([_]i64{ 8192, 32768, 131072 }) |cols| {
            for ([_]i64{ 1, 2, 4, 8, 16, 32 }) |nrows| {
                try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ cols, nrows, 1, 1 }, 2048 });
            }
        }
        // bitonic vs radix crossover
        for ([_]i64{ 520, 1000, 2048, 3000, 4096 }) |cols| {
            for ([_]i64{ 2, 16, 32, 48, 64, 128, 256, 512 }) |nrows| {
                try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ cols, nrows, 1, 1 }, 16 });
            }
        }
        // backend sampler: one row of the vocab (llama-sampler.cpp top_k)
        for ([_]i32{ 20, 40 }) |k| {
            try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ 151936, 1, 1, 1 }, k });
        }

        // short rows, many of them: MoE routing and group selection. The opposite corner from
        // the indexer, and the one where a work-group per row is the wasteful choice.
        for ([_]i64{ 2, 16, 128, 1024 }) |cols| {
            for ([_]i64{ 1024, 8192 }) |nrows| {
                for ([_]i64{ 1, 2, 8, 16, 32 }) |k| {
                    if (k <= cols) {
                        try list.add(mod.cases.TestTopK, .{ mod.c.GGML_TYPE_F32, .{ cols, nrows, 1, 1 }, k });
                    }
                }
            }
        }

        for ([_]i64{ 1, 4, 8, 16 }) |nrows| {
            for ([_]i64{ 128, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000, 2000000 }) |cols| {
                try list.add(mod.cases.TestCumsum, .{ mod.c.GGML_TYPE_F32, .{ cols, nrows, 1, 1 } });
            }
        }

        // Examples from granite-4.0-h-1b/ggml-model-Q8_0.gguf
        try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ 515, 3328, 1, 1 }, .{ 4, 3328, 1, 1 } }); // prefill
        try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ 937, 8192, 1, 1 }, .{ 4, 8192, 1, 1 } }); // prefill
        try list.add(mod.cases.TestSsmConv, .{ mod.c.GGML_TYPE_F32, .{ 4, 3328, 1, 1 }, .{ 4, 3328, 1, 1 } }); // generate
        try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ 515, 3328, 1, 1 }, .{ 4, 3328, 1, 1 }, true }); // prefill
        try list.add(mod.cases.TestSsmConvBiasSilu, .{ mod.c.GGML_TYPE_F32, .{ 4, 3328, 1, 1 }, .{ 4, 3328, 1, 1 }, true }); // generate
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 48, 1, 512, 1 }); // prefill
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 64, 48, 1, 1, 1 }); // generate
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 80, 128, 1, 512, 1 }); // Nemotron-9B prefill
        try list.add(mod.cases.TestSsmScan, .{ mod.c.GGML_TYPE_F32, 128, 80, 128, 1, 1, 1 }); // Nemotron-9B generate

        // acc
        try list.add(mod.cases.TestAcc, .{ mod.c.GGML_TYPE_F32, .{ 256, 17, 1, 1 }, .{ 256, 16, 1, 1 }, -1 });
        try list.add(mod.cases.TestAcc, .{ mod.c.GGML_TYPE_F32, .{ 256, 17, 2, 3 }, .{ 256, 16, 2, 3 }, -1 });
        try list.add(mod.cases.TestAcc, .{ mod.c.GGML_TYPE_F32, .{ 256, 17, 2, 3 }, .{ 128, 16, 2, 3 }, -1 });
        try list.add(mod.cases.TestAcc, .{ mod.c.GGML_TYPE_F32, .{ 256, 17, 2, 3 }, .{ 256, 16, 2, 3 }, 1 });
        try list.add(mod.cases.TestAcc, .{ mod.c.GGML_TYPE_F32, .{ 256, 17, 2, 3 }, .{ 128, 16, 2, 3 }, 2 });
        try list.add(mod.cases.TestAcc, .{ mod.c.GGML_TYPE_F32, .{ 256, 17, 2, 3 }, .{ 64, 16, 2, 3 }, 3 });

        // GATED_DELTA_NET: realistic model configurations
        // TG: n_seq_tokens=1 (autoregressive)
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 1, 1 }); // Qwen3.5-like: 32 heads, d=128
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 16, 64, 1, 1 }); // smaller model
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 1, 1, 1, false, true }); // KDA
        // PP: n_seq_tokens=64,256 (prompt processing)
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 64, 1 }); // PP-64
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 256, 1 }); // PP-256
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 512, 1 }); // PP-512
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 1024, 1 }); // PP-1024
        // Small model configs (fewer heads = less GPU occupancy for autoregressive)
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 4, 128, 64, 1 }); // 4h PP-64
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 4, 128, 256, 1 }); // 4h PP-256
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 4, 128, 512, 1 }); // 4h PP-512
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 4, 128, 1024, 1 }); // 4h PP-1024
        try list.add(mod.cases.TestGatedDeltaNet, .{ mod.c.GGML_TYPE_F32, 32, 128, 64, 1, 1, false, true }); // KDA PP-64

        // lightning_indexer
        for ([_]i64{ 256, 4096, 65536 }) |kv| {
            for ([_]i64{ 1, 512, 2048 }) |bs| {
                for ([_]i64{ 4, 32, 64 }) |nh| {
                    for ([_]i64{ 1, 4 }) |ns| {
                        for ([_]mod.c.enum_ggml_type{
                            mod.c.GGML_TYPE_F32,  mod.c.GGML_TYPE_F16,  mod.c.GGML_TYPE_BF16,
                            mod.c.GGML_TYPE_Q8_0, mod.c.GGML_TYPE_Q5_1, mod.c.GGML_TYPE_Q5_0,
                            mod.c.GGML_TYPE_Q4_1, mod.c.GGML_TYPE_Q4_0, mod.c.GGML_TYPE_IQ4_NL,
                        }) |type_k| {
                            try list.add(mod.cases.TestLightningIndexer, .{ 128, nh, kv, bs, ns, ns, type_k });
                        }
                    }
                }
            }
        }

        // top_k_mask, Flash-Next QSA shapes: 512 pools of 4 + a 3-cell tail, and DSA shapes: 2048 cells
        for ([_]i64{ 18432, 131072 }) |n_kv| {
            for ([_]i64{ 1, 2048 }) |nb| {
                try list.add(mod.cases.TestTopKMask, .{ mod.c.GGML_TYPE_F16, n_kv, 4, @divTrunc(n_kv, 4), 512, 3, true, nb, 1, 1, true });
                try list.add(mod.cases.TestTopKMask, .{ mod.c.GGML_TYPE_F16, n_kv, 0, 0, 2048, 0, false, nb, 1, 1, true });
            }
        }

        // launch-overhead isolation: single L2_NORM launch vs batched siblings at the GDN
        // production shape (strided qkv views) -- perf-mode only, the eval list has its own
        // 2/4-wide coverage
        for ([_]i64{ 128, 256 }) |n| {
            try list.add(mod.cases.TestL2Norm, .{ mod.c.GGML_TYPE_F32, .{ n, 16, 16, 1 }, 1e-12, false, false });
            try list.add(mod.cases.TestL2NormBatch, .{ mod.c.GGML_TYPE_F32, .{ n, 16, 16, 1 }, 2, 1e-12, true });
            try list.add(mod.cases.TestL2NormBatch, .{ mod.c.GGML_TYPE_F32, .{ n, 16, 16, 1 }, 4, 1e-12, true });
        }
    }
};
