//! Bench case barrel: one struct per `test_*` class of `test-backend-ops.cpp`,
//! plus the bench framework the cases build on.

const std = @import("std");
const bench = @import("../module.zig");

// The framework, re-exported for the cases.
pub const c = bench.c;
pub const config = bench.config;
pub const Graph = bench.Graph;
pub const TensorInit = bench.TensorInit;
pub const Compare = bench.Compare;
pub const Vars = bench.Vars;
pub const Helpers = bench.Helpers;
pub const GgmlType = bench.GgmlType;
pub const GgmlPrec = bench.GgmlPrec;
pub const OpPool = bench.OpPool;
pub const ScaleMode = bench.ScaleMode;

/// The C function type `TestBinBcast` takes (`ggml_add`, `ggml_sub`, `ggml_mul`, `ggml_div`).
pub const BinOp = @import("test_bin_bcast.zig").BinOp;

// The cases.
pub const TestAcc = @import("test_acc.zig").TestAcc;
pub const TestAddAdd = @import("test_add_add.zig").TestAddAdd;
pub const TestAddId = @import("test_add_id.zig").TestAddId;
pub const TestAddRmsNorm = @import("test_add_rms_norm.zig").TestAddRmsNorm;
pub const TestAddUnaryMul = @import("test_add_unary_mul.zig").TestAddUnaryMul;
pub const TestArange = @import("test_arange.zig").TestArange;
pub const TestArgmax = @import("test_argmax.zig").TestArgmax;
pub const TestArgsort = @import("test_argsort.zig").TestArgsort;
pub const TestBinBcast = @import("test_bin_bcast.zig").TestBinBcast;
pub const TestCeil = @import("test_ceil.zig").TestCeil;
pub const TestClamp = @import("test_clamp.zig").TestClamp;
pub const TestCol2im1d = @import("test_col2im_1d.zig").TestCol2im1d;
pub const TestConcat = @import("test_concat.zig").TestConcat;
pub const TestCont = @import("test_cont.zig").TestCont;
pub const TestConv1dDw = @import("test_conv_1d_dw.zig").TestConv1dDw;
pub const TestConv2d = @import("test_conv_2d.zig").TestConv2d;
pub const TestConv2dDw = @import("test_conv_2d_dw.zig").TestConv2dDw;
pub const TestConv3d = @import("test_conv_3d.zig").TestConv3d;
pub const TestConvTranspose1d = @import("test_conv_transpose_1d.zig").TestConvTranspose1d;
pub const TestConvTranspose2d = @import("test_conv_transpose_2d.zig").TestConvTranspose2d;
pub const TestCos = @import("test_cos.zig").TestCos;
pub const TestCountEqual = @import("test_count_equal.zig").TestCountEqual;
pub const TestCpy = @import("test_cpy.zig").TestCpy;
pub const TestCrossEntropyLoss = @import("test_cross_entropy_loss.zig").TestCrossEntropyLoss;
pub const TestCrossEntropyLossBack = @import("test_cross_entropy_loss_back.zig").TestCrossEntropyLossBack;
pub const TestCumsum = @import("test_cumsum.zig").TestCumsum;
pub const TestDiag = @import("test_diag.zig").TestDiag;
pub const TestDiagMaskInf = @import("test_diag_mask_inf.zig").TestDiagMaskInf;
pub const TestDsv4Hc = @import("test_dsv4_hc.zig").TestDsv4Hc;
pub const TestDsv4HcComb = @import("test_dsv4_hc_comb.zig").TestDsv4HcComb;
pub const TestDsv4HcPost = @import("test_dsv4_hc_post.zig").TestDsv4HcPost;
pub const TestDsv4HcPre = @import("test_dsv4_hc_pre.zig").TestDsv4HcPre;
pub const TestDup = @import("test_dup.zig").TestDup;
pub const TestFalcon = @import("test_falcon.zig").TestFalcon;
pub const TestFill = @import("test_fill.zig").TestFill;
pub const TestFlashAttnExt = @import("test_flash_attn_ext.zig").TestFlashAttnExt;
pub const TestFlashAttnExtKvRows = @import("test_flash_attn_ext_kv_rows.zig").TestFlashAttnExtKvRows;
pub const TestFlashAttnExtLargeLogits = @import("test_flash_attn_ext_large_logits.zig").TestFlashAttnExtLargeLogits;
pub const TestFloor = @import("test_floor.zig").TestFloor;
pub const TestGatedDeltaNet = @import("test_gated_delta_net.zig").TestGatedDeltaNet;
pub const TestGatedDeltaNetCacheFusion = @import("test_gated_delta_net_cache_fusion.zig").TestGatedDeltaNetCacheFusion;
pub const TestGenericOp = @import("test_generic_op.zig").TestGenericOp;
pub const TestGetRows = @import("test_get_rows.zig").TestGetRows;
pub const TestGetRowsAfterMulMat = @import("test_get_rows_after_mul_mat.zig").TestGetRowsAfterMulMat;
pub const TestGetRowsBack = @import("test_get_rows_back.zig").TestGetRowsBack;
pub const TestGla = @import("test_gla.zig").TestGla;
pub const TestGlu = @import("test_glu.zig").TestGlu;
pub const TestGluSplit = @import("test_glu_split.zig").TestGluSplit;
pub const TestGroupNorm = @import("test_group_norm.zig").TestGroupNorm;
pub const TestGroupNormMulAdd = @import("test_group_norm_mul_add.zig").TestGroupNormMulAdd;
pub const TestIm2col = @import("test_im2col.zig").TestIm2col;
pub const TestIm2col3d = @import("test_im2col_3d.zig").TestIm2col3d;
pub const TestInterpolate = @import("test_interpolate.zig").TestInterpolate;
pub const TestL2Norm = @import("test_l2_norm.zig").TestL2Norm;
pub const TestL2NormBatch = @import("test_l2_norm_batch.zig").TestL2NormBatch;
pub const TestLeakyRelu = @import("test_leaky_relu.zig").TestLeakyRelu;
pub const TestLightningIndexer = @import("test_lightning_indexer.zig").TestLightningIndexer;
pub const TestLlama = @import("test_llama.zig").TestLlama;
pub const TestLlm = @import("test_llm.zig").TestLlm;
pub const TestLog = @import("test_log.zig").TestLog;
pub const TestMean = @import("test_mean.zig").TestMean;
pub const TestMoeReduce = @import("test_moe_reduce.zig").TestMoeReduce;
pub const TestMulMat = @import("test_mul_mat.zig").TestMulMat;
pub const TestMulMatAdd = @import("test_mul_mat_add.zig").TestMulMatAdd;
pub const TestMulMatHadamard = @import("test_mul_mat_hadamard.zig").TestMulMatHadamard;
pub const TestMulMatId = @import("test_mul_mat_id.zig").TestMulMatId;
pub const TestMulMatIdDup = @import("test_mul_mat_id_dup.zig").TestMulMatIdDup;
pub const TestMulMatIdFusion = @import("test_mul_mat_id_fusion.zig").TestMulMatIdFusion;
pub const TestMulMatIdShared = @import("test_mul_mat_id_shared.zig").TestMulMatIdShared;
pub const TestMulMatIdW4a4 = @import("test_mul_mat_id_w4a4.zig").TestMulMatIdW4a4;
pub const TestMulMatIdW4a8 = @import("test_mul_mat_id_w4a8.zig").TestMulMatIdW4a8;
pub const TestMulMatPos = @import("test_mul_mat_pos.zig").TestMulMatPos;
pub const TestMulMatReorderThenBigBatch = @import("test_mul_mat_reorder_then_big_batch.zig").TestMulMatReorderThenBigBatch;
pub const TestMulMatVecFusion = @import("test_mul_mat_vec_fusion.zig").TestMulMatVecFusion;
pub const TestMulMatW4a4 = @import("test_mul_mat_w4a4.zig").TestMulMatW4a4;
pub const TestMulMatW4a8 = @import("test_mul_mat_w4a8.zig").TestMulMatW4a8;
pub const TestNorm = @import("test_norm.zig").TestNorm;
pub const TestNormMulAdd = @import("test_norm_mul_add.zig").TestNormMulAdd;
pub const TestNormScale = @import("test_norm_scale.zig").TestNormScale;
pub const TestOptStepAdamw = @import("test_opt_step_adamw.zig").TestOptStepAdamw;
pub const TestOptStepSgd = @import("test_opt_step_sgd.zig").TestOptStepSgd;
pub const TestOutProd = @import("test_out_prod.zig").TestOutProd;
pub const TestPad = @import("test_pad.zig").TestPad;
pub const TestPadExt = @import("test_pad_ext.zig").TestPadExt;
pub const TestPadReflect1d = @import("test_pad_reflect_1d.zig").TestPadReflect1d;
pub const TestPool1d = @import("test_pool1d.zig").TestPool1d;
pub const TestPool2d = @import("test_pool2d.zig").TestPool2d;
pub const TestReluSqr = @import("test_relu_sqr.zig").TestReluSqr;
pub const TestRepeat = @import("test_repeat.zig").TestRepeat;
pub const TestRepeatBack = @import("test_repeat_back.zig").TestRepeatBack;
pub const TestRmsNorm = @import("test_rms_norm.zig").TestRmsNorm;
pub const TestRmsNormBack = @import("test_rms_norm_back.zig").TestRmsNormBack;
pub const TestRmsNormMulAdd = @import("test_rms_norm_mul_add.zig").TestRmsNormMulAdd;
pub const TestRmsNormMulRope = @import("test_rms_norm_mul_rope.zig").TestRmsNormMulRope;
pub const TestRoll = @import("test_roll.zig").TestRoll;
pub const TestRope = @import("test_rope.zig").TestRope;
pub const TestRopeSetRows = @import("test_rope_set_rows.zig").TestRopeSetRows;
pub const TestRound = @import("test_round.zig").TestRound;
pub const TestRwkvWkv6 = @import("test_rwkv_wkv6.zig").TestRwkvWkv6;
pub const TestRwkvWkv7 = @import("test_rwkv_wkv7.zig").TestRwkvWkv7;
pub const TestScale = @import("test_scale.zig").TestScale;
pub const TestSet = @import("test_set.zig").TestSet;
pub const TestSetRows = @import("test_set_rows.zig").TestSetRows;
pub const TestSiluBack = @import("test_silu_back.zig").TestSiluBack;
pub const TestSin = @import("test_sin.zig").TestSin;
pub const TestSnakeFuse = @import("test_snake_fuse.zig").TestSnakeFuse;
pub const TestSoftMax = @import("test_soft_max.zig").TestSoftMax;
pub const TestSoftMaxBack = @import("test_soft_max_back.zig").TestSoftMaxBack;
pub const TestSoftcap = @import("test_softcap.zig").TestSoftcap;
pub const TestSolveTri = @import("test_solve_tri.zig").TestSolveTri;
pub const TestSqr = @import("test_sqr.zig").TestSqr;
pub const TestSqrt = @import("test_sqrt.zig").TestSqrt;
pub const TestSsmConv = @import("test_ssm_conv.zig").TestSsmConv;
pub const TestSsmConvBiasSilu = @import("test_ssm_conv_bias_silu.zig").TestSsmConvBiasSilu;
pub const TestSsmScan = @import("test_ssm_scan.zig").TestSsmScan;
pub const TestSsmScanCacheFusion = @import("test_ssm_scan_cache_fusion.zig").TestSsmScanCacheFusion;
pub const TestSsmScanRollback = @import("test_ssm_scan_rollback.zig").TestSsmScanRollback;
pub const TestSum = @import("test_sum.zig").TestSum;
pub const TestSumRows = @import("test_sum_rows.zig").TestSumRows;
pub const TestSwigluClamp = @import("test_swiglu_clamp.zig").TestSwigluClamp;
pub const TestSwigluOai = @import("test_swiglu_oai.zig").TestSwigluOai;
pub const TestTimestepEmbedding = @import("test_timestep_embedding.zig").TestTimestepEmbedding;
pub const TestTopK = @import("test_top_k.zig").TestTopK;
pub const TestTopKInf = @import("test_top_k_inf.zig").TestTopKInf;
pub const TestTopKMask = @import("test_top_k_mask.zig").TestTopKMask;
pub const TestTopkMoe = @import("test_topk_moe.zig").TestTopkMoe;
pub const TestTopkQsa = @import("test_topk_qsa.zig").TestTopkQsa;
pub const TestTri = @import("test_tri.zig").TestTri;
pub const TestTrunc = @import("test_trunc.zig").TestTrunc;
pub const TestUnary = @import("test_unary.zig").TestUnary;
pub const TestUnaryMul = @import("test_unary_mul.zig").TestUnaryMul;
pub const TestUpscale = @import("test_upscale.zig").TestUpscale;
pub const TestXielu = @import("test_xielu.zig").TestXielu;

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "every case compiles against the Case interface" {
    inline for (@typeInfo(@This()).@"struct".decl_names) |name| {
        const T = @field(@This(), name);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"struct" and @hasDecl(T, "buildGraph")) {
            var case: T = .{};
            _ = bench.Case.of(T, &case);
        }
    }
}
