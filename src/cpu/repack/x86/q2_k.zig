//! The AVX2 gemv and gemm for `q2_K` interleaved eight columns at a time.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/repack.cpp` at v0.9.0
//! (`cd959bae9`): `ggml_gemv_q2_K_8x8_q8_K` and `ggml_gemm_q2_K_8x8_q8_K`,
//! their `__AVX2__` arms, and the gemm's `__AVX512BW__ && __AVX512DQ__` arm,
//! 1,480 lines of C that are the same body at 512 bits over two blocks
//! (`wide.zig`).
//!
//! # Eight 2-bit planes a pass
//!
//! Each `sb` takes 128 weights per column: eight 16-weight sub-blocks, `j`,
//! held as 2-bit fields of four registers. Sub-block `j` is the field at bit
//! `2 * (j / 2)` of register pair `j % 2`, two chunks of eight weights each.
//! Its 4-bit scale and 4-bit min share a byte, scales low.
//!
//! # The gemm's two row loops share one body
//!
//! As in `q4_b32.zig` and `q4_k.zig`: the C's two bodies diffed line by
//! line after normalising the row index, identical; written once here.

const std = @import("std");
const qblocks = @import("../../../quants/blocks.zig");
const blocks = @import("../blocks.zig");
const avx = @import("../../quants/x86/avx2.zig");
const common = @import("common.zig");
const wide = @import("wide.zig");

const m256i = avx.m256i;
const m128i = avx.m128i;

const QK_K = 256;

/// The 2-bit field of sub-block `j` from the four raw registers of one
/// column group: chunk `k` of eight weights.
inline fn plane(raw: [4]m256i, comptime j: usize, comptime k: usize) m256i {
    const m3b = avx.mm256Set1Epi8(3);
    const r = raw[(j % 2) * 2 + k];
    const shift = 2 * (j / 2);
    return avx.mm256AndSi256(if (shift == 0) r else avx.mm256SrliEpi16(r, shift), m3b);
}

/// The four scale-and-min registers of one `sb`, and the mins widened.
const ScalesMins = struct {
    scales: [4]m128i,
    mins: [4]m256i,
};

inline fn scalesMins(b: *const blocks.block_q2_Kx8, sb: usize) ScalesMins {
    const m4b_sse = avx.mmSet1Epi8(0xF);
    var r: ScalesMins = undefined;
    inline for (0..4) |p| {
        const mins_and_scales = avx.mmLoaduSi128(b.scales[16 * p + sb * 64 ..].ptr);
        r.scales[p] = avx.mmAndSi128(mins_and_scales, m4b_sse);
        r.mins[p] = avx.mm256Cvtepu8Epi16(avx.mmAndSi128(avx.mmSrliEpi16(mins_and_scales, 4), m4b_sse));
    }
    return r;
}

comptime {
    @export(&ggmlGemvQ2K8x8Q8K, .{ .name = "ggml_gemv_q2_K_8x8_q8_K" });
}

/// Ports `ggml_gemv_q2_K_8x8_q8_K` (arch/x86/repack.cpp:1713 @cd959bae9), the
/// `__AVX2__` arm.
///
/// As `q4_K`'s gemv, the row is stored as `permute(acc_row) - acc_min_rows`
/// at `y * nr`.
pub fn ggmlGemvQ2K8x8Q8K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    // The unrolled sub-block loops evaluate the intrinsics' comptime masks
    // at every call site.
    @setEvalBranchQuota(20000);
    const qk = QK_K;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(@rem(nc, 8) == 0);
    _ = bs;

    const deltamask = avx.mmSetEpi8(.{ 15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0 });
    const finalpermutemask = avx.mm256SetEpi32(.{ 7, 5, 3, 1, 6, 4, 2, 0 });

    const scalemask1 = avx.mmSetEpi8(.{ 14, 14, 6, 6, 12, 12, 4, 4, 10, 10, 2, 2, 8, 8, 0, 0 });
    const scalemask2 = avx.mmSetEpi8(.{ 15, 15, 7, 7, 13, 13, 5, 5, 11, 11, 3, 3, 9, 9, 1, 1 });

    const b_nb: usize = @intCast(@divTrunc(n, QK_K));

    const b_ptr_start: [*]const blocks.block_q2_Kx8 = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const qblocks.Q8_K = @ptrCast(@alignCast(vy));

    for (0..@intCast(nr)) |y| {
        const a_ptr = a_ptr_start + (y * nb);

        for (0..@intCast(@divTrunc(nc, 8))) |x| {
            const b_ptr = b_ptr_start + (x * b_nb);

            var acc_row = avx.mm256SetzeroPs();
            var acc_min_rows = avx.mm256SetzeroPs();

            for (0..nb) |b| {
                const row_scale_f32 = avx.mm256Set1Ps(a_ptr[b].d);
                const col_scale_f32 = common.f32cx8RearrangeLoad(&b_ptr[b].d, deltamask);
                const col_dmin_f32 = common.f32cx8Load(&b_ptr[b].dmin);

                var iacc_b = avx.mm256SetzeroSi256();
                var iacc_min_b = avx.mm256SetzeroSi256();

                for (0..QK_K / 128) |sb| {
                    const qs = b_ptr[b].qs[sb * 256 ..].ptr;
                    var raw0123: [4]m256i = undefined;
                    var raw4567: [4]m256i = undefined;
                    inline for (0..4) |i| {
                        raw0123[i] = avx.mm256LoaduSi256(qs + 64 * i);
                        raw4567[i] = avx.mm256LoaduSi256(qs + 64 * i + 32);
                    }

                    const sm = scalesMins(&b_ptr[b], sb);

                    var iacc_j: [8]m256i = undefined;
                    inline for (0..8) |j| {
                        const scales_j = avx.mm256Cvtepu8Epi16(avx.mmShuffleEpi8(sm.scales[j / 2], if (j % 2 == 0) scalemask1 else scalemask2));
                        const lhs_vec = common.repeat128(avx.mmLoaduSi128(a_ptr[b].qs[16 * j + sb * 128 ..].ptr));
                        var iacc = avx.mm256SetzeroSi256();
                        inline for (0..2) |k| {
                            const r0123 = plane(raw0123, j, k);
                            const r4567 = plane(raw4567, j, k);
                            const imm0: u8 = if (k == 0) 0 else 170;
                            const imm1: u8 = if (k == 0) 85 else 255;
                            iacc = avx.mm256AddEpi16(iacc, avx.mm256MaddubsEpi16(avx.mm256BlendEpi32(r0123, avx.mm256ShuffleEpi32(r4567, 177), 170), avx.mm256ShuffleEpi32(lhs_vec, imm0)));
                            iacc = avx.mm256AddEpi16(iacc, avx.mm256MaddubsEpi16(avx.mm256BlendEpi32(avx.mm256ShuffleEpi32(r0123, 177), r4567, 170), avx.mm256ShuffleEpi32(lhs_vec, imm1)));
                        }
                        iacc_j[j] = avx.mm256MaddEpi16(iacc, scales_j);
                    }

                    const iacc_sb = avx.mm256AddEpi32(
                        avx.mm256AddEpi32(avx.mm256AddEpi32(iacc_j[0], iacc_j[1]), avx.mm256AddEpi32(iacc_j[2], iacc_j[3])),
                        avx.mm256AddEpi32(avx.mm256AddEpi32(iacc_j[4], iacc_j[5]), avx.mm256AddEpi32(iacc_j[6], iacc_j[7])),
                    );

                    const q8s = common.repeat128(avx.mmLoaduSi128(a_ptr[b].bsums[sb * 8 ..].ptr));
                    const iacc_min_sb_01 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi32(q8s, 0), sm.mins[0]);
                    const iacc_min_sb_23 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi32(q8s, 85), sm.mins[1]);
                    const iacc_min_sb_45 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi32(q8s, 170), sm.mins[2]);
                    const iacc_min_sb_67 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi32(q8s, 255), sm.mins[3]);
                    const iacc_min_sb = avx.mm256AddEpi32(avx.mm256AddEpi32(iacc_min_sb_01, iacc_min_sb_23), avx.mm256AddEpi32(iacc_min_sb_45, iacc_min_sb_67));

                    iacc_b = avx.mm256AddEpi32(iacc_b, iacc_sb);
                    iacc_min_b = avx.mm256AddEpi32(iacc_min_b, iacc_min_sb);
                }

                acc_row = avx.mm256FmaddPs(avx.mm256Cvtepi32Ps(iacc_b), avx.mm256MulPs(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = avx.mm256FmaddPs(avx.mm256Cvtepi32Ps(iacc_min_b), avx.mm256MulPs(col_dmin_f32, row_scale_f32), acc_min_rows);
            }

            acc_row = avx.mm256Permutevar8x32Ps(acc_row, finalpermutemask);
            avx.mm256StoreuPs(s + (y * @as(usize, @intCast(nr)) + x * 8), avx.mm256SubPs(acc_row, acc_min_rows));
        }
    }
}

/// One `sb`'s columns for the gemm: every sub-block's two chunks in both
/// shuffle patterns, the scales arranged for each column half, the mins --
/// at 512 bits, two blocks' sixteen columns, the second in the high half.
fn Rhs(comptime L: type) type {
    return struct {
        /// `[j][k]`: sub-block `j`, chunk `k`.
        m0145_sp1: [8][2]L.I,
        m2367_sp1: [8][2]L.I,
        m0145_sp2: [8][2]L.I,
        m2367_sp2: [8][2]L.I,
        scale_0145: [8]L.I,
        scale_2367: [8]L.I,
        mins: [4]L.I,
    };
}

/// One block's 256-bit inputs to the per-`sb` column preparation: the raw
/// 2-bit planes in the formats `B0B1B4B5` and `B2B3B6B7`, every sub-block's
/// scales widened to sixteen bits, and the mins.
const RawRhs = struct {
    raw0145: [4]m256i,
    raw2367: [4]m256i,
    scales: [8]m256i,
    mins: [4]m256i,
};

fn rawRhs(b: *const blocks.block_q2_Kx8, sb: usize) RawRhs {
    // The unrolled sub-block loops evaluate the intrinsics' comptime masks
    // at every call site.
    @setEvalBranchQuota(20000);
    const requiredOrder = avx.mm256SetEpi32(.{ 3, 2, 1, 0, 7, 6, 5, 4 });
    const scalesmask1_sse = avx.mmSetEpi8(.{ 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0 });
    const scalesmask2_sse = avx.mmSetEpi8(.{ 15, 15, 13, 13, 11, 11, 9, 9, 7, 7, 5, 5, 3, 3, 1, 1 });

    const qs = b.qs[sb * 256 ..].ptr;
    var r: RawRhs = undefined;
    inline for (0..4) |i| {
        const raw0123 = avx.mm256LoaduSi256(qs + 64 * i);
        const raw4567 = avx.mm256LoaduSi256(qs + 64 * i + 32);
        r.raw0145[i] = avx.mm256BlendEpi32(raw0123, avx.mm256Permutevar8x32Epi32(raw4567, requiredOrder), 240);
        r.raw2367[i] = avx.mm256BlendEpi32(avx.mm256Permutevar8x32Epi32(raw0123, requiredOrder), raw4567, 240);
    }

    const sm = scalesMins(b, sb);
    r.mins = sm.mins;
    inline for (0..8) |j| {
        r.scales[j] = avx.mm256Cvtepu8Epi16(avx.mmShuffleEpi8(sm.scales[j / 2], if (j % 2 == 0) scalesmask1_sse else scalesmask2_sse));
    }
    return r;
}

/// The 2-bit field of sub-block `j` from the four raw registers of one
/// column group, at `L`'s width: chunk `k` of eight weights.
inline fn planeWide(comptime L: type, raw: [4]L.I, comptime j: usize, comptime k: usize) L.I {
    const m3b = L.set1Epi8(3);
    const r = raw[(j % 2) * 2 + k];
    const shift = 2 * (j / 2);
    return L.andSi(if (shift == 0) r else L.srliEpi16(r, shift), m3b);
}

/// The per-sb column preparation of `ggml_gemm_q2_K_8x8_q8_K`
/// (arch/x86/repack.cpp:3526 @cd959bae9), for `b0` and, at 512 bits, `b1`
/// in the high half.
///
/// The 512-bit arm loads both blocks' scale bytes into one `__m256i` per
/// sub-block pair and widens with `_mm512_cvtepu8_epi16`; its `pshufb` works
/// per 128-bit lane with the mask repeated. Per lane that is each block's
/// AVX2 value, so the scales and mins are `join`s of `rawRhs`.
fn prepareRhs(comptime L: type, b0: *const blocks.block_q2_Kx8, b1: *const blocks.block_q2_Kx8, sb: usize) Rhs(L) {
    @setEvalBranchQuota(20000);
    const r0 = rawRhs(b0, sb);
    const r1 = if (L.width == 512) rawRhs(b1, sb) else r0;

    var raw0145: [4]L.I = undefined;
    var raw2367: [4]L.I = undefined;
    inline for (0..4) |i| {
        raw0145[i] = L.join(r0.raw0145[i], r1.raw0145[i]);
        raw2367[i] = L.join(r0.raw2367[i], r1.raw2367[i]);
    }

    var r: Rhs(L) = undefined;
    inline for (0..8) |j| {
        inline for (0..2) |k| {
            const m0145 = planeWide(L, raw0145, j, k);
            const m2367 = planeWide(L, raw2367, j, k);
            r.m0145_sp1[j][k] = L.shuffleEpi32(m0145, 136);
            r.m2367_sp1[j][k] = L.shuffleEpi32(m2367, 136);
            r.m0145_sp2[j][k] = L.shuffleEpi32(m0145, 221);
            r.m2367_sp2[j][k] = L.shuffleEpi32(m2367, 221);
        }
    }

    inline for (0..4) |p| r.mins[p] = L.join(r0.mins[p], r1.mins[p]);
    inline for (0..8) |j| {
        const scales_j = L.join(r0.scales[j], r1.scales[j]);
        r.scale_0145[j] = L.shuffleEpi32(scales_j, 68);
        r.scale_2367[j] = L.shuffleEpi32(scales_j, 238);
    }
    return r;
}

/// One sub-block's two chunks in one shuffle pattern: two `maddubs`,
/// columns unsigned first, summed in sixteen bits.
inline fn dot2(comptime L: type, rhs: [2]L.I, lhs: [2]L.I) L.I {
    return L.addEpi16(L.maddubsEpi16(rhs[0], lhs[0]), L.maddubsEpi16(rhs[1], lhs[1]));
}

/// The per-group body of `ggml_gemm_q2_K_8x8_q8_K` (arch/x86/repack.cpp:3526
/// @cd959bae9), either arm: one interleaved block of four activation rows
/// against one prepared sb, into four row accumulators and four minima
/// accumulators.
fn accumulate4(comptime L: type, acc: *[4]L.F, acc_min: *[4]L.F, rhs: *const Rhs(L), col_scale_f32: L.F, col_dmin_f32: L.F, a: *const blocks.block_q8_Kx4, sb: usize) void {
    // The unrolled sub-block loops evaluate the intrinsics' comptime masks
    // at every call site.
    @setEvalBranchQuota(40000);
    const qs = a.qs[512 * sb ..].ptr;

    var iacc_mat: [4]L.I = @splat(L.setzero());
    inline for (0..8) |j| {
        var lhs_01_sp1: [2]L.I = undefined;
        var lhs_23_sp1: [2]L.I = undefined;
        var lhs_01_sp2: [2]L.I = undefined;
        var lhs_23_sp2: [2]L.I = undefined;
        inline for (0..2) |k| {
            const lhs_mat_0123 = avx.mm256LoaduSi256(qs + 64 * j + 32 * k);
            const lhs_mat_01 = L.rep(avx.mm256Permute2f128Si256(lhs_mat_0123, lhs_mat_0123, 0));
            const lhs_mat_23 = L.rep(avx.mm256Permute2f128Si256(lhs_mat_0123, lhs_mat_0123, 17));
            lhs_01_sp1[k] = L.shuffleEpi32(lhs_mat_01, 160);
            lhs_23_sp1[k] = L.shuffleEpi32(lhs_mat_23, 160);
            lhs_01_sp2[k] = L.shuffleEpi32(lhs_mat_01, 245);
            lhs_23_sp2[k] = L.shuffleEpi32(lhs_mat_23, 245);
        }
        // Both shuffle patterns summed in sixteen bits, then the scales;
        // the eight sub-blocks' sums are added in 32 bits, which no order
        // changes.
        const m00 = L.maddEpi16(L.addEpi16(dot2(L, rhs.m0145_sp1[j], lhs_01_sp1), dot2(L, rhs.m0145_sp2[j], lhs_01_sp2)), rhs.scale_0145[j]);
        const m01 = L.maddEpi16(L.addEpi16(dot2(L, rhs.m2367_sp1[j], lhs_01_sp1), dot2(L, rhs.m2367_sp2[j], lhs_01_sp2)), rhs.scale_2367[j]);
        const m10 = L.maddEpi16(L.addEpi16(dot2(L, rhs.m0145_sp1[j], lhs_23_sp1), dot2(L, rhs.m0145_sp2[j], lhs_23_sp2)), rhs.scale_0145[j]);
        const m11 = L.maddEpi16(L.addEpi16(dot2(L, rhs.m2367_sp1[j], lhs_23_sp1), dot2(L, rhs.m2367_sp2[j], lhs_23_sp2)), rhs.scale_2367[j]);
        iacc_mat[0] = L.addEpi32(iacc_mat[0], m00);
        iacc_mat[1] = L.addEpi32(iacc_mat[1], m01);
        iacc_mat[2] = L.addEpi32(iacc_mat[2], m10);
        iacc_mat[3] = L.addEpi32(iacc_mat[3], m11);
    }

    // Straighten out to make 4 row vectors
    const iacc_row = [4]L.I{
        L.blendEpi32(iacc_mat[0], L.shuffleEpi32(iacc_mat[1], 78), 204),
        L.blendEpi32(L.shuffleEpi32(iacc_mat[0], 78), iacc_mat[1], 204),
        L.blendEpi32(iacc_mat[2], L.shuffleEpi32(iacc_mat[3], 78), 204),
        L.blendEpi32(L.shuffleEpi32(iacc_mat[2], 78), iacc_mat[3], 204),
    };

    const row_scale_f32_sse = avx.mmLoadPs(&a.d);
    const row_scale_f32 = L.repF(avx.mm256SetM128(row_scale_f32_sse, row_scale_f32_sse));

    const lanes = [4]u8{ 0, 85, 170, 255 };
    inline for (0..4) |i| {
        const rs = L.shufflePs(row_scale_f32, lanes[i]);
        acc[i] = L.fmaddPs(L.cvtepi32Ps(iacc_row[i]), L.mulPs(col_scale_f32, rs), acc[i]);
    }

    // The four rows' bsums, as `[rows 0-1 | rows 2-3][sub-blocks 0-3 | 4-7]`
    const bs = a.bsums[32 * sb ..].ptr;
    const lhs_bsums_01_0123 = L.rep(common.repeat128(avx.mmLoaduSi128(bs)));
    const lhs_bsums_23_0123 = L.rep(common.repeat128(avx.mmLoaduSi128(bs + 8)));
    const lhs_bsums_01_4567 = L.rep(common.repeat128(avx.mmLoaduSi128(bs + 16)));
    const lhs_bsums_23_4567 = L.rep(common.repeat128(avx.mmLoaduSi128(bs + 24)));

    // Row `i` takes lanes `0, 85` (even rows) or `170, 255` (odd rows) of its
    // pair's bsums, against mins 01 and 23, then 45 and 67.
    inline for (0..4) |i| {
        const lo = if (i < 2) lhs_bsums_01_0123 else lhs_bsums_23_0123;
        const hi = if (i < 2) lhs_bsums_01_4567 else lhs_bsums_23_4567;
        const imm_a: u8 = if (i % 2 == 0) 0 else 170;
        const imm_b: u8 = if (i % 2 == 0) 85 else 255;
        const m_01 = L.maddEpi16(L.shuffleEpi32(lo, imm_a), rhs.mins[0]);
        const m_23 = L.maddEpi16(L.shuffleEpi32(lo, imm_b), rhs.mins[1]);
        const m_45 = L.maddEpi16(L.shuffleEpi32(hi, imm_a), rhs.mins[2]);
        const m_67 = L.maddEpi16(L.shuffleEpi32(hi, imm_b), rhs.mins[3]);
        const iacc_row_min = L.addEpi32(L.addEpi32(m_01, m_23), L.addEpi32(m_45, m_67));
        const rs = L.shufflePs(row_scale_f32, lanes[i]);
        acc_min[i] = L.fmaddPs(L.cvtepi32Ps(iacc_row_min), L.mulPs(col_dmin_f32, rs), acc_min[i]);
    }
}

/// A column block's `d` or `dmin` as floats: `GGML_F32Cx8_LOAD`, or
/// `GGML_F32Cx8x2_LOAD` over two blocks at 512 bits.
inline fn colLoad(comptime L: type, d0: anytype, d1: anytype) L.F {
    if (L.width == 256) return common.f32cx8Load(d0);
    return L.joinF(common.f32cx8Load(d0), common.f32cx8Load(d1));
}

/// Both row loops of `ggml_gemm_q2_K_8x8_q8_K` at `L`'s width, over the
/// eight-column blocks `x_begin..x_end`: sixteen rows a pass, then four.
fn rowLoops(comptime L: type, y: *usize, x_begin: usize, x_end: usize, nb: usize, b_nb: usize, nr_u: usize, s: [*]f32, bs: usize, b_ptr_start: [*]const blocks.block_q2_Kx8, a_ptr_start: [*]const blocks.block_q8_Kx4) void {
    const anr = nr_u - nr_u % 16; // Used to align nr with boundary of 16
    const xstep = L.cols / 8;

    while (y.* < anr / 4) : (y.* += 4) {
        var a_ptrs: [4][*]const blocks.block_q8_Kx4 = undefined;

        a_ptrs[0] = a_ptr_start + (y.* * nb);
        for (0..3) |i| a_ptrs[i + 1] = a_ptrs[i] + nb;

        var x = x_begin;
        while (x < x_end) : (x += xstep) {
            const b_ptr_0 = b_ptr_start + (x * b_nb);
            const b_ptr_1 = b_ptr_start + ((x + xstep - 1) * b_nb);

            var acc_rows: [16]L.F = @splat(L.setzeroPs());
            var acc_min_rows: [16]L.F = @splat(L.setzeroPs());

            for (0..nb) |b| {
                const col_scale_f32 = colLoad(L, &b_ptr_0[b].d, &b_ptr_1[b].d);
                const col_dmin_f32 = colLoad(L, &b_ptr_0[b].dmin, &b_ptr_1[b].dmin);

                for (0..QK_K / 128) |sb| {
                    const rhs = prepareRhs(L, &b_ptr_0[b], &b_ptr_1[b], sb);
                    for (0..4) |rp| accumulate4(L, acc_rows[rp * 4 ..][0..4], acc_min_rows[rp * 4 ..][0..4], &rhs, col_scale_f32, col_dmin_f32, &a_ptrs[rp][b], sb);
                }
            }
            for (0..16) |i| L.storeuPs(s + ((y.* * 4 + i) * bs + x * 8), L.subPs(acc_rows[i], acc_min_rows[i]));
        }
    }
    while (y.* < nr_u / 4) : (y.* += 1) {
        const a_ptr = a_ptr_start + (y.* * nb);

        var x = x_begin;
        while (x < x_end) : (x += xstep) {
            const b_ptr_0 = b_ptr_start + (x * b_nb);
            const b_ptr_1 = b_ptr_start + ((x + xstep - 1) * b_nb);

            var acc_rows: [4]L.F = @splat(L.setzeroPs());
            var acc_min_rows: [4]L.F = @splat(L.setzeroPs());

            for (0..nb) |b| {
                const col_scale_f32 = colLoad(L, &b_ptr_0[b].d, &b_ptr_1[b].d);
                const col_dmin_f32 = colLoad(L, &b_ptr_0[b].dmin, &b_ptr_1[b].dmin);

                for (0..QK_K / 128) |sb| {
                    const rhs = prepareRhs(L, &b_ptr_0[b], &b_ptr_1[b], sb);
                    accumulate4(L, &acc_rows, &acc_min_rows, &rhs, col_scale_f32, col_dmin_f32, &a_ptr[b], sb);
                }
            }
            for (0..4) |i| L.storeuPs(s + ((y.* * 4 + i) * bs + x * 8), L.subPs(acc_rows[i], acc_min_rows[i]));
        }
    }
}

comptime {
    @export(&ggmlGemmQ2K8x8Q8K, .{ .name = "ggml_gemm_q2_K_8x8_q8_K" });
}

/// Ports `ggml_gemm_q2_K_8x8_q8_K` (arch/x86/repack.cpp:3526 @cd959bae9).
///
/// With `__AVX512BW__ && __AVX512DQ__` the 512-bit arm, from line 3570,
/// runs first over the columns up to the last multiple of
/// sixteen; the AVX2 loops then take any left, from the first row again.
pub fn ggmlGemmQ2K8x8Q8K(n: c_int, s: [*]f32, bs: usize, vx: *const anyopaque, vy: *const anyopaque, nr: c_int, nc: c_int) callconv(.c) void {
    const qk = QK_K;
    const nb: usize = @intCast(@divTrunc(n, qk));

    std.debug.assert(@rem(n, qk) == 0);
    std.debug.assert(@rem(nr, 4) == 0);
    std.debug.assert(@rem(nc, 8) == 0);

    const b_ptr_start: [*]const blocks.block_q2_Kx8 = @ptrCast(@alignCast(vx));
    const a_ptr_start: [*]const blocks.block_q8_Kx4 = @ptrCast(@alignCast(vy));
    const b_nb: usize = @intCast(@divTrunc(n, QK_K));
    var y: usize = 0;

    var xstart: usize = 0;
    const nr_u: usize = @intCast(nr);
    const nc_u: usize = @intCast(nc);

    if (comptime wide.has_avx512bw_dq) {
        const anc = nc_u - nc_u % 16; // Used to align nc with boundary of 16
        rowLoops(wide.Wide(512), &y, 0, anc / 8, nb, b_nb, nr_u, s, bs, b_ptr_start, a_ptr_start);
        if (anc != nc_u) {
            xstart = anc / 8;
            y = 0;
        }
    }

    rowLoops(wide.Wide(256), &y, xstart, nc_u / 8, nb, b_nb, nr_u, s, bs, b_ptr_start, a_ptr_start);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
