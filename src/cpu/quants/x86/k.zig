//! AVX2 dot products for the K-quant super-block formats.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.9.0,
//! `cd959bae9`), the `__AVX2__` arms of the K-quant dot products. Each
//! function names the C it replaces and the line it began at; the `__AVX__`
//! and scalar arms below each are not compiled at x86-64-v3.
//!
//! # The integer work is exact; the float epilogue is not
//!
//! Every product and sum before `_mm256_cvtepi32_ps` is integer, wrapping
//! or saturating exactly as the instruction does -- `avx2.zig` reaches those
//! instructions. What can move a bit is the float tail, so each kernel's
//! fusions were read from the reference's disassembly, not from the C:
//! `_mm256_fmadd_ps` is explicit, and clang contracts the scalar
//! `summs += dmin * x` into one `vfnmadd231ss`.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../../../quants/blocks.zig");
const avx = @import("avx2.zig");
const common = @import("common.zig");
const c = impl.c;

const f = common.f;
const as = common.as;

comptime {
    @export(&ggmlVecDotQ2KQ8K, .{ .name = "ggml_vec_dot_q2_K_q8_K" });
}

/// Ports `ggml_vec_dot_q2_K_q8_K` (arch/x86/quants.c:1638 @cd959bae9), the
/// `__AVX2__` arm.
///
/// The mins go straight into the float accumulator, eight lanes fused with
/// `dmin` before any weight is read; the weights follow, fused with `d`. Two
/// `fmadd`s into one `acc` per super-block, in that order. The C asserts
/// only `nrc == 1` here, not `n % QK_K == 0`.
pub fn ggmlVecDotQ2KQ8K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q2_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m3 = avx.mm256Set1Epi8(3);
    const m4 = avx.mmSet1Epi8(0xF);

    var acc = avx.mm256SetzeroPs();

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        var q2: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const mins_and_scales = avx.mmLoaduSi128(&x[i].scales);
        const scales8 = avx.mmAndSi128(mins_and_scales, m4);
        const mins8 = avx.mmAndSi128(avx.mmSrliEpi16(mins_and_scales, 4), m4);
        const mins = avx.mm256Cvtepi8Epi16(mins8);
        const prod = avx.mm256MaddEpi16(mins, avx.mm256LoaduSi256(&y[i].bsums));

        acc = avx.mm256FmaddPs(avx.mm256BroadcastSs(&dmin), avx.mm256Cvtepi32Ps(prod), acc);

        const all_scales = avx.mm256Cvtepi8Epi16(scales8);
        const l_scales = avx.mm256Extracti128Si256(all_scales, 0);
        const h_scales = avx.mm256Extracti128Si256(all_scales, 1);
        const scales = [2]avx.m256i{ avx.mm256SetM128i(l_scales, l_scales), avx.mm256SetM128i(h_scales, h_scales) };

        var sumi = avx.mm256SetzeroSi256();

        for (0..c.QK_K / 128) |j| {
            const q2bits = avx.mm256LoaduSi256(q2);
            q2 += 32;

            const q8_0 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_1 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_2 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_3 = avx.mm256LoaduSi256(q8);
            q8 += 32;

            const q2_0 = avx.mm256AndSi256(q2bits, m3);
            const q2_1 = avx.mm256AndSi256(avx.mm256SrliEpi16(q2bits, 2), m3);
            const q2_2 = avx.mm256AndSi256(avx.mm256SrliEpi16(q2bits, 4), m3);
            const q2_3 = avx.mm256AndSi256(avx.mm256SrliEpi16(q2bits, 6), m3);

            var p0 = avx.mm256MaddubsEpi16(q2_0, q8_0);
            var p1 = avx.mm256MaddubsEpi16(q2_1, q8_1);
            var p2 = avx.mm256MaddubsEpi16(q2_2, q8_2);
            var p3 = avx.mm256MaddubsEpi16(q2_3, q8_3);

            p0 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(0)), p0);
            p1 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(1)), p1);
            p2 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(2)), p2);
            p3 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(3)), p3);

            p0 = avx.mm256AddEpi32(p0, p1);
            p2 = avx.mm256AddEpi32(p2, p3);

            sumi = avx.mm256AddEpi32(sumi, avx.mm256AddEpi32(p0, p2));
        }

        acc = avx.mm256FmaddPs(avx.mm256BroadcastSs(&d), avx.mm256Cvtepi32Ps(sumi), acc);
    }

    s[0] = common.hsumFloat8(acc);
}

comptime {
    @export(&ggmlVecDotQ3KQ8K, .{ .name = "ggml_vec_dot_q3_K_q8_K" });
}

/// Ports `ggml_vec_dot_q3_K_q8_K` (arch/x86/quants.c:1830 @cd959bae9), the
/// `__AVX2__` arm.
///
/// Two low bits from `qs` and the third from `hmask`, inverted: the high
/// part is multiplied separately and subtracted, so it is 4 where the bit is
/// clear and 0 where it is set. **`is` is never advanced** in this arm --
/// every 128-weight half takes shuffles 0 to 3 of its own `scales[j]` -- and
/// the port keeps it that way, since it is what the C computes.
pub fn ggmlVecDotQ3KQ8K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    const x = as(blocks.Q3_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m3 = avx.mm256Set1Epi8(3);
    const mone = avx.mm256Set1Epi8(1);
    const m32 = avx.mmSet1Epi8(32);

    var acc = avx.mm256SetzeroPs();

    var aux: [3]u32 = undefined;

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);

        var q3: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        // Set up scales
        @memcpy(std.mem.asBytes(&aux), &x[i].scales);
        var scales128 = avx.mmSetEpi32(
            @bitCast(((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)),
            @bitCast(((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)),
            @bitCast((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)),
            @bitCast((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)),
        );
        scales128 = avx.mmSubEpi8(scales128, m32);
        const all_scales = avx.mm256Cvtepi8Epi16(scales128);
        const l_scales = avx.mm256Extracti128Si256(all_scales, 0);
        const h_scales = avx.mm256Extracti128Si256(all_scales, 1);
        const scales = [2]avx.m256i{ avx.mm256SetM128i(l_scales, l_scales), avx.mm256SetM128i(h_scales, h_scales) };

        // high bit
        const hbits = avx.mm256LoaduSi256(&x[i].hmask);

        // integer accumulator
        var sumi = avx.mm256SetzeroSi256();

        var bit: u32 = 0;
        const is: usize = 0;

        for (0..c.QK_K / 128) |j| {
            // load low 2 bits
            const q3bits = avx.mm256LoaduSi256(q3);
            q3 += 32;

            // prepare low and high bits
            const q3l_0 = avx.mm256AndSi256(q3bits, m3);
            const q3h_0 = avx.mm256SlliEpi16(avx.mm256SrliEpi16(avx.mm256AndnotSi256(hbits, avx.mm256SlliEpi16(mone, bit)), bit), 2);
            bit += 1;

            const q3l_1 = avx.mm256AndSi256(avx.mm256SrliEpi16(q3bits, 2), m3);
            const q3h_1 = avx.mm256SlliEpi16(avx.mm256SrliEpi16(avx.mm256AndnotSi256(hbits, avx.mm256SlliEpi16(mone, bit)), bit), 2);
            bit += 1;

            const q3l_2 = avx.mm256AndSi256(avx.mm256SrliEpi16(q3bits, 4), m3);
            const q3h_2 = avx.mm256SlliEpi16(avx.mm256SrliEpi16(avx.mm256AndnotSi256(hbits, avx.mm256SlliEpi16(mone, bit)), bit), 2);
            bit += 1;

            const q3l_3 = avx.mm256AndSi256(avx.mm256SrliEpi16(q3bits, 6), m3);
            const q3h_3 = avx.mm256SlliEpi16(avx.mm256SrliEpi16(avx.mm256AndnotSi256(hbits, avx.mm256SlliEpi16(mone, bit)), bit), 2);
            bit += 1;

            // load Q8 quants
            const q8_0 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_1 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_2 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_3 = avx.mm256LoaduSi256(q8);
            q8 += 32;

            // The low two bits and the high one are multiplied separately so
            // `maddubs` can take both, and the high part subtracted.
            const q8s_0 = avx.mm256MaddubsEpi16(q3h_0, q8_0);
            const q8s_1 = avx.mm256MaddubsEpi16(q3h_1, q8_1);
            const q8s_2 = avx.mm256MaddubsEpi16(q3h_2, q8_2);
            const q8s_3 = avx.mm256MaddubsEpi16(q3h_3, q8_3);

            var p16_0 = avx.mm256MaddubsEpi16(q3l_0, q8_0);
            var p16_1 = avx.mm256MaddubsEpi16(q3l_1, q8_1);
            var p16_2 = avx.mm256MaddubsEpi16(q3l_2, q8_2);
            var p16_3 = avx.mm256MaddubsEpi16(q3l_3, q8_3);

            p16_0 = avx.mm256SubEpi16(p16_0, q8s_0);
            p16_1 = avx.mm256SubEpi16(p16_1, q8s_1);
            p16_2 = avx.mm256SubEpi16(p16_2, q8s_2);
            p16_3 = avx.mm256SubEpi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(is + 0)), p16_0);
            p16_1 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(is + 1)), p16_1);
            p16_2 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(is + 2)), p16_2);
            p16_3 = avx.mm256MaddEpi16(avx.mm256ShuffleEpi8(scales[j], common.getScaleShuffleQ3k(is + 3)), p16_3);

            // accumulate
            p16_0 = avx.mm256AddEpi32(p16_0, p16_1);
            p16_2 = avx.mm256AddEpi32(p16_2, p16_3);
            sumi = avx.mm256AddEpi32(sumi, avx.mm256AddEpi32(p16_0, p16_2));
        }

        // multiply with block scale and accumulate
        acc = avx.mm256FmaddPs(avx.mm256BroadcastSs(&d), avx.mm256Cvtepi32Ps(sumi), acc);
    }

    s[0] = common.hsumFloat8(acc);
}

comptime {
    @export(&ggmlVecDotQ4KQ8K, .{ .name = "ggml_vec_dot_q4_K_q8_K" });
}

/// Ports `ggml_vec_dot_q4_K_q8_K` (arch/x86/quants.c:2102 @cd959bae9), the
/// `__AVX2__` arm.
///
/// `q5_K` without the fifth bit, and with the mins kept as four lanes,
/// `acc_m`, folded once at the end -- `(m0 + m2) + (m1 + m3)` -- where
/// `q5_K` folds each super-block's to a scalar first.
pub fn ggmlVecDotQ4KQ8K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q4_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m4 = avx.mm256Set1Epi8(0xF);

    var acc = avx.mm256SetzeroPs();
    var acc_m = avx.mmSetzeroPs();

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        const utmp = common.unpackScales(&x[i].scales);

        var q4: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const mins_and_scales = avx.mm256Cvtepu8Epi16(avx.mmSetEpi32(
            @bitCast(utmp[3]),
            @bitCast(utmp[2]),
            @bitCast(utmp[1]),
            @bitCast(utmp[0]),
        ));

        const q8sums = avx.mm256LoaduSi256(&y[i].bsums);
        const q8s = avx.mmHaddEpi16(avx.mm256Extracti128Si256(q8sums, 0), avx.mm256Extracti128Si256(q8sums, 1));
        const prod = avx.mmMaddEpi16(avx.mm256Extracti128Si256(mins_and_scales, 1), q8s);
        acc_m = avx.mmFmaddPs(avx.mmSet1Ps(dmin), avx.mmCvtepi32Ps(prod), acc_m);

        const sc128 = avx.mm256Extracti128Si256(mins_and_scales, 0);
        const scales = avx.mm256SetM128i(sc128, sc128);

        var sumi = avx.mm256SetzeroSi256();

        for (0..c.QK_K / 64) |j| {
            const scale_l = avx.mm256ShuffleEpi8(scales, common.getScaleShuffleK4(2 * j + 0));
            const scale_h = avx.mm256ShuffleEpi8(scales, common.getScaleShuffleK4(2 * j + 1));

            const q4bits = avx.mm256LoaduSi256(q4);
            q4 += 32;
            const q4l = avx.mm256AndSi256(q4bits, m4);
            const q4h = avx.mm256AndSi256(avx.mm256SrliEpi16(q4bits, 4), m4);

            const q8l = avx.mm256LoaduSi256(q8);
            q8 += 32;
            var p16l = avx.mm256MaddubsEpi16(q4l, q8l);
            p16l = avx.mm256MaddEpi16(scale_l, p16l);

            const q8h = avx.mm256LoaduSi256(q8);
            q8 += 32;
            var p16h = avx.mm256MaddubsEpi16(q4h, q8h);
            p16h = avx.mm256MaddEpi16(scale_h, p16h);
            const sumj = avx.mm256AddEpi32(p16l, p16h);

            sumi = avx.mm256AddEpi32(sumi, sumj);
        }

        const vd = avx.mm256Set1Ps(d);
        acc = avx.mm256FmaddPs(vd, avx.mm256Cvtepi32Ps(sumi), acc);
    }

    acc_m = avx.mmAddPs(acc_m, avx.mmMovehlPs(acc_m, acc_m));
    acc_m = avx.mmAddSs(acc_m, avx.mmMovehdupPs(acc_m));

    s[0] = common.hsumFloat8(acc) + avx.mmCvtssF32(acc_m);
}

comptime {
    @export(&ggmlVecDotQ5KQ8K, .{ .name = "ggml_vec_dot_q5_K_q8_K" });
}

/// Ports `ggml_vec_dot_q5_K_q8_K` (arch/x86/quants.c:2280 @cd959bae9), the
/// `__AVX2__` arm.
///
/// The mins are applied first, against `q8_K`'s `bsums`, through two
/// `phadd`s into one scalar per super-block; then four 32-weight groups per
/// half, the fifth bit taken from `qh` one bit plane at a time.
///
/// `summs += dmin * hsum` is one `vfnmadd231ss` in the reference: `dmin` is
/// negated, so `-(y.d * x.dmin) * hsum + summs`, which `@mulAdd` with the
/// negated factor computes to the same bits.
pub fn ggmlVecDotQ5KQ8K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q5_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m4 = avx.mm256Set1Epi8(0xF);
    const mzero = avx.mmSetzeroSi128();
    const mone = avx.mm256Set1Epi8(1);

    var acc = avx.mm256SetzeroPs();

    var summs: f32 = 0;

    for (0..nb) |i| {
        var q5: [*]const u8 = &x[i].qs;
        var q8: [*]const i8 = &y[i].qs;

        const d = y[i].d * f(x[i].d);
        const dmin = -y[i].d * f(x[i].dmin);

        const utmp = common.unpackScales(&x[i].scales);

        const mins_and_scales = avx.mm256Cvtepu8Epi16(avx.mmSetEpi32(
            @bitCast(utmp[3]),
            @bitCast(utmp[2]),
            @bitCast(utmp[1]),
            @bitCast(utmp[0]),
        ));

        const q8sums = avx.mm256LoaduSi256(&y[i].bsums);
        const q8s = avx.mmHaddEpi16(avx.mm256Extracti128Si256(q8sums, 0), avx.mm256Extracti128Si256(q8sums, 1));
        const prod = avx.mmMaddEpi16(avx.mm256Extracti128Si256(mins_and_scales, 1), q8s);
        const hsum = avx.mmHaddEpi32(avx.mmHaddEpi32(prod, mzero), mzero);
        summs = @mulAdd(f32, dmin, @floatFromInt(avx.mmExtractEpi32(hsum, 0)), summs);

        const sc128 = avx.mm256Extracti128Si256(mins_and_scales, 0);
        const scales = avx.mm256SetM128i(sc128, sc128);

        const hbits = avx.mm256LoaduSi256(&x[i].qh);
        var hmask = mone;

        var sumi = avx.mm256SetzeroSi256();

        var bit: u32 = 0;

        for (0..c.QK_K / 64) |j| {
            const scale_0 = avx.mm256ShuffleEpi8(scales, common.getScaleShuffleK4(2 * j + 0));
            const scale_1 = avx.mm256ShuffleEpi8(scales, common.getScaleShuffleK4(2 * j + 1));

            const q5bits = avx.mm256LoaduSi256(q5);
            q5 += 32;

            const q5l_0 = avx.mm256AndSi256(q5bits, m4);
            const q5h_0 = avx.mm256SlliEpi16(avx.mm256SrliEpi16(avx.mm256AndSi256(hbits, hmask), bit), 4);
            bit += 1;
            const q5_0 = avx.mm256AddEpi8(q5l_0, q5h_0);
            hmask = avx.mm256SlliEpi16(hmask, 1);

            const q5l_1 = avx.mm256AndSi256(avx.mm256SrliEpi16(q5bits, 4), m4);
            const q5h_1 = avx.mm256SlliEpi16(avx.mm256SrliEpi16(avx.mm256AndSi256(hbits, hmask), bit), 4);
            bit += 1;
            const q5_1 = avx.mm256AddEpi8(q5l_1, q5h_1);
            hmask = avx.mm256SlliEpi16(hmask, 1);

            const q8_0 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_1 = avx.mm256LoaduSi256(q8);
            q8 += 32;

            var p16_0 = avx.mm256MaddubsEpi16(q5_0, q8_0);
            var p16_1 = avx.mm256MaddubsEpi16(q5_1, q8_1);

            p16_0 = avx.mm256MaddEpi16(scale_0, p16_0);
            p16_1 = avx.mm256MaddEpi16(scale_1, p16_1);

            sumi = avx.mm256AddEpi32(sumi, avx.mm256AddEpi32(p16_0, p16_1));
        }

        const vd = avx.mm256Set1Ps(d);
        acc = avx.mm256FmaddPs(vd, avx.mm256Cvtepi32Ps(sumi), acc);
    }

    s[0] = common.hsumFloat8(acc) + summs;
}

comptime {
    @export(&ggmlVecDotQ6KQ8K, .{ .name = "ggml_vec_dot_q6_K_q8_K" });
}

/// Ports `ggml_vec_dot_q6_K_q8_K` (arch/x86/quants.c:2490 @cd959bae9), the
/// `__AVX2__` arm.
///
/// Six-bit weights, four in `ql` and two in `qh`, with signed `int8`
/// scales. The bias of 32 comes off once per super-block, as `bsums`
/// against the scales shifted left by five, subtracted in integers before
/// the one fused multiply-add.
pub fn ggmlVecDotQ6KQ8K(
    n: c_int,
    s: [*c]f32,
    bs: usize,
    vx: ?*const anyopaque,
    bx: usize,
    vy: ?*const anyopaque,
    by: usize,
    nrc: c_int,
) callconv(.c) void {
    std.debug.assert(@rem(n, c.QK_K) == 0);
    std.debug.assert(nrc == 1);
    _ = .{ bs, bx, by };

    const x = as(blocks.Q6_K, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    const m3 = avx.mm256Set1Epi8(3);
    const m15 = avx.mm256Set1Epi8(15);

    var acc = avx.mm256SetzeroPs();

    for (0..nb) |i| {
        const d = y[i].d * f(x[i].d);

        var q4: [*]const u8 = &x[i].ql;
        var qh: [*]const u8 = &x[i].qh;
        var q8: [*]const i8 = &y[i].qs;

        const q8sums = avx.mm256LoaduSi256(&y[i].bsums);
        const scales = avx.mmLoaduSi128(&x[i].scales);
        const scales_16 = avx.mm256Cvtepi8Epi16(scales);
        const q8sclsub = avx.mm256SlliEpi32(avx.mm256MaddEpi16(q8sums, scales_16), 5);

        var sumi = avx.mm256SetzeroSi256();

        var is: usize = 0;

        for (0..c.QK_K / 128) |_| {
            const q4bits1 = avx.mm256LoaduSi256(q4);
            q4 += 32;
            const q4bits2 = avx.mm256LoaduSi256(q4);
            q4 += 32;
            const q4bitsH = avx.mm256LoaduSi256(qh);
            qh += 32;

            const q4h_0 = avx.mm256SlliEpi16(avx.mm256AndSi256(q4bitsH, m3), 4);
            const q4h_1 = avx.mm256SlliEpi16(avx.mm256AndSi256(q4bitsH, avx.mm256Set1Epi8(12)), 2);
            const q4h_2 = avx.mm256AndSi256(q4bitsH, avx.mm256Set1Epi8(48));
            const q4h_3 = avx.mm256SrliEpi16(avx.mm256AndSi256(q4bitsH, avx.mm256Set1Epi8(-64)), 2);

            const q4_0 = avx.mm256OrSi256(avx.mm256AndSi256(q4bits1, m15), q4h_0);
            const q4_1 = avx.mm256OrSi256(avx.mm256AndSi256(q4bits2, m15), q4h_1);
            const q4_2 = avx.mm256OrSi256(avx.mm256AndSi256(avx.mm256SrliEpi16(q4bits1, 4), m15), q4h_2);
            const q4_3 = avx.mm256OrSi256(avx.mm256AndSi256(avx.mm256SrliEpi16(q4bits2, 4), m15), q4h_3);

            const q8_0 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_1 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_2 = avx.mm256LoaduSi256(q8);
            q8 += 32;
            const q8_3 = avx.mm256LoaduSi256(q8);
            q8 += 32;

            var p16_0 = avx.mm256MaddubsEpi16(q4_0, q8_0);
            var p16_1 = avx.mm256MaddubsEpi16(q4_1, q8_1);
            var p16_2 = avx.mm256MaddubsEpi16(q4_2, q8_2);
            var p16_3 = avx.mm256MaddubsEpi16(q4_3, q8_3);

            const scale_0 = avx.mmShuffleEpi8(scales, common.getScaleShuffle(is + 0));
            const scale_1 = avx.mmShuffleEpi8(scales, common.getScaleShuffle(is + 1));
            const scale_2 = avx.mmShuffleEpi8(scales, common.getScaleShuffle(is + 2));
            const scale_3 = avx.mmShuffleEpi8(scales, common.getScaleShuffle(is + 3));
            is += 4;

            p16_0 = avx.mm256MaddEpi16(avx.mm256Cvtepi8Epi16(scale_0), p16_0);
            p16_1 = avx.mm256MaddEpi16(avx.mm256Cvtepi8Epi16(scale_1), p16_1);
            p16_2 = avx.mm256MaddEpi16(avx.mm256Cvtepi8Epi16(scale_2), p16_2);
            p16_3 = avx.mm256MaddEpi16(avx.mm256Cvtepi8Epi16(scale_3), p16_3);

            sumi = avx.mm256AddEpi32(sumi, avx.mm256AddEpi32(p16_0, p16_1));
            sumi = avx.mm256AddEpi32(sumi, avx.mm256AddEpi32(p16_2, p16_3));
        }

        sumi = avx.mm256SubEpi32(sumi, q8sclsub);
        acc = avx.mm256FmaddPs(avx.mm256BroadcastSs(&d), avx.mm256Cvtepi32Ps(sumi), acc);
    }

    s[0] = common.hsumFloat8(acc);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
