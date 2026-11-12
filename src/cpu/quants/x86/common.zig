//! The static helpers `arch/x86/quants.c` shares between its kernels.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.9.0,
//! `cd959bae9`): the `__AVX2__` arms of the helpers at the top of the file
//! and the scale shuffles before the dot products. Each names the C it
//! replaces and the line it began at. At x86-64-v3 and v4 neither AVX-VNNI
//! nor AVX512-VNNI is defined, so `mul_sum_us8_pairs_float` and
//! `mul_sum_i8_pairs_float` take their plain `maddubs` arms; `tigerlake` has
//! AVX512-VNNI, and the first takes its `vpdpbusd` arm.

const std = @import("std");
const convert = @import("../../convert.zig");
const avx = @import("avx2.zig");

const m256i = avx.m256i;
const m128i = avx.m128i;
const m256 = avx.m256;

/// `GGML_CPU_FP16_TO_FP32`: at x86-64-v3 the table lookup, as the reference
/// compiles it.
pub inline fn f(h: u16) f32 {
    return convert.cpuFp16ToFp32(h);
}

/// A kernel's `vx`/`vy` as the block array it is.
pub inline fn as(comptime Block: type, p: ?*const anyopaque) [*]const Block {
    return @ptrCast(@alignCast(p.?));
}

/// Ports `hsum_float_8` (arch/x86/quants.c:43 @cd959bae9).
///
/// `((x0+x4) + (x2+x6)) + ((x1+x5) + (x3+x7))`: the halves summed, then
/// `movehl` and `movehdup` fold the four lanes pairwise. The order is the
/// answer's last bit.
pub inline fn hsumFloat8(x: m256) f32 {
    var res = avx.mm256Extractf128Ps(x, 1);
    res = avx.mmAddPs(res, avx.mm256Castps256Ps128(x));
    res = avx.mmAddPs(res, avx.mmMovehlPs(res, res));
    res = avx.mmAddSs(res, avx.mmMovehdupPs(res));
    return avx.mmCvtssF32(res);
}

/// Ports `get_scale_shuffle_q3k` (arch/x86/quants.c:518 @cd959bae9): the
/// control that broadcasts scale pair `4i, 4i+1` across the low 128-bit half
/// and `4i+2, 4i+3` across the high one.
pub inline fn getScaleShuffleQ3k(i: usize) m256i {
    const k_shuffle: [128]u8 = comptime blk: {
        var t: [128]u8 = undefined;
        for (0..8) |u| {
            for (0..8) |j| {
                t[16 * u + 2 * j] = 2 * u;
                t[16 * u + 2 * j + 1] = 2 * u + 1;
            }
        }
        break :blk t;
    };
    return avx.mm256LoaduSi256(k_shuffle[32 * i ..].ptr);
}

/// Ports `get_scale_shuffle_k4` (arch/x86/quants.c:527 @cd959bae9): the
/// `pshufb` control that broadcasts scale pair `i` -- bytes `2i` and `2i+1`
/// of a 16-bit scale vector -- across all 32 bytes.
pub inline fn getScaleShuffleK4(i: usize) m256i {
    const k_shuffle: [256]u8 = comptime blk: {
        var t: [256]u8 = undefined;
        for (0..8) |r| {
            for (0..16) |j| {
                t[32 * r + 2 * j] = 2 * r;
                t[32 * r + 2 * j + 1] = 2 * r + 1;
            }
        }
        break :blk t;
    };
    return avx.mm256LoaduSi256(k_shuffle[32 * i ..].ptr);
}

/// Ports `hsum_i32_8` (arch/x86/quants.c:52 @cd959bae9): the halves summed,
/// then the high pair onto the low, then lane 1 onto lane 0. Integer, so the
/// order cannot change the answer, but it is the C's.
pub inline fn hsumI328(a: m256i) i32 {
    const sum128 = avx.mmAddEpi32(avx.mm256Castsi256Si128(a), avx.mm256_extractf128_si256(a, 1));
    const hi64 = avx.mmUnpackhiEpi64(sum128, sum128);
    const sum64 = avx.mmAddEpi32(hi64, sum128);
    const hi32 = avx.mmShuffleEpi32(sum64, avx.mmShuffle(2, 3, 0, 1));
    return avx.mmCvtsi128Si32(avx.mmAddEpi32(sum64, hi32));
}

/// Ports `mul_add_epi8` (arch/x86/quants.c:69 @cd959bae9): signed bytes times
/// signed bytes, pairwise into `i16`, by moving `x`'s sign onto `y` so that
/// `maddubs` can take `|x|` as its unsigned operand.
pub inline fn mulAddEpi8(x: m256i, y: m256i) m256i {
    const ax = avx.mm256SignEpi8(x, x);
    const sy = avx.mm256SignEpi8(y, x);
    return avx.mm256MaddubsEpi16(ax, sy);
}

/// Ports `bytes_from_bits_32` (arch/x86/quants.c:76 @cd959bae9): 32 bits to
/// 32 bytes, each `0xFF` where its bit is set and `0x00` where it is not.
pub inline fn bytesFromBits32(x: *const [4]u8) m256i {
    const x32 = std.mem.readInt(u32, x, .little);
    const shuf_mask = avx.mm256SetEpi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    var bytes = avx.mm256ShuffleEpi8(avx.mm256Set1Epi32(@bitCast(x32)), shuf_mask);
    const bit_mask = avx.mm256Set1Epi64x(0x7fbfdfeff7fbfdfe);
    bytes = avx.mm256OrSi256(bytes, bit_mask);
    return avx.mm256CmpeqEpi8(bytes, avx.mm256Set1Epi64x(-1));
}

/// Ports `bytes_from_nibbles_32` (arch/x86/quants.c:90 @cd959bae9): sixteen
/// bytes to 32 nibbles in `[0, 15]`, the low nibbles first.
pub inline fn bytesFromNibbles32(rsi: *const [16]u8) m256i {
    const tmp = avx.mmLoaduSi128(rsi);
    const bytes = avx.mm256SetM128i(avx.mmSrliEpi16(tmp, 4), tmp);
    const lowMask = avx.mm256Set1Epi8(0xF);
    return avx.mm256AndSi256(lowMask, bytes);
}

/// Ports `sum_i16_pairs_float` (arch/x86/quants.c:99 @cd959bae9).
pub inline fn sumI16PairsFloat(x: m256i) m256 {
    const ones = avx.mm256Set1Epi16(1);
    const summed_pairs = avx.mm256MaddEpi16(ones, x);
    return avx.mm256Cvtepi32Ps(summed_pairs);
}

/// Ports `mul_sum_us8_pairs_float` (arch/x86/quants.c:105 @cd959bae9): the
/// `__AVX512VNNI__ && __AVX512VL__` arm where the target has it, the plain
/// arm otherwise -- neither VNNI is in x86-64-v3 or v4.
pub inline fn mulSumUs8PairsFloat(ax: m256i, sy: m256i) m256 {
    if (comptime avx.has_avx512vnni_vl) {
        const zero = avx.mm256SetzeroSi256();
        const summed_pairs = avx.mm256DpbusdEpi32(zero, ax, sy);
        return avx.mm256Cvtepi32Ps(summed_pairs);
    }
    if (comptime avx.has_avxvnni_only) @compileError("the __AVXVNNI__ arm of mul_sum_us8_pairs_float is not ported");
    const dot = avx.mm256MaddubsEpi16(ax, sy);
    return sumI16PairsFloat(dot);
}

/// Ports `mul_sum_i8_pairs_float` (arch/x86/quants.c:122 @cd959bae9), the
/// arm without `__AVXVNNIINT8__`.
pub inline fn mulSumI8PairsFloat(x: m256i, y: m256i) m256 {
    if (comptime avx.has_avxvnniint8) @compileError("the __AVXVNNIINT8__ arm of mul_sum_i8_pairs_float is not ported");
    const ax = avx.mm256SignEpi8(x, x);
    const sy = avx.mm256SignEpi8(y, x);
    return mulSumUs8PairsFloat(ax, sy);
}

/// Ports `get_scale_shuffle` (arch/x86/quants.c:540 @cd959bae9): the
/// 128-bit control that broadcasts bytes `2i` and `2i+1` eight times each.
pub inline fn getScaleShuffle(i: usize) m128i {
    const k_shuffle: [128]u8 = comptime blk: {
        var t: [128]u8 = undefined;
        for (0..16) |r| {
            for (0..8) |j| t[8 * r + j] = r;
        }
        break :blk t;
    };
    return avx.mmLoaduSi128(k_shuffle[16 * i ..].ptr);
}

/// The scale shuffle that opens each super-block of
/// `ggml_vec_dot_q4_K_q8_K` and `ggml_vec_dot_q5_K_q8_K` (arch/x86/quants.c:2102, 2280
/// @cd959bae9): twelve packed bytes of 6-bit scales and mins to four words,
/// scales in the first two and mins in the last two.
pub inline fn unpackScales(packed_scales: *const [12]u8) [4]u32 {
    const kmask1: u32 = 0x3f3f3f3f;
    const kmask2: u32 = 0x0f0f0f0f;
    const kmask3: u32 = 0x03030303;

    var utmp: [4]u32 = undefined;
    @memcpy(std.mem.asBytes(&utmp)[0..12], packed_scales);

    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    return utmp;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
