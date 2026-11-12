//! AVX2 dot products for the ternary formats, `tq1_0` and `tq2_0`.
//!
//! # Provenance
//!
//! Ported from `llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (v0.9.0,
//! `cd959bae9`), the `__AVX2__` arms at lines 1376 and 1508. Each function
//! names the C it replaces and the line it began at.
//!
//! # Unfused, unlike the rest of the file
//!
//! Both accumulate with `_mm256_add_ps(_mm256_mul_ps(..), sumf)`: two
//! intrinsics, so two roundings. The reference compiles them to `vmulps` and
//! `vaddps`, and the port keeps them apart.
//!
//! # The weights are stored as `{0, 1, 2}`
//!
//! The kernels take the dot product of that against `q8_K` and subtract
//! `bsums` once at the end, which shifts every weight to `{-1, 0, 1}` in
//! one integer step. The partial sums stay sixteen-bit throughout.

const std = @import("std");
const impl = @import("../../../impl.zig");
const blocks = @import("../../../quants/blocks.zig");
const avx = @import("avx2.zig");
const common = @import("common.zig");
const c = impl.c;

const f = common.f;
const as = common.as;

/// The base-3 digit extraction `ggml_vec_dot_tq1_0_q8_K` applies to each
/// register (arch/x86/quants.c:1440 @cd959bae9): cancel the +1 that the
/// average adds so it acts as a halving add, multiply by 3 through two
/// averages, and keep the top two bits.
inline fn topTrit(q: avx.m256i) avx.m256i {
    var r = avx.mm256SubsEpu8(q, avx.mm256Set1Epi8(1));
    r = avx.mm256AvgEpu8(r, avx.mm256AvgEpu8(r, avx.mm256SetzeroSi256()));
    return avx.mm256AndSi256(avx.mm256SrliEpi16(r, 6), avx.mm256Set1Epi8(3));
}

/// Times nine on bytes, as `ggml_vec_dot_tq1_0_q8_K` writes it
/// (arch/x86/quants.c:1440 @cd959bae9): a shift by three masked to the
/// byte, plus the value.
inline fn times9(q: avx.m256i) avx.m256i {
    return avx.mm256AddEpi8(avx.mm256AndSi256(avx.mm256SlliEpi16(q, 3), avx.mm256Set1Epi8(-8)), q);
}

inline fn times9128(q: avx.m128i) avx.m128i {
    return avx.mmAddEpi8(avx.mmAndSi128(avx.mmSlliEpi16(q, 3), avx.mmSet1Epi8(-8)), q);
}

comptime {
    @export(&ggmlVecDotTq10Q8K, .{ .name = "ggml_vec_dot_tq1_0_q8_K" });
}

/// Ports `ggml_vec_dot_tq1_0_q8_K` (arch/x86/quants.c:1440 @cd959bae9), the
/// `__AVX2__` arm.
///
/// Five trits a byte, extracted by multiplying by powers of three and taking
/// the top two bits; then four more a byte from `qh`, multiplied in sixteen
/// bits because AVX2 has no byte multiply.
pub fn ggmlVecDotTq10Q8K(
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

    const x = as(blocks.TQ1_0, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    var sumf = avx.mm256SetzeroPs();

    for (0..nb) |i| {
        // 16-bit sums
        var sumi0 = avx.mm256SetzeroSi256();
        var sumi1 = avx.mm256SetzeroSi256();
        var sumi2 = avx.mm256SetzeroSi256();

        const yq: [*]const i8 = &y[i].qs;

        // first 32 bytes of 5 elements
        {
            var qx0 = avx.mm256LoaduSi256(&x[i].qs);
            // 8-bit multiplies with shifts, masks and adds
            var qx1 = avx.mm256AddEpi8(qx0, avx.mm256AddEpi8(qx0, qx0)); // 1 * 3
            var qx2 = times9(qx0); // 1 * 9
            var qx3 = times9(qx1); // 3 * 9
            var qx4 = times9(qx2); // 9 * 9

            qx0 = topTrit(qx0);
            qx1 = topTrit(qx1);
            qx2 = topTrit(qx2);
            qx3 = topTrit(qx3);
            qx4 = topTrit(qx4);

            const qy0 = avx.mm256LoaduSi256(yq + 0);
            const qy1 = avx.mm256LoaduSi256(yq + 32);
            const qy2 = avx.mm256LoaduSi256(yq + 64);
            const qy3 = avx.mm256LoaduSi256(yq + 96);
            const qy4 = avx.mm256LoaduSi256(yq + 128);

            qx0 = avx.mm256MaddubsEpi16(qx0, qy0);
            qx1 = avx.mm256MaddubsEpi16(qx1, qy1);
            qx2 = avx.mm256MaddubsEpi16(qx2, qy2);
            qx3 = avx.mm256MaddubsEpi16(qx3, qy3);
            qx4 = avx.mm256MaddubsEpi16(qx4, qy4);

            sumi0 = avx.mm256AddEpi16(sumi0, avx.mm256AddEpi16(qx0, qx1));
            sumi1 = avx.mm256AddEpi16(sumi1, avx.mm256AddEpi16(qx2, qx3));
            sumi2 = avx.mm256AddEpi16(sumi2, qx4);
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            const qx0 = avx.mmLoaduSi128(x[i].qs[32..].ptr);
            const qh = std.mem.readInt(u32, &x[i].qh, .little);
            var qx5_l = avx.mm256Cvtepu8Epi16(avx.mmSet1Epi32(@bitCast(qh)));
            const qx1 = avx.mmAddEpi8(qx0, avx.mmAddEpi8(qx0, qx0)); // 1 * 3
            const qx2 = times9128(qx0); // 1 * 9
            const qx3 = times9128(qx1); // 3 * 9
            const qx4 = times9128(qx2); // 9 * 9
            var qx01 = avx.mm256SetM128i(qx1, qx0);
            var qx23 = avx.mm256SetM128i(qx3, qx2);

            // avx2 does not have 8-bit multiplies, so 16-bit it is.
            qx5_l = avx.mm256MulloEpi16(qx5_l, avx.mm256SetEpi16(.{ 27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1 }));
            qx5_l = avx.mm256AndSi256(qx5_l, avx.mm256Set1Epi16(0xFF));
            const qx5 = avx.mmPackusEpi16(avx.mm256Castsi256Si128(qx5_l), avx.mm256Extracti128Si256(qx5_l, 1));

            var qx45 = avx.mm256SetM128i(qx5, qx4);

            qx01 = topTrit(qx01);
            qx23 = topTrit(qx23);
            qx45 = topTrit(qx45);

            const qy01 = avx.mm256LoaduSi256(yq + 160);
            const qy23 = avx.mm256LoaduSi256(yq + 192);
            const qy45 = avx.mm256LoaduSi256(yq + 224);

            qx01 = avx.mm256MaddubsEpi16(qx01, qy01);
            qx23 = avx.mm256MaddubsEpi16(qx23, qy23);
            qx45 = avx.mm256MaddubsEpi16(qx45, qy45);

            sumi0 = avx.mm256AddEpi16(sumi0, qx01);
            sumi1 = avx.mm256AddEpi16(sumi1, qx23);
            sumi2 = avx.mm256AddEpi16(sumi2, qx45);
        }

        const ysum = avx.mm256LoaduSi256(&y[i].bsums);
        const d = avx.mm256Set1Ps(y[i].d * f(x[i].d));

        sumi0 = avx.mm256SubEpi16(sumi0, ysum);
        sumi0 = avx.mm256AddEpi16(sumi0, avx.mm256AddEpi16(sumi1, sumi2));
        sumi0 = avx.mm256MaddEpi16(sumi0, avx.mm256Set1Epi16(1));

        sumf = avx.mm256AddPs(avx.mm256MulPs(avx.mm256Cvtepi32Ps(sumi0), d), sumf);
    }

    s[0] = common.hsumFloat8(sumf);
}

comptime {
    @export(&ggmlVecDotTq20Q8K, .{ .name = "ggml_vec_dot_tq2_0_q8_K" });
}

/// Ports `ggml_vec_dot_tq2_0_q8_K` (arch/x86/quants.c:1572 @cd959bae9), the
/// `__AVX2__` arm.
///
/// Four two-bit weights a byte, each plane against its own 32 `q8_K`
/// values.
pub fn ggmlVecDotTq20Q8K(
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

    const x = as(blocks.TQ2_0, vx);
    const y = as(blocks.Q8_K, vy);

    const nb: usize = @intCast(@divTrunc(n, c.QK_K));

    var sumf = avx.mm256SetzeroPs();

    for (0..nb) |i| {
        // 16-bit sums, because 256*127 still fits
        var sumi0 = avx.mm256SetzeroSi256();
        var sumi1 = avx.mm256SetzeroSi256();

        const yq: [*]const i8 = &y[i].qs;

        var j: usize = 0;
        while (j < x[i].qs.len) : (j += 32) {
            var qx0 = avx.mm256LoaduSi256(x[i].qs[j..].ptr);
            var qx1 = avx.mm256SrliEpi16(qx0, 2);
            var qx2 = avx.mm256SrliEpi16(qx0, 4);
            var qx3 = avx.mm256SrliEpi16(qx0, 6);

            // 0, 1, 2 (should not be 3)
            qx0 = avx.mm256AndSi256(qx0, avx.mm256Set1Epi8(3));
            qx1 = avx.mm256AndSi256(qx1, avx.mm256Set1Epi8(3));
            qx2 = avx.mm256AndSi256(qx2, avx.mm256Set1Epi8(3));
            qx3 = avx.mm256AndSi256(qx3, avx.mm256Set1Epi8(3));

            const qy0 = avx.mm256LoaduSi256(yq + j * 4 + 0);
            const qy1 = avx.mm256LoaduSi256(yq + j * 4 + 32);
            const qy2 = avx.mm256LoaduSi256(yq + j * 4 + 64);
            const qy3 = avx.mm256LoaduSi256(yq + j * 4 + 96);

            qx0 = avx.mm256MaddubsEpi16(qx0, qy0);
            qx1 = avx.mm256MaddubsEpi16(qx1, qy1);
            qx2 = avx.mm256MaddubsEpi16(qx2, qy2);
            qx3 = avx.mm256MaddubsEpi16(qx3, qy3);

            sumi0 = avx.mm256AddEpi16(sumi0, avx.mm256AddEpi16(qx0, qx1));
            sumi1 = avx.mm256AddEpi16(sumi1, avx.mm256AddEpi16(qx2, qx3));
        }

        const ysum = avx.mm256LoaduSi256(&y[i].bsums);
        const d = avx.mm256Set1Ps(y[i].d * f(x[i].d));

        sumi0 = avx.mm256AddEpi16(sumi0, sumi1);
        sumi0 = avx.mm256SubEpi16(sumi0, ysum);
        sumi0 = avx.mm256MaddEpi16(sumi0, avx.mm256Set1Epi16(1));

        sumf = avx.mm256AddPs(avx.mm256MulPs(avx.mm256Cvtepi32Ps(sumi0), d), sumf);
    }

    s[0] = common.hsumFloat8(sumf);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
