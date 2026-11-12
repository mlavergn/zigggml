//! x86's `arch-fallback.h` renames for `repack.cpp`: the 28 kernels
//! `arch/x86/repack.cpp` has no version of, so the generic body is the real
//! one.
//!
//! # Provenance
//!
//! **Not a port of a function.** It reproduces
//! `llama.cpp/ggml/src/ggml-cpu/arch-fallback.h` at v0.9.0 (`cd959bae9`),
//! lines 88-119, the `__x86_64__` block, which `#define`s each generic to
//! the bare name before `repack.cpp` compiles. The bodies are the generic
//! ports in `cpu/repack/`; this exports each under the name the C gives it.
//! `arm/fallback.zig` does the same for ARM's eight.

const nibble4 = @import("../nibble4.zig");
const q1_0 = @import("../q1_0.zig");
const q4_0 = @import("../q4_0.zig");
const q4_k = @import("../q4_k.zig");
const q5_k = @import("../q5_k.zig");
const q6_k = @import("../q6_k.zig");
const q8_0 = @import("../q8_0.zig");
const quantize = @import("../quantize.zig");

comptime {
    @export(&quantize.ggmlQuantizeMatQ804x4Generic, .{ .name = "ggml_quantize_mat_q8_0_4x4" });
    @export(&quantize.ggmlQuantizeMatQ8K4x4Generic, .{ .name = "ggml_quantize_mat_q8_K_4x4" });

    @export(&q1_0.ggmlGemvQ104x4Q80Generic, .{ .name = "ggml_gemv_q1_0_4x4_q8_0" });
    @export(&q1_0.ggmlGemvQ104x8Q80Generic, .{ .name = "ggml_gemv_q1_0_4x8_q8_0" });
    @export(&q4_0.ggmlGemvQ404x4Q80Generic, .{ .name = "ggml_gemv_q4_0_4x4_q8_0" });
    @export(&q4_0.ggmlGemvQ404x8Q80Generic, .{ .name = "ggml_gemv_q4_0_4x8_q8_0" });
    @export(&q4_k.ggmlGemvQ4K8x4Q8KGeneric, .{ .name = "ggml_gemv_q4_K_8x4_q8_K" });
    @export(&q5_k.ggmlGemvQ5K8x4Q8KGeneric, .{ .name = "ggml_gemv_q5_K_8x4_q8_K" });
    @export(&q5_k.ggmlGemvQ5K8x8Q8KGeneric, .{ .name = "ggml_gemv_q5_K_8x8_q8_K" });
    @export(&q6_k.ggmlGemvQ6K8x4Q8KGeneric, .{ .name = "ggml_gemv_q6_K_8x4_q8_K" });
    @export(&q6_k.ggmlGemvQ6K8x8Q8KGeneric, .{ .name = "ggml_gemv_q6_K_8x8_q8_K" });
    @export(&nibble4.ggmlGemvIq4Nl4x4Q80Generic, .{ .name = "ggml_gemv_iq4_nl_4x4_q8_0" });
    @export(&nibble4.ggmlGemvMxfp44x4Q80Generic, .{ .name = "ggml_gemv_mxfp4_4x4_q8_0" });
    @export(&q8_0.ggmlGemvQ804x4Q80Generic, .{ .name = "ggml_gemv_q8_0_4x4_q8_0" });
    @export(&q8_0.ggmlGemvQ804x8Q80Generic, .{ .name = "ggml_gemv_q8_0_4x8_q8_0" });

    @export(&q1_0.ggmlGemmQ104x4Q80Generic, .{ .name = "ggml_gemm_q1_0_4x4_q8_0" });
    @export(&q1_0.ggmlGemmQ104x8Q80Generic, .{ .name = "ggml_gemm_q1_0_4x8_q8_0" });
    @export(&q4_0.ggmlGemmQ404x4Q80Generic, .{ .name = "ggml_gemm_q4_0_4x4_q8_0" });
    @export(&q4_0.ggmlGemmQ404x8Q80Generic, .{ .name = "ggml_gemm_q4_0_4x8_q8_0" });
    @export(&q4_k.ggmlGemmQ4K8x4Q8KGeneric, .{ .name = "ggml_gemm_q4_K_8x4_q8_K" });
    @export(&q5_k.ggmlGemmQ5K8x4Q8KGeneric, .{ .name = "ggml_gemm_q5_K_8x4_q8_K" });
    @export(&q5_k.ggmlGemmQ5K8x8Q8KGeneric, .{ .name = "ggml_gemm_q5_K_8x8_q8_K" });
    @export(&q6_k.ggmlGemmQ6K8x4Q8KGeneric, .{ .name = "ggml_gemm_q6_K_8x4_q8_K" });
    @export(&q6_k.ggmlGemmQ6K8x8Q8KGeneric, .{ .name = "ggml_gemm_q6_K_8x8_q8_K" });
    @export(&nibble4.ggmlGemmIq4Nl4x4Q80Generic, .{ .name = "ggml_gemm_iq4_nl_4x4_q8_0" });
    @export(&nibble4.ggmlGemmMxfp44x4Q80Generic, .{ .name = "ggml_gemm_mxfp4_4x4_q8_0" });
    @export(&q8_0.ggmlGemmQ804x4Q80Generic, .{ .name = "ggml_gemm_q8_0_4x4_q8_0" });
    @export(&q8_0.ggmlGemmQ804x8Q80Generic, .{ .name = "ggml_gemm_q8_0_4x8_q8_0" });
}
