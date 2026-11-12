# UPSTREAM

zigggml's ggml comes from **llama.zig**, which ports it from **llama.cpp**.
llama.zig drives every sync with upstream. zigggml's Zig sources started as
copies of llama.zig's and have since been restyled to `ZIGSTYLE.md` (camelCase
functions, house doc format, lint-clean), so upstream changes are ported into
them by hand. Only `vendor/` is re-copied.

## Pin

<!-- pin:begin (written by scripts/sync) -->
- llama.zig: `f08a6a0 (f08a6a0)` (tag `v0.9.0+3`)
- llama.cpp: `v0.9.0 (cd959bae9)`
- synced: 2026-11-11
<!-- pin:end -->

The copy is fresh (no git history carried over). Lineage lives in each ported file's provenance header, which names its upstream source path and line. zigggml's version follows llama.zig's ggml state at the llama.cpp pin.

## What comes from where

| zigggml | source | notes |
| --- | --- | --- |
| `src/**` | `llama.zig/src/ggml/**` | restyled; minus `ported.zig` and `.mc.zig`; `src/root.zig` is zigggml's |
| `build/metal_embed.zig`, `build/cuda_kernels.zig`, `build/vulkan_shaders.zig` | `llama.zig/build/` | restyled |
| `build/ggml.zig` | `llama.zig/build/llamacpp.zig`, `build.zig` | ported by hand: `addGgml`, the CUDA/Vulkan/Metal helpers, the option resolution |
| `cli/cpu.zig` | `llama.zig/tools/common/cpu.zig` | restyled |
| `scripts/vk-toolchain` | `llama.zig/scripts/vk-toolchain` | header comment adapted |
| `vendor/ggml/LICENSE` | `llama.cpp/LICENSE` | MIT |
| `vendor/ggml/include/` | `llama.cpp/ggml/include/` | `ggml.h` is regenerated at build time with `config.zon`'s `GGML_DEFAULT_N_THREADS` |
| `vendor/ggml/src/*.h` | `llama.cpp/ggml/src/*.h` | for the `ggml_c` translation, the `.m` files and `nvcc` |
| `vendor/ggml/src/ggml-metal/` | `llama.cpp/ggml/src/ggml-metal/` | the two `.m` files, headers, `kernels/` (the `.cpp` files are ported) |
| `vendor/ggml/src/ggml-cuda/` | `llama.cpp/ggml/src/ggml-cuda/` | compiled to fatbins; no host code is linked |
| `vendor/ggml/src/ggml-vulkan/vulkan-shaders/` | `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/` | shader sources and `vulkan-shaders-gen.cpp` |

## Sync procedure

1. Sync llama.zig with upstream first (its `SYNC.md`), and let its gates pass.
2. `make sync` (or `scripts/sync <path-to-llama.zig>`). This re-copies `vendor/`, updates the pin, and lists the llama.zig files changed since the previous pin.
3. Port those changes into the restyled sources by hand. Keep each file's provenance header, and keep the export of every C symbol the change touches. If `build/llamacpp.zig` changed, port the ggml-relevant parts into `build/ggml.zig`.
4. `make validate`.
5. Symbol parity: build llama.zig's `reference` step with the same backends into a scratch prefix, and compare.

   ```sh
   syms() { nm -g --defined-only "$1" | awk 'NF==3 && $2!="W" && $2!="V" {print $3}' | sort -u; }
   diff <(syms "$LLAMA_ZIG_PREFIX/lib/libggml.a") <(syms zig-out/lib/libggml.a)
   ```

   The output must be empty.

## Known divergences from llama.zig

- Vulkan SPIR-V is generated here from the vendored shader sources with the pinned glslc, not shared with llama.zig's build. `glslc -O` is not reproducible on `tri_f16`, so that one shader's bytes may differ. The exported symbols are unaffected.
- The three empty C++ units llama.zig still compiles (`ggml-cpu/hbm.cpp`, `amx/amx.cpp`, `amx/mmq.cpp`) are not vendored. They are empty on every supported target and export nothing.
