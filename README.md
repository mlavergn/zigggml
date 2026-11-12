# zigggml

> GGML, the tensor and compute-graph library behind llama.cpp, as a standalone Zig module.

## Overview

zigggml is the Zig port of GGML from [llama.zig](../llama.zig), extracted into a reusable library: the core (tensors, ops, graphs, quantization, GGUF, the scheduler) and every backend (CPU, Metal, CUDA, Vulkan, BLAS over Accelerate). Its `libggml.a` exports the same C ABI as llama.zig's, symbol for symbol, so C, C++ and Zig callers use the upstream `ggml.h` API unchanged.

The `zigggml-cli` executable lists the backends and devices, and benchmarks every op on every backend: wall time, CPU and GPU usage, and throughput, with each non-CPU result checked against the CPU.

## Requirements

- Zig 0.17.0 or later
- macOS (Apple Silicon) or Linux (arm64 / x86_64)
- macOS: Xcode or its command-line tools (the SDK; Metal and Accelerate)
- CUDA (`-Dcuda`, Linux): the CUDA toolkit (`nvcc`, cuBLAS) and an NVIDIA driver
- Vulkan (`-Dvulkan`): `glslc` from SDK 1.4.363 (`make vk-toolchain` builds it), the Vulkan headers and loader
- `make validate` also needs the sibling `../styleguide` checkout and `zlint`

## Setup

```sh
git clone git@github.com:mlavergn/zigggml.git
cd zigggml
make build          # zig build: libggml.a, headers, zigggml-cli
make validate       # clean, format, lint, build, test
```

Backends beyond the CPU (and Metal, on every Apple target):

```sh
zig build -Dcuda                 # -Dcuda-arch=121a, -Dcuda-toolkit=..., -Dcuda-fa-quants=...
make vk-toolchain && zig build -Dvulkan
zig build -Dblas                 # Apple only
```

## Configuration

`config.zon` holds build-time settings for the library:

- `.default_n_threads`: the value of `GGML_DEFAULT_N_THREADS` in the generated and installed `ggml.h`, the CPU thread count ggml uses when a caller sets none. Upstream's is 4.

The bench CLI defaults to `common_cpu_get_num_math()` threads, as llama.cpp's tools do.

## Usage

As a library, link `libggml.a` and include `zig-out/include/ggml.h`, or from Zig depend on this package and import the `ggml_c` module (the translated headers) while linking the `ggml` artifact.

The CLI:

```sh
zig-out/bin/zigggml-cli devices
make bench                                         # the perf case list, every backend, CPU thread sweep
make bench ARGS="--op MUL_MAT --type q4_0 --threads 8"
make bench ARGS="--full --backend CUDA0 --json" ZIG_ARGS="-Dcuda"
```

`bench` runs `test-backend-ops`' case lists, ported to Zig: the perf list by default, the full eval list with `--full`. Each case gets 1 warmup and 10 timed runs (`--warmup`, `--runs`), and reports min, median and max per-op wall time, process CPU%, GPU% (NVML, CUDA builds), and GFLOPS or GB/s. Each non-CPU backend is checked against the CPU (NMSE, upstream's thresholds). `zigggml-cli bench --help` lists every option.

## Architecture

```
src/            the ggml port (copied verbatim from llama.zig's src/ggml/; see UPSTREAM.md)
  module.zig      barrel: forces every ported translation unit into the archive
  cpu/ metal/ cuda/ vulkan/ quants/   backends and kernels
vendor/ggml/    upstream inputs that are not Zig: headers, Metal .m and kernels,
                CUDA .cu, Vulkan shader sources (llama.cpp v0.9.0, MIT)
build/ggml.zig  the library build: backends, header generation, nvcc/glslc steps
cli/            zigggml-cli: devices, bench
  bench/          the port of test-backend-ops: runner, report, case lists
  bench/cases/    one file per test-backend-ops case class
scripts/        sync (re-copy from llama.zig), vk-toolchain
```

`make sync` re-copies everything from `../llama.zig`; `UPSTREAM.md` records the pin and the procedure.
