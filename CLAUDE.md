# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`zigggml` is GGML (tensors, ops, graphs, quantization, GGUF, scheduler; CPU, Metal, CUDA, Vulkan and BLAS backends) as a standalone Zig module, extracted from the Zig port in `../llama.zig/src/ggml/`. Its `libggml.a` must export **the same C ABI as llama.zig's**, because llama.zig will later depend on this module. `zigggml-cli` lists devices and benchmarks every op (`bench`). `UPSTREAM.md` is the pin and sync procedure.

Scope boundaries:
- Switching llama.zig onto zigggml is separate, future work. Nothing here touches llama.zig.
- llama.zig's parity gates against the reference build (`*-diff`, `test-backend-ops`) stay in llama.zig. zigggml builds only the port: there are no `*-port` toggles, and the `config` options the sources read for them are fixed to the port.
- The version tracks llama.zig's ggml state at the llama.cpp pin (`0.9.0`).

## Hard constraints

- **Zig 0.17.0** (`.minimum_zig_version` in `build.zig.zon`).
- `src/` began as a copy of `../llama.zig/src/ggml/` and is now restyled to ZIGSTYLE. Upstream changes are ported in by hand (`UPSTREAM.md`); `make sync` never overwrites it. **Keep every provenance header** (`Ports \`x\` (file.c:N @commit)`), which records the upstream source.
- **Never edit `vendor/ggml/`.** It is upstream's verbatim, and `make sync` re-copies it.
- Every C symbol keeps its name. A ported function is camelCase in Zig and exported with `@export(&fooBar, .{ .name = "foo_bar" })` plus `callconv(.c)`. `extern fn` declarations keep their C names.
- **No Zig package dependencies.** `.dependencies` holds only `zigbuildx` (build-only, pinned by `.url` + `.hash`, never `.path`). System SDKs (CUDA, Vulkan, Apple frameworks) are linked, not depended on.
- `build.zig.zon` is the source of truth for name and version. `.paths = .{ "src" }`.
- `config.zon` sets `GGML_DEFAULT_N_THREADS`. `build/ggml.zig` rewrites that `#define` in a generated copy of `vendor/ggml/include/` (the whole directory, since the headers include `"ggml.h"` quoted), which feeds both the translation and the installed headers.
- macOS builds resolve the Xcode SDK through `zigbuildx.Xcode`.

## Build / Test

`make validate` (`clean format lint build test`) is the pre-commit gate. `format`, `lint` and `validate` come from `../styleguide/Makefile.in`.

- `make test` runs the library's ~420 unit tests (~2.5 min in Debug), the CLI tests, and `bench --smoke`. Use `zig build test-cli` for the CLI alone (seconds).
- Zig's test runner prints `failed command:` above the stderr of *passing* tests (ggml logs backend registration). Read the `Build Summary` line. Any `log.err` inside a test fails it.
- `-Dtest-filter=<str>` filters both test binaries.
- `make lint` covers every Zig file (`build.zig`, `build/`, `cli/`, `src/`). zlint misses references inside `callconv(...)`, so a calling-convention constant reads as unused; write the convention inline.
- CPU is the default. `-Dcuda` (an nvcc build takes ~3 min), `-Dvulkan`, `-Dblas` (Apple only). Metal is always on for Apple targets.
- Symbol parity check: see `UPSTREAM.md`. The diff against llama.zig's `reference` build with the same backends must be empty. One anonymous-namespace symbol in `cub_shim.cu` differs by path hash; that is expected.

## Architecture conventions

- Consumers use ggml's C API: link the `ggml` artifact and import the translated headers as the public `ggml_c` module. `src/module.zig` exports no Zig API; it only forces every ported unit into the archive.
- CUDA executables need Zig's libc++ and gcc's `libgcc.a` on their link. `ggml.addLinkDeps` adds them, plus the CUDA/Vulkan libraries and SDK paths, since search paths do not travel with a static library. NVML is linked into the CLI only.
- **For stdout/stderr use `std.Io.File.stdout().writerStreaming(io, &buf)`, not `.writer(...)`.** `writer()` defaults to positional writes, which interleave destructively with the streaming writes `std.log` makes.
- Bench cases (`cli/bench/cases/`) port `test-backend-ops.cpp` one class per file. Fields are the C++ constructor's parameters **in order, with the C++ defaults**, so `CaseList.add(T, .{args...})` mirrors `new T(args...)`. The interface and optional overrides are documented in `cli/bench/case.zig`. A test instantiates every case, so an interface break fails `test-cli`.
- `bench` departs from `test-backend-ops` on purpose:
  - It times both case lists (upstream times only its perf list) with a fixed warmup plus N runs, not "at least one second".
  - Before timing, it requires *every* node of a graph to be supported, not just the output, and skips CUDA graphs with skipped (`-1`) expert ids (`Runner.perf`). Without these, eval-list cases abort or fault on CUDA.
  - There is no memory column: per-op peak RSS isn't meaningful.
- No per-function entry tracing in `src/`: these are low-level kernels. Test diagnostics use `std.log.warn` (`log.err` fails a test).
- Translate-C gives `[*c]` for `ggml_tensor` (it coerces to `*`) but `?*` for opaque types (`ggml_cgraph`, `ggml_context`). Variables named `i0`…`i3` shadow primitive types; use `j0`….

## Open items

- Metal and BLAS have not been built or symbol-checked: they need an Apple host. Build with `-Dblas` on macOS and run the `UPSTREAM.md` parity check against llama.zig's `reference` step.
- On CUDA, `MUL_MAT_ID` with skipped expert ids is unsupported upstream (`TAG_MUL_MAT_ID_SKIP`, llama.cpp PR 26631). The fix belongs upstream, in `mmid.cu` and its kernels, and arrives through llama.zig.

## Style

`../styleguide/ZIGSTYLE.md` is authoritative and imported below. Conform code to it, not the reverse. The rule most often broken by habit: **enum variants are camelCase** (`.startOfBlock`), unlike Zig std.

**Before using any `std` API, check `../styleguide/ZIGMIGRATE.md`.** Never use a deprecated API, and clear any deprecation warning in the same change that produced it.

`../styleguide` is the shared sibling checkout (its own repo). Never edit it from this repo; flag upstream if a rule is wrong.

@../styleguide/ZIGSTYLE.md
