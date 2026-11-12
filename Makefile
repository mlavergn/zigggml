###############################################
#
# Makefile — zigggml
#
# `make validate` is the pre-commit gate; `make build` is the entry point.
#
###############################################

# Sources the styleguide's format and lint targets walk. Set before the
# include, which only defaults them.
ZIG_DIRS := src cli build

# Lint covers what zigggml owns: build.zig, build/ and cli/. src/ is a
# verbatim copy of llama.zig's src/ggml/, which llama.zig does not lint either
# (its exported C names alone are ~1,800 case-convention hits), and
# `make sync` overwrites it. The build-time tools below are verbatim copies
# from llama.zig and are excluded for the same reason.
LINT_EXCLUDE := build/vulkan_shaders.zig build/cuda_kernels.zig
LINT_FILES := $(filter-out $(LINT_EXCLUDE),build.zig $(shell find build cli -name '*.zig' | sort))

# Optional include to extend the makefile targets
-include ../styleguide/Makefile.in

.DEFAULT_GOAL := all

.PHONY: build dist run bench sync vk-toolchain

# ---------------------------------------------
# Configuration
# ---------------------------------------------

# The released version, read from build.zig.zon (the source of truth).
VERSION := $(shell sed -n -E 's/^[[:space:]]*\.version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' build.zig.zon)

# ---------------------------------------------
# Primary workflows
# ---------------------------------------------

# Build, format, and test.
all: build format test
	@echo "done"

# ---------------------------------------------
# Build
# ---------------------------------------------

# Build the library and CLI.
build:
	zig build

# Build optimized for release.
dist:
	zig build --release=fast

# Build and run the CLI. Pass arguments with `make run ARGS="Zig"`.
run:
	zig build run -- $(ARGS)

# Benchmark every op on every built backend. Pass options with
# `make bench ARGS="--op MUL_MAT --backend CPU"`, backends with
# `make bench ZIG_ARGS="-Dcuda"`.
bench:
	zig build run -Doptimize=ReleaseFast $(ZIG_ARGS) -- bench $(ARGS)

# ---------------------------------------------
# Cross-compilation
# ---------------------------------------------

# Compile for Linux.
linux:
	zig build -Dtarget=x86_64-linux

# Compile for Windows.
windows:
	zig build -Dtarget=x86_64-windows

# List every target Zig can build for.
targets:
	zig targets

# ---------------------------------------------
# Test
# ---------------------------------------------

# Run the unit test suite.
test:
	zig build test --summary all

# ---------------------------------------------
# Upstream
# ---------------------------------------------

# Re-copy ggml, the build helpers and vendor/ from llama.zig (see UPSTREAM.md).
# Pass another checkout with `make sync LLAMA_ZIG=../path`.
LLAMA_ZIG ?= ../llama.zig
sync:
	./scripts/sync $(LLAMA_ZIG)

# Build the pinned Vulkan toolchain (glslc from SDK 1.4.363) into .tools/vulkan/.
vk-toolchain:
	./scripts/vk-toolchain

# ---------------------------------------------
# Documentation
# ---------------------------------------------

# Build the API docs and serve them locally.
docs:
	zig build docs
	open "http://127.0.0.1:8080" &
	python3 -m http.server -b 127.0.0.1 8080 -d zig-out/docs

# ---------------------------------------------
# Release
# ---------------------------------------------

# Tag the build.zig.zon version (e.g. 1.0.0) and push it.
tag:
	@test -n "$(VERSION)" || { echo "❌ could not read .version from build.zig.zon"; exit 1; }
	git tag -a "$(VERSION)" -m "$(VERSION)"
	git push
	git push --tags

# ---------------------------------------------
# Environment
# ---------------------------------------------

# Open the working copy in SourceTree.
st:
	open -a SourceTree .

# Open the project in the editor.
open:
	code .

# Open the repository on GitHub.
github:
	open "https://github.com/mlavergn/zigggml"

# Clone a fresh working copy.
clone:
	git clone git@github.com:mlavergn/zigggml.git

# Start Claude Code here.
claude:
	claude

# ---------------------------------------------
# Housekeeping
# ---------------------------------------------

# Remove every build artifact.
clean:
	rm -rf .zig-cache
	rm -rf zig-out
