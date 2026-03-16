## Proxima

A C++17 implementation of the [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320) approximate nearest neighbor search algorithm. Built for clarity and experimentation, Proxima provides a readable, from-scratch HNSW index with configurable parameters, multiple distance metrics, and platform-aware SIMD acceleration.

Based on the paper: *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"* by Yu. A. Malkov and D. A. Yashunin ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

## Features

- **Multi-layer HNSW graph** with exponentially decaying level distribution
- **Heuristic neighbor selection** (Algorithm 4 from the paper) for better graph connectivity on clustered data
- **Multiple distance metrics**: L2 (squared Euclidean), inner product, cosine similarity
- **SIMD acceleration**: AVX2 on x86_64, NEON on ARM64, with automatic fallback to scalar
- **Configurable parameters**: `M`, `efConstruction`, `efSearch`, distance type, and force-scalar mode
- **Incremental insertion**: add vectors one at a time or in bulk via `create()`
- **Benchmark suite**: C++ benchmarks against brute-force, Python benchmarks against [hnswlib](https://github.com/nmslib/hnswlib), with automated comparison reports and plots
- **GoogleTest unit tests** with address and undefined behavior sanitizers

## Prerequisites

- CMake >= 3.16
- A C++17 compiler (Clang or GCC)
- Python 3.10+ (for benchmarks only)
- clang-format and clang-tidy (for linting, installed via `make setup`)

## Setup

```bash
# Install development tools (clang-format, clang-tidy via LLVM)
make setup

# Build the project
make build
```

On macOS, the setup script installs LLVM via Homebrew and adds it to your PATH. On Linux, it uses your system package manager (apt/dnf/pacman).

## Build

```bash
# Configure and build (incremental)
make build

# Clean rebuild
make rebuild
```

## Testing

```bash
# Run all tests (with address + UB sanitizers)
make test
```

## Benchmarks

The benchmark suite compares Proxima's `HnswCPU` against brute-force search (C++) and against [hnswlib](https://github.com/nmslib/hnswlib) (Python). Results are saved in a timestamped folder under `benchmarks/results/`.

```bash
# Run the full benchmark suite (C++ + Python + comparison + plots)
make bench

# Run C++ benchmarks only (requires RUN_DIR)
make cppbench RUN_DIR=my-run

# Run Python benchmarks only (requires RUN_DIR)
make pybench RUN_DIR=my-run

# Regenerate plots for the latest run
make plot
```

Each `make bench` invocation creates a folder at `benchmarks/results/<dd-mm-yyyy-hh-mm>/` containing:

- `cpp_results.csv` -- C++ benchmark results (scalar and SIMD modes)
- `python_results.csv` -- Python hnswlib benchmark results
- `comparison.csv` -- head-to-head comparison
- `plots/build_s.png` -- build time comparison chart
- `plots/query_us.png` -- query time comparison chart
- `report.md` -- full markdown report with system info, tables, and plots

### Reproducing Benchmarks

```bash
# Full suite: builds project, sets up Python venv, runs benchmarks, generates report
make bench
```

The benchmark scenarios range from 1K to 10M vectors across dimensions 64, 128, and 256 with K values of 5, 10, and 50. The Python venv and hnswlib dependency are installed automatically on the first run.

## Code Quality

```bash
# Format all C++ source files
make format

# Check formatting (CI-friendly, exits non-zero on diff)
make format-check

# Run clang-tidy static analysis
make lint

# Run clang-tidy and apply automatic fixes
make lint-fix
```

## Project Structure

```
proxima/
  src/
    hnsw.h              # HnswCPU class definition
    hnsw.cpp            # HNSW graph construction and search
    dist/
      dispatch.h        # Distance type enum and dispatch
      dispatch.cpp      # Runtime SIMD detection and dispatch
      l2.h              # L2 distance (scalar, AVX2, NEON)
      inner_product.h   # Inner product distance
      cosine.h          # Cosine distance
  tests/
    test_hnsw.cpp       # GoogleTest unit tests
  benchmarks/
    bench.cpp           # C++ benchmark harness
    bench.py            # Python hnswlib benchmark
    compare.py          # Comparison analysis and report generation
    plot.py             # Plotting with matplotlib + catppuccin theme
    requirements.txt    # Python dependencies
    results/            # Timestamped benchmark outputs
  scripts/
    setup.sh            # Dev environment setup
  CMakeLists.txt
  Makefile
```

## References

- Malkov, Y.A. and Yashunin, D.A., 2018. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE Transactions on Pattern Analysis and Machine Intelligence. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- [hnswlib](https://github.com/nmslib/hnswlib) -- Header-only C++ HNSW library with Python bindings
