## Proxima

A C++17 implementation of the [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320) approximate nearest neighbor search algorithm.

Designed for learning, experimentation, and performance evaluation, Proxima provides a readable, from-scratch HNSW index with configurable parameters, multiple distance metrics, and platform-aware SIMD acceleration.

Based on the paper **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"** by Yu. A. Malkov and D. A. Yashunin ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

### Features

- **Multi-layer HNSW graph** with exponentially decaying level distribution

- **Heuristic neighbor selection** (Algorithm 4 from the paper) for better graph connectivity on clustered data

- **Multiple distance metrics** :
  - L2 (squared Euclidean)
  - Inner product
  - Cosine similarity

- **SIMD acceleration**:
  - AVX2 + FMA on x86_64
  - NEON on ARM64
  - Automatic scalar fallback

- **Python bindings** using [nanobind](https://nanobind.readthedocs.io/)
- **Benchmark suite**:
  - C++ Proxima vs brute-force
  - Python Proxima bindings vs [hnswlib](https://github.com/nmslib/hnswlib)
  - Automated comparison reports and plots

- **GoogleTest unit tests** with AddressSanitizer and UndefinedBehaviorSanitizer
- **clang-format and clang-tidy** integration


### Prerequisites

#### C++

- CMake >= 3.18
- C++17 compiler
  - Clang
  - GCC

#### Python

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)
- nanobind (installed automatically by the Python build)
- pytest
- hnswlib (for Python benchmarks)

#### Development tools

- clang-format
- clang-tidy

These can be installed automatically with:

```bash
make setup
```

On macOS, the setup script installs LLVM through Homebrew and adds it to `PATH`.

On Linux, it uses the available system package manager (`apt`, `dnf`, or `pacman`).

---

# Quick Start

### 1. Install development tools

```bash
make setup
```

### 2. Build the C++ project

```bash
make build
```

### 3. Run C++ tests

```bash
make test
```

### 4. Install the Python package

Using `uv`:

```bash
uv pip install -e ".[test,benchmark]"
```

This installs Proxima in editable mode together with:

- `pytest`
- `numpy`
- `hnswlib`

The Python extension is built automatically through CMake + nanobind.

### 5. Verify Python bindings

```bash
python -c "import proxima; print(proxima)"
```

You can also verify the exported classes:

```bash
python -c "import proxima; print(proxima.HnswCPU); print(proxima.DistanceType)"
```

---

# C++ Build

Configure and build incrementally:

```bash
make build
```

Equivalent CMake commands:

```bash
cmake -S . -B build \
    -DPROXIMA_BUILD_TESTS=ON \
    -DPROXIMA_BUILD_BENCHMARKS=ON

cmake --build build --parallel 8
```

Build only with CMake:

```bash
cmake -S . -B build
cmake --build build --parallel
```

#### Clean rebuild

```bash
make rebuild
```

Or manually:

```bash
rm -rf build

cmake -S . -B build \
    -DPROXIMA_BUILD_TESTS=ON \
    -DPROXIMA_BUILD_BENCHMARKS=ON

cmake --build build --parallel
```

---

# SIMD Builds

Proxima automatically supports scalar and platform-specific SIMD implementations.

### Scalar build

```bash
cmake -S . -B build \
    -DPROXIMA_ENABLE_AVX2=OFF
```

Then:

```bash
cmake --build build --parallel
```

### AVX2 build

On x86_64:

```bash
cmake -S . -B build \
    -DPROXIMA_ENABLE_AVX2=ON
```

Then:

```bash
cmake --build build --parallel
```

On ARM64/macOS, AVX2 should normally remain disabled:

```bash
cmake -S . -B build \
    -DPROXIMA_ENABLE_AVX2=OFF
```

NEON is detected automatically by the compiler.

---

# Testing

### Run all C++ tests

```bash
make test
```

This builds the project and runs the GoogleTest suite with:

- AddressSanitizer
- UndefinedBehaviorSanitizer

You can also run the test binary directly:

```bash
./build/test_proxima
```

Or use CTest:

```bash
cd build
ctest --output-on-failure
```

### Run Python tests

First install the Python test dependencies:

```bash
uv pip install -e ".[test]"
```

Then:

```bash
pytest tests/python
```

Or:

```bash
python -m pytest tests/python
```

Run a specific test:

```bash
pytest tests/python/test_proxima.py
```

Run with verbose output:

```bash
pytest -v tests/python
```

---

# Python Package / Bindings

Proxima's Python package consists of:

```text
python/
└── proxima/
    └── __init__.py
```

with the native extension:

```text
_proxima
```

generated from:

```text
bindings/python.cpp
```

The package is built using:

```text
uv
 ↓
scikit-build-core
 ↓
CMake
 ↓
nanobind
 ↓
C++ Proxima
```

### Install in editable mode

```bash
uv pip install -e ".[test,benchmark]"
```

### Rebuild the Python extension

After changing C++ or binding code:

```bash
uv pip install -e ".[test,benchmark]"
```

If you want to force a completely clean package build:

```bash
uv pip uninstall proxima
uv pip install -e ".[test,benchmark]"
```

### Verify the installation

```bash
python -c "import proxima; print(proxima)"
```

```bash
python -c "from proxima import HnswCPU, DistanceType; print(HnswCPU); print(DistanceType)"
```

---

# Python Benchmarks

There are two Python benchmark implementations:

```text
benchmarks/bench_proxima_pybinding.py
benchmarks/bench_hnswlib.py
```

`bench_proxima_pybinding.py` benchmarks the Proxima C++ implementation through the Python/nanobind bindings.

`bench_hnswlib.py` benchmarks the Python bindings of hnswlib.

Install the benchmark dependencies:

```bash
uv pip install -e ".[benchmark]"
```

### Run Proxima Python benchmark

```bash
python benchmarks/bench_proxima_pybinding.py
```

### Run hnswlib benchmark

```bash
python benchmarks/bench_hnswlib.py
```

The benchmark results are written as CSV files under the configured benchmark results directory.

---

# C++ Benchmarks

The C++ benchmark is:

```text
benchmarks/bench_proxima.cpp
```

It benchmarks:

- Proxima scalar implementation
- Proxima SIMD implementation
- Brute-force KNN
- Query latency
- Build time
- Recall
- Speedup over brute force

### Run C++ benchmark directly

After building:

```bash
./build/bench_proxima
```

Specify an output directory:

```bash
./build/bench_proxima benchmarks/results/my-run
```

### Run through Make

```bash
make cppbench RUN_DIR=my-run
```

---

# Full Benchmark Suite

The recommended way to reproduce the complete benchmark is:

```bash
make bench
```

This runs the complete pipeline:

```text
Build C++
    ↓
Install Python dependencies
    ↓
C++ Proxima benchmark
    ↓
Python Proxima benchmark
    ↓
Python hnswlib benchmark
    ↓
Comparison
    ↓
Plots
    ↓
Markdown report
```

### Run individual benchmark stages

#### C++ Proxima

```bash
make cppbench RUN_DIR=my-run
```

#### Python Proxima bindings

```bash
make pybench RUN_DIR=my-run
```

#### Generate plots

```bash
make plot
```

#### Generate comparison

```bash
python benchmarks/compare.py benchmarks/results/my-run
```

#### Generate plots manually

```bash
python benchmarks/plot.py benchmarks/results/my-run
```

---

# Benchmark Results

Each benchmark run creates a timestamped directory:

```text
benchmarks/results/<dd-mm-yyyy-hh-mm>/
```

A complete run contains:

```text
benchmarks/results/<run>/
├── cpp_results.csv
├── python_results.csv
├── comparison.csv
├── plots/
│   ├── build_s.png
│   └── query_us.png
└── report.md
```

#### `cpp_results.csv`

C++ Proxima results:

```text
impl,N,DIM,K,build_s,query_us,brute_us,speedup,recall
```

The implementations include:

```text
cpp_scalar
cpp_simd
```

#### `python_results.csv`

Python benchmark results for Proxima's Python bindings and hnswlib.

#### `comparison.csv`

Head-to-head comparison between implementations.

#### `report.md`

Complete benchmark report containing:

- System information
- Benchmark configuration
- Build times
- Query latency
- Speedups
- Recall
- Generated plots

---

# Benchmark Scenarios

The benchmark evaluates different combinations of:

#### Dataset size

```text
1,000
5,000
10,000
50,000
100,000
500,000
```

and larger configurations where supported by the benchmark.

#### Dimensions

```text
64
128
256
```

#### K

```text
5
10
50
```

This allows the effect of dataset size, dimensionality, and requested nearest-neighbor count to be evaluated independently.

---

# Code Quality

### Format C++

Format all C++ sources:

```bash
make format
```

This uses `clang-format`.

### Check formatting

```bash
make format-check
```

This is useful for CI because it exits with a non-zero status when formatting changes are required.

### Run clang-tidy

```bash
make lint
```

Run clang-tidy with automatic fixes:

```bash
make lint-fix
```

If clang-tidy reports system-header/compiler errors, make sure the `clang-tidy` binary being used corresponds to the compiler/toolchain used to build the project.

---

# Complete Development Workflow

For a normal development cycle:

```bash
# 1. Install development tools
make setup

# 2. Build C++
make build

# 3. Run C++ tests
make test

# 4. Install Python package + test/benchmark dependencies
uv pip install -e ".[test,benchmark]"

# 5. Verify Python bindings
python -c "import proxima; print(proxima)"

# 6. Run Python tests
pytest tests/python

# 7. Run complete benchmark suite
make bench

# 8. Check formatting
make format-check

# 9. Run static analysis
make lint
```

---

# Clean Everything

To remove the CMake build directory:

```bash
rm -rf build
```

To remove the Python package from the environment:

```bash
uv pip uninstall proxima
```

To perform a completely clean C++ rebuild:

```bash
rm -rf build

cmake -S . -B build \
    -DPROXIMA_BUILD_TESTS=ON \
    -DPROXIMA_BUILD_BENCHMARKS=ON

cmake --build build --parallel
```

Then reinstall the Python package:

```bash
uv pip install -e ".[test,benchmark]"
```

### References

- Malkov, Y. A. and Yashunin, D. A. (2018). **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs**. IEEE Transactions on Pattern Analysis and Machine Intelligence. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- [hnswlib](https://github.com/nmslib/hnswlib) — Header-only C++ HNSW library with Python bindings
- [nanobind](https://nanobind.readthedocs.io/) — Lightweight C++/Python binding library
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) — Modern Python build backend for CMake projects
- [uv](https://docs.astral.sh/uv/) — Python package and project manager
