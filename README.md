## Proxima

A C++17 implementation of the [Hierarchical Navigable Small World (HNSW)](https://arxiv.org/abs/1603.09320) approximate nearest neighbor search algorithm.

Proxima is designed for learning, experimentation, and performance evaluation, with a readable from-scratch implementation, configurable parameters, multiple distance metrics, SIMD acceleration, and Python bindings.

Based on **"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"** by Yu. A. Malkov and D. A. Yashunin.

### Features

- Multi-layer HNSW graph with exponentially decaying level distribution
- Heuristic neighbor selection from the HNSW paper
- Distance metrics:
  - L2 / squared Euclidean
  - Inner product
  - Cosine similarity
- SIMD acceleration:
  - AVX2 + FMA on x86_64
  - NEON on ARM64
  - Scalar fallback
- Python bindings using [nanobind](https://nanobind.readthedocs.io/)
- GoogleTest unit tests with `AddressSanitizer` and `UndefinedBehaviorSanitizer`
- C++ and Python benchmarks
- Comparison against brute-force search and [hnswlib](https://github.com/nmslib/hnswlib)
- `clang-format` and `clang-tidy` integration
- Python packaging with `uv`, scikit-build-core, CMake, and nanobind
- Python 3.12+ with `abi3` wheels

### Prerequisites

#### C++

- CMake >= 3.18
- C++17 compiler
- clang-format
- clang-tidy

#### Python

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

Development dependencies are managed through `uv`.

Install development tools and dependencies:

```sh
make setup
make python
```

### Development

The Makefile provides the main development interface:

```sh
$ make help

Proxima

Build:
  make setup                 Install development tools
  make configure             Configure CMake
  make build                 Build C++ project
  make rebuild               Clean rebuild
  make clean                 Remove build directory

Tests:
  make test                  Run C++ tests
  make python-test           Run Python tests
  make python-all-test       Run all Python tests
  make ci                    Run full CI checks

Python:
  make python                Install Python package/dependencies
  make package               Build wheel + sdist
  make wheel                 Build wheel only
  make sdist                 Build source distribution
  make package-check         Validate distributions
  make package-local         Test wheel in clean local venv
  make package-test          Alias for package-local

Publishing:
  make publish-test TOKEN=... Upload to TestPyPI
  make publish TOKEN=...      Upload to PyPI

Benchmarks:
  make cppbench              Run C++ benchmark
  make pybindings-bench      Run Python bindings benchmark
  make pybench               Run hnswlib benchmark
  make bench                 Run complete benchmark suite
  make compare               Generate comparison
  make plot                  Generate plots

Code Quality:
  make format                Format C++ code
  make format-check          Check formatting
  make lint                  Run clang-tidy
  make lint-fix              Run clang-tidy with fixes

Release:
  make release-check         Run all release checks

Benchmark example:
  make bench RUN_DIR=my-run
```

Build, test, format, lint, benchmark, and packaging workflows are all available through `make`.

### Benchmarks

Run the complete benchmark suite:

```sh
make bench
```

This runs:

- C++ Proxima vs brute-force
- Python bindings vs brute-force
- Python hnswlib vs brute-force
- Comparison reports
- Plots

Results are stored under:

```text
benchmarks/results/<run-directory>/
```

A custom run directory can be specified with:

```sh
make bench RUN_DIR=my-run
```

### Python Package


```sh
# Build the source distribution
make package

# Validate the distributions
make package-check

# Test the actual built wheel in a clean temporary virtual environment (which is automatically removed after the test completes)
make package-test
```

#### TestPyPI

Test the complete publishing workflow before releasing to PyPI:

```sh

# This :

# 1. Builds the package.
# 2. Uploads it to [TestPyPI](https://test.pypi.org/).
# 3. Creates a temporary virtual environment.
# 4. Installs `proxima` from TestPyPI.
# 5. Installs the test dependencies from PyPI.
# 6. Runs the Python test suite.
# 7. Removes the temporary environment.

make publish-test TOKEN=pypi-<testpypi-token>
```

> [!NOTE]
> TestPyPI does not allow a previously used distribution filename to be uploaded again, even if the previous release was deleted. If `0.1.0` has already been uploaded, bump the version to `0.1.1` before publishing another build.

### Publishing

Releases are published to PyPI through GitHub Actions.

Create a version tag :

```sh
git tag v0.1.1
git push origin v0.1.1
```

The release workflow :

1. Builds wheels and the source distribution on Linux and macOS.
2. Creates a GitHub Release.
3. Publishes the distributions to PyPI.

The GitHub Actions workflow uses **PyPI Trusted Publishing**, so a PyPI API token is not required in repository secrets.

For local publishing, `uv` can be used directly:

```sh
uv publish \
    --token "$PYPI_TOKEN" \
    dist/*
```

> [!NOTE]
> TestPyPI and PyPI use separate credentials. A TestPyPI token cannot be used to publish to PyPI.

### Python Compatibility

Proxima uses nanobind's stable Python ABI:

```text
cp312-abi3
```

The package targets **Python 3.12+**, allowing the same `abi3` wheel to be used across supported Python versions on the same platform and architecture.

### License

MIT