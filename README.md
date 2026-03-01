## Proxima

A C++17 implementation of the Hierarchical Navigable Small World (HNSW)
approximate nearest neighbor index.

### Features

- Multi-layer HNSW graph construction
- L2 distance search
- Configurable M and efConstruction
- GoogleTest unit tests
- Benchmark harness
- Python comparison against hnswlib

### Build Instructions

Requires `CMake >= 3.16` and a `C++17` compiler.

```bash
mkdir build
cd build
cmake ..
make -j
```

### Running Tests

```bash
cd build
ctest

or

./test_hnsw
```

### Running Benchmark

```bash
cd build
./bench

# Python Comparison (hnswlib)
pip install hnswlib numpy

cd python
python benchmark_hnswlib.py
```

### Notes

This implementation focuses on clarity and correctness.
It does not include SIMD, memory pooling, or heuristic neighbor
selection optimizations
