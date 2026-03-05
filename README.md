## Proxima

A C++17 implementation of the Hierarchical Navigable Small World (HNSW) approximate nearest neighbor index. It is designed for simplicity and clarity, making it a good starting point for HNSW research and production-level experimentation

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
make build
```

### Running Tests

```bash
make test
```

### Running Benchmark

```bash
# C++ benchmark only
make cppbench

# Python benchmark only (auto-creates venv)
make pybench

# Run both and compare
make bench
```

### Benchmarks

**System Information**

| Property         | Value          |
| ---------------- | -------------- |
| Operating System | macOS 14.6     |
| Architecture     | arm64          |
| CPU              | arm            |
| CPU Cores        | 8 (logical: 8) |
| Memory           | 24.0 GB        |
| Disk             | 460.4 GB       |
| Python Version   | 3.14.0         |

#### Results

We benchmark our `HnswCPU` implementation vs a brute-force approach towards ANN search and here are the results:

| dataset | dim | k   | build_s | query_us | brute_query_us | speedup | recall |
| ------- | --- | --- | ------- | -------- | -------------- | ------- | ------ |
| 1000    | 128 | 10  | 4.82842 | 153.11   | 541.06         | 3.5x    | 1      |
| 5000    | 64  | 10  | 22.0749 | 175.76   | 1421.1         | 8.1x    | 1      |
| 10000   | 32  | 5   | 33.9611 | 124.68   | 1794.08        | 14.4x   | 1      |
| 50000   | 64  | 10  | 237.215 | 302.02   | 14316.2        | 47.4x   | 0.98   |
| 100000  | 32  | 10  | 377.432 | 275.72   | 16240.8        | 58.9x   | 1      |

Even with our straightforward `HnswCPU` implementation, we see **orders-of-magnitude speedups over brute-force search**, while maintaining **near-perfect recall**. For small datasets, the speedup is noticeable, but for larger datasets, the performance gains are dramatic — reducing query times from tens of milliseconds to just a few.

<br>

But, this is nothing when we compare it to [hnswlib](https://github.com/nmslib/hnswlib). This is a header-only library written in `C++` and has `Python` bindings. This is how that performs:

| dataset | dim | k   | build_s  | query_us | brute_query_us | speedup | recall |
| ------- | --- | --- | -------- | -------- | -------------- | ------- | ------ |
| 1000    | 128 | 10  | 0.043238 | 30.8108  | 82.1614        | 2.7x    | 1.0    |
| 5000    | 64  | 10  | 0.166046 | 25.0196  | 198.0519       | 7.9x    | 1.0    |
| 10000   | 32  | 5   | 0.229604 | 18.0888  | 270.9317       | 15.0x   | 1.0    |
| 50000   | 64  | 10  | 3.222314 | 47.6003  | 2097.3110      | 44.1x   | 0.99   |
| 100000  | 32  | 10  | 3.936344 | 33.1593  | 2716.1694      | 82.0x   | 1.0    |

> [!NOTE]
> `hnswlib` leverages SIMD, multi-threading, etc for extra speed.

A basic SIMD implementation is available in the `wip-simd-support` branch, but it is experimental and may produce worse results in some cases.

#### Note

- Benchmarks include both small and large datasets
- All recall values are measured against exact nearest neighbor search
