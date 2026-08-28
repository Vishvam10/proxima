import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from proxima import DistanceType, HnswCPU


@dataclass(frozen=True)
class Scenario:
    N: int
    DIM: int
    K: int


SCENARIOS = [
    # Small datasets
    Scenario(1000, 64, 5),
    Scenario(1000, 64, 10),
    Scenario(1000, 64, 50),
    Scenario(1000, 128, 5),
    Scenario(1000, 128, 10),
    Scenario(1000, 128, 50),
    Scenario(1000, 256, 5),
    Scenario(1000, 256, 10),
    Scenario(1000, 256, 50),
    # Medium datasets
    Scenario(5000, 64, 5),
    Scenario(5000, 64, 10),
    Scenario(5000, 64, 50),
    Scenario(5000, 128, 5),
    Scenario(5000, 128, 10),
    Scenario(5000, 128, 50),
    Scenario(5000, 256, 5),
    Scenario(5000, 256, 10),
    Scenario(5000, 256, 50),
    Scenario(10000, 64, 5),
    Scenario(10000, 64, 10),
    Scenario(10000, 64, 50),
    Scenario(10000, 128, 5),
    Scenario(10000, 128, 10),
    Scenario(10000, 128, 50),
    Scenario(10000, 256, 5),
    Scenario(10000, 256, 10),
    Scenario(10000, 256, 50),
    # Large datasets
    Scenario(50000, 64, 5),
    Scenario(50000, 64, 10),
    Scenario(50000, 64, 50),
    Scenario(50000, 128, 5),
    Scenario(50000, 128, 10),
    Scenario(50000, 128, 50),
    Scenario(50000, 256, 5),
    Scenario(50000, 256, 10),
    Scenario(50000, 256, 50),
    Scenario(100000, 64, 5),
    Scenario(100000, 64, 10),
    Scenario(100000, 64, 50),
    Scenario(100000, 128, 5),
    Scenario(100000, 128, 10),
    Scenario(100000, 128, 50),
    Scenario(100000, 256, 5),
    Scenario(100000, 256, 10),
    Scenario(100000, 256, 50),
    # Extra large datasets
    Scenario(500000, 64, 5),
    Scenario(500000, 64, 10),
    Scenario(500000, 64, 50),
    Scenario(500000, 128, 5),
    Scenario(500000, 128, 10),
    Scenario(500000, 128, 50),
    Scenario(500000, 256, 5),
    Scenario(500000, 256, 10),
    Scenario(500000, 256, 50),
]


def brute_force_knn(data: np.ndarray, query: np.ndarray, k: int):
    distances = np.sum(
        (data - query) ** 2,
        axis=1,
    )

    return np.argpartition(
        distances,
        k - 1,
    )[:k]


def print_header():
    print(
        "+----------------------+--------+--------+------+--------------+"
        "--------------+--------------+------------+----------+"
    )
    print(
        "| Mode                 | N      | Dim    | K    | Build(s)     |"
        " Query(us)    | Brute(us)    | Speedup    | Recall   |"
    )
    print(
        "+----------------------+--------+--------+------+--------------+"
        "--------------+--------------+------------+----------+"
    )


def print_row(
    mode,
    scenario,
    build_s,
    query_us,
    brute_us,
    speedup,
    recall,
):
    print(
        f"| {mode:<20} | "
        f"{scenario.N:<6} | "
        f"{scenario.DIM:<6} | "
        f"{scenario.K:<4} | "
        f"{build_s:<12.2f} | "
        f"{query_us:<12.2f} | "
        f"{brute_us:<12.2f} | "
        f"{speedup:<9.2f}x | "
        f"{recall:<8.4f} |"
    )


def benchmark_mode(
    mode: str,
    force_scalar: bool,
    writer,
    rng,
):
    for scenario in SCENARIOS:
        data = rng.random(
            (scenario.N, scenario.DIM),
            dtype=np.float32,
        )

        index = HnswCPU(
            M=16,
            ef_construction=200,
            seed=42,
            distance_type=DistanceType.L2,
            force_scalar=force_scalar,
        )

        start = time.perf_counter()

        index.create(data)

        build_s = time.perf_counter() - start

        qcount = min(100, scenario.N)
        correct = 0

        start = time.perf_counter()

        for i in range(qcount):
            result = index.search(
                data[i],
                scenario.K,
                200,
            )

            if int(i) in result:
                correct += 1

        query_us = (time.perf_counter() - start) * 1_000_000 / qcount
        recall = correct / qcount

        start = time.perf_counter()

        for i in range(qcount):
            brute_force_knn(
                data,
                data[i],
                scenario.K,
            )

        brute_us = (time.perf_counter() - start) * 1_000_000 / qcount
        speedup = brute_us / query_us

        print_row(
            mode,
            scenario,
            build_s,
            query_us,
            brute_us,
            speedup,
            recall,
        )

        writer.writerow(
            {
                "impl": mode,
                "N": scenario.N,
                "DIM": scenario.DIM,
                "K": scenario.K,
                "build_s": build_s,
                "query_us": query_us,
                "brute_us": brute_us,
                "speedup": speedup,
                "recall": recall,
            }
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "out_dir",
        nargs="?",
        default="benchmarks/results",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = out_dir / "python_bindings_results.csv"

    print("\n\nPython bindings benchmarks\n")

    print_header()

    rng = np.random.default_rng(42)

    with output.open("w", newline="") as f:
        fieldnames = [
            "impl",
            "N",
            "DIM",
            "K",
            "build_s",
            "query_us",
            "brute_us",
            "speedup",
            "recall",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        benchmark_mode(
            "python_cpp_scalar",
            True,
            writer,
            rng,
        )

        benchmark_mode(
            "python_cpp_simd",
            False,
            writer,
            rng,
        )

    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
