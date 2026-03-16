import csv
import time
import hnswlib
import numpy as np
from pathlib import Path

OUT = Path("benchmarks/results")
OUT.mkdir(exist_ok=True)
csv_path = OUT / "python_results.csv"

SCENARIOS = [
    # Small datasets
    (1000, 128, 10),
    (5000, 128, 10),
    (10000, 128, 10),
    (5000, 64, 10),
    (5000, 256, 10),
    (5000, 128, 5),
    (5000, 128, 50),

    # Medium datasets
    (50000, 128, 10),
    (100000, 128, 10),
    (100000, 64, 10),
    (100000, 256, 10),
    (100000, 128, 50),

    # Large datasets
    (500000, 128, 10),
    (1000000, 128, 10),
    (1000000, 64, 10),
    (1000000, 256, 10),
    (1000000, 128, 50),

    # Extra-large datasets
    (5000000, 128, 10),
    (10000000, 128, 10)
]


def brute_force_knn(data, query, k):
    dists = np.sum((data - query) ** 2, axis=1)
    return np.argsort(dists)[:k]


def print_table(data):
    # Header
    print(
        "+------------+--------+--------+------+--------------+--------------+--------------+------------+----------+"
    )
    print(
        "| Mode       | N      | Dim    | K    | Build(s)    | Query(us)    | Brute(us)    | Speedup    | Recall   |"
    )
    print(
        "+------------+--------+--------+------+--------------+--------------+--------------+------------+----------+"
    )

    # Rows
    for row in data:
        mode, N, DIM, K, build_s, query_us, brute_us, speedup, recall = row
        print(
            f"| {mode:<10} | {N:<6} | {DIM:<6} | {K:<4} | {build_s:<12.2f} | {query_us:<12.2f} | {brute_us:<12.2f} | {speedup:<10.2f} | {recall:<8.2f} |"
        )

    # Footer
    print(
        "+------------+--------+--------+------+--------------+--------------+--------------+------------+----------+"
    )


def main():
    data_for_table = []

    print("\n\nPython Benchmarks\n\n")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )

        for N, DIM, K in SCENARIOS:
            np.random.seed(42)
            data = np.random.rand(N, DIM).astype(np.float32)

            # Build HNSW
            index = hnswlib.Index(space="l2", dim=DIM)
            index.init_index(max_elements=N, M=16, ef_construction=200)
            t1 = time.time()
            index.add_items(data)
            t2 = time.time()
            build_s = t2 - t1

            index.set_ef(200)
            q = data[: min(100, N)]

            # HNSW query
            t3 = time.time()
            labels, _ = index.knn_query(q, k=K)
            t4 = time.time()
            query_us = (t4 - t3) / len(q) * 1e6

            # Brute-force query
            t5 = time.time()
            for i in range(len(q)):
                brute_force_knn(data, q[i], K)
            t6 = time.time()
            brute_us = (t6 - t5) / len(q) * 1e6

            speedup = brute_us / query_us
            recall = np.mean([i in labels[i] for i in range(len(q))])

            writer.writerow(
                ["python", N, DIM, K, build_s, query_us, brute_us, speedup, recall]
            )
            data_for_table.append(
                ("python", N, DIM, K, build_s, query_us, brute_us, speedup, recall)
            )

    print_table(data_for_table)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
