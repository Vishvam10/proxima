import csv
import sys
import time
import hnswlib
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = BASE / "results" / run_dir
OUT.mkdir(exist_ok=True, parents=True)
csv_path = OUT / "python_results.csv"

SCENARIOS = [
    # Small datasets
    (1000, 64, 5), (1000, 64, 10), (1000, 64, 50),
    (1000, 128, 5), (1000, 128, 10), (1000, 128, 50),
    (1000, 256, 5), (1000, 256, 10), (1000, 256, 50),
    # Medium datasets
    (5000, 64, 5), (5000, 64, 10), (5000, 64, 50),
    (5000, 128, 5), (5000, 128, 10), (5000, 128, 50),
    (5000, 256, 5), (5000, 256, 10), (5000, 256, 50),
    (10000, 64, 5), (10000, 64, 10), (10000, 64, 50),
    (10000, 128, 5), (10000, 128, 10), (10000, 128, 50),
    (10000, 256, 5), (10000, 256, 10), (10000, 256, 50),
    # Large datasets
    (50000, 64, 5), (50000, 64, 10), (50000, 64, 50),
    (50000, 128, 5), (50000, 128, 10), (50000, 128, 50),
    (50000, 256, 5), (50000, 256, 10), (50000, 256, 50),
    (100000, 64, 5), (100000, 64, 10), (100000, 64, 50),
    (100000, 128, 5), (100000, 128, 10), (100000, 128, 50),
    (100000, 256, 5), (100000, 256, 10), (100000, 256, 50),
    # Extra-large datasets
    (500000, 64, 5), (500000, 64, 10), (500000, 64, 50),
    (500000, 128, 5), (500000, 128, 10), (500000, 128, 50),
    (500000, 256, 5), (500000, 256, 10), (500000, 256, 50),
]


def brute_force_knn(data, query, k):
    dists = np.sum((data - query) ** 2, axis=1)
    return np.argsort(dists)[:k]


def print_table_header():
    print(
        "+------------+--------+--------+------+--------------+--------------+--------------+------------+----------+"
    )
    print(
        "| Mode       | N      | Dim    | K    | Build(s)    | Query(us)    | Brute(us)    | Speedup    | Recall   |"
    )
    print(
        "+------------+--------+--------+------+--------------+--------------+--------------+------------+----------+"
    )


def print_table_footer():
    print(
        "+------------+--------+--------+------+--------------+--------------+--------------+------------+----------+"
    )


def print_table_row(mode, N, DIM, K, build_s, query_us, brute_us, speedup, recall):
    # Format speedup with x inside column
    speedup_str = f"{speedup:.2f}x"
    print(
        f"| {mode:<10} | {N:<6} | {DIM:<6} | {K:<4} | "
        f"{build_s:<12.2f} | {query_us:<12.2f} | {brute_us:<12.2f} | "
        f"{speedup_str:<10} | {recall:<8.2f} |",
        flush=True
    )


def main():
    print("\n\nPython Benchmarks\n\n", flush=True)
    print_table_header()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["impl", "N", "DIM", "K", "build_s", "query_us", "brute_us", "speedup", "recall"]
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

            # CSV
            writer.writerow(["python", N, DIM, K, build_s, query_us, brute_us, speedup, recall])

            # Print row live
            print_table_row("python", N, DIM, K, build_s, query_us, brute_us, speedup, recall)

    print_table_footer()
    print(f"\nSaved {csv_path}", flush=True)


if __name__ == "__main__":
    main()