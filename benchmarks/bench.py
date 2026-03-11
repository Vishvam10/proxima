import csv
import time
from pathlib import Path

import hnswlib
import numpy as np

SCENARIOS = [
    (1000, 128, 10),
    (5000, 64, 10),
    (10000, 32, 5),
    (50000, 64, 10),
    (100000, 32, 10),
]

# (hnswlib space name, our label)
DISTANCES = [
    ("l2", "l2"),
    ("ip", "inner_product"),
    ("cosine", "cosine"),
]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "python_results.csv"

    print()
    print("Python Benchmarks")
    print(
        f"{'Dataset':>6} | {'Dim':>3} | {'K':>2} | "
        f"{'Build (s)':>9} | {'Query (us)':>10} | "
        f"{'Brute (us)':>10} | {'Speedup':>7} | {'Recall':>6}"
    )
    print("-" * 82)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "dim", "k", "build_s", "query_us",
             "brute_query_us", "speedup", "recall"]
        )

            for N, DIM, K in SCENARIOS:
                data = np.random.rand(N, DIM).astype(np.float32)

                index = hnswlib.Index(space=space, dim=DIM)
                index.init_index(max_elements=N, ef_construction=200, M=16)

                t1 = time.time()
                index.add_items(data)
                t2 = time.time()
                build_time = t2 - t1

                index.set_ef(200)
                q = data[: min(100, N)]

            t3 = time.time()
            labels, _ = index.knn_query(q, k=K)
            t4 = time.time()
            query_time = (t4 - t3) / len(q) * 1e6

            recall = float(np.mean([i in labels[i] for i in range(len(q))]))

            t5 = time.time()
            for i in range(len(q)):
                dists = np.sum((data - q[i]) ** 2, axis=1)
                np.argpartition(dists, K)[:K]
            t6 = time.time()
            brute_query_time = (t6 - t5) / len(q) * 1e6

            speedup = brute_query_time / query_time

            print(
                f"{N:6} | {DIM:3} | {K:2} | "
                f"{build_time:9.4f} | {query_time:10.3f} | "
                f"{brute_query_time:10.3f} | {speedup:6.2f}x | {recall:6.4f}"
            )

            writer.writerow(
                [N, DIM, K, build_time, query_time,
                 brute_query_time, speedup, recall]
            )

    print()


if __name__ == "__main__":
    main()
