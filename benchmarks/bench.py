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

DISTANCES = [
    ("l2", "l2"),
    ("ip", "inner_product"),
    ("cosine", "cosine"),
]

COLS = [
    ("Distance", 14),
    ("Dataset", 10),
    ("Dim", 6),
    ("K", 4),
    ("Build(s)", 10),
    ("Query(us)", 11),
    ("Brute(us)", 11),
    ("Speedup", 9),
    ("Recall", 8),
]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "python_results.csv"

    print()
    print("Python Benchmarks\n")

    header = "".join(f"{name:<{w}}" for name, w in COLS)
    print(header)
    print("-" * sum(w for _, w in COLS))

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "distance",
                "dataset",
                "dim",
                "k",
                "build_s",
                "query_us",
                "brute_query_us",
                "speedup",
                "recall",
            ]
        )

        for space, dist_label in DISTANCES:
            for N, DIM, K in SCENARIOS:
                np.random.seed(42)
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

                build_i = round(build_time)
                query_i = round(query_time)
                brute_i = round(brute_query_time)
                speed_i = round(speedup)
                recall_i = round(recall * 100)

                row = [
                    dist_label,
                    N,
                    DIM,
                    K,
                    build_i,
                    query_i,
                    brute_i,
                    speed_i,
                    recall_i,
                ]

                line = "".join(f"{str(val):<{w}}" for val, (_, w) in zip(row, COLS))

                print(line)

                writer.writerow(
                    [
                        dist_label,
                        N,
                        DIM,
                        K,
                        build_time,
                        query_time,
                        brute_query_time,
                        speedup,
                        recall,
                    ]
                )

    print()
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
