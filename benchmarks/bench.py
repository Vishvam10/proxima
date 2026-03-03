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
    print("Python Benchmarks (hnswlib)")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "dataset", "dim", "k", "build_s", "query_us", "recall"])

        for space, label in DISTANCES:
            print(f"\n[{label}]")
            print(
                f"{'Dataset':>7} | {'Dim':>3} | {'K':>2} | "
                f"{'Build (s)':>9} | {'Query (us)':>10} | {'Recall':>6}"
            )
            print("-" * 60)

            np.random.seed(42)

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

                print(
                    f"{N:7} | {DIM:3} | {K:2} | "
                    f"{build_time:9.4f} | {query_time:10.3f} | {recall:6.4f}"
                )

                writer.writerow([label, N, DIM, K, build_time, query_time, recall])

    print()


if __name__ == "__main__":
    main()
