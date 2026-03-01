import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


Key = Tuple[int, int, int]


def load_results(path: Path) -> Dict[Key, Dict[str, float]]:
    results: Dict[Key, Dict[str, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key: Key = (
                int(row["dataset"]),
                int(row["dim"]),
                int(row["k"]),
            )
            results[key] = {
                "build_s": float(row["build_s"]),
                "query_us": float(row["query_us"]),
                "recall": float(row["recall"]),
            }
    return results


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    cpp_path = base_dir / "cpp_results.csv"
    py_path = base_dir / "python_results.csv"

    if not cpp_path.exists():
        raise SystemExit(f"Missing C++ results file: {cpp_path}")
    if not py_path.exists():
        raise SystemExit(f"Missing Python results file: {py_path}")

    cpp = load_results(cpp_path)
    py = load_results(py_path)

    cpp_keys = set(cpp.keys())
    py_keys = set(py.keys())
    common = sorted(cpp_keys & py_keys)

    if not common:
        raise SystemExit("No common benchmark scenarios between C++ and Python results.")

    missing_cpp = py_keys - cpp_keys
    missing_py = cpp_keys - py_keys

    if missing_cpp:
        print("Warning: scenarios present only in Python results (ignored):")
        for n, d, k in sorted(missing_cpp):
            print(f"  dataset={n}, dim={d}, k={k}")
        print()

    if missing_py:
        print("Warning: scenarios present only in C++ results (ignored):")
        for n, d, k in sorted(missing_py):
            print(f"  dataset={n}, dim={d}, k={k}")
        print()

    header = (
        f"{'Dataset':>7} | {'Dim':>4} | {'K':>3} | "
        f"{'Build C++ (s)':>13} | {'Build Py (s)':>12} | {'C++/Py':>7} | "
        f"{'Query C++ (us)':>14} | {'Query Py (us)':>13} | {'C++/Py':>7} | "
        f"{'Recall C++':>10} | {'Recall Py':>9}"
    )
    print()
    print(header)
    print("-" * len(header))

    for n, d, k in common:
        c = cpp[(n, d, k)]
        p = py[(n, d, k)]

        build_ratio = c["build_s"] / p["build_s"] if p["build_s"] > 0 else float("inf")
        query_ratio = c["query_us"] / p["query_us"] if p["query_us"] > 0 else float("inf")

        print(
            f"{n:7d} | {d:4d} | {k:3d} | "
            f"{c['build_s']:13.4f} | {p['build_s']:12.4f} | {build_ratio:7.3f} | "
            f"{c['query_us']:14.3f} | {p['query_us']:13.3f} | {query_ratio:7.3f} | "
            f"{c['recall']:10.4f} | {p['recall']:9.4f}"
        )

    print()


if __name__ == "__main__":
    main()

