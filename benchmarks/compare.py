import csv
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
                "brute_query_us": float(row["brute_query_us"]),
                "speedup": float(row["speedup"]),
                "recall": float(row["recall"]),
            }
    return results


def percentage_str(cpp_val: float, py_val: float) -> str:
    if py_val == 0:
        return "inf"
    diff = (cpp_val / py_val - 1.0) * 100
    sign = "-" if diff >= 0 else "+"
    return f"{sign}{abs(diff):.1f}%"


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    cpp_path = base_dir / "cpp_results.csv"
    py_path = base_dir / "python_results.csv"
    output_csv = base_dir / "comparison.csv"

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

    # Warn about missing scenarios
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
        f"{'Build C++ (s)':>13} | {'Build Py (s)':>12} | {'ΔBuild':>7} | "
        f"{'Query C++ (us)':>14} | {'Query Py (us)':>13} | {'ΔQuery':>7} | "
        f"{'Brute C++ (us)':>14} | {'Brute Py (us)':>13} | "
        f"{'Speedup C++':>11} | {'Speedup Py':>10} | "
        f"{'Recall C++':>10} | {'Recall Py':>9}"
    )
    print()
    print(header)
    print("-" * len(header))

    csv_rows = []
    for n, d, k in common:
        c = cpp[(n, d, k)]
        p = py[(n, d, k)]

        build_pct = percentage_str(c["build_s"], p["build_s"])
        query_pct = percentage_str(c["query_us"], p["query_us"])

        print(
            f"{n:7d} | {d:4d} | {k:3d} | "
            f"{c['build_s']:13.4f} | {p['build_s']:12.4f} | {build_pct:>7} | "
            f"{c['query_us']:14.3f} | {p['query_us']:13.3f} | {query_pct:>7} | "
            f"{c['brute_query_us']:14.3f} | {p['brute_query_us']:13.3f} | "
            f"{c['speedup']:10.2f}x | {p['speedup']:9.2f}x | "
            f"{c['recall']:10.4f} | {p['recall']:9.4f}"
        )

        csv_rows.append({
            "dataset": n,
            "dim": d,
            "k": k,
            "build_cpp_s": c["build_s"],
            "build_py_s": p["build_s"],
            "build_pct": build_pct,
            "query_cpp_us": c["query_us"],
            "query_py_us": p["query_us"],
            "query_pct": query_pct,
            "brute_cpp_us": c["brute_query_us"],
            "brute_py_us": p["brute_query_us"],
            "speedup_cpp": c["speedup"],
            "speedup_py": p["speedup"],
            "recall_cpp": c["recall"],
            "recall_py": p["recall"],
        })

    with output_csv.open("w", newline="") as f:
        fieldnames = [
            "dataset", "dim", "k",
            "build_cpp_s", "build_py_s", "build_pct",
            "query_cpp_us", "query_py_us", "query_pct",
            "brute_cpp_us", "brute_py_us",
            "speedup_cpp", "speedup_py",
            "recall_cpp", "recall_py",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nComparison exported to {output_csv}")


if __name__ == "__main__":
    main()