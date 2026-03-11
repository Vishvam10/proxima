import csv
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

Key = Tuple[str, int, int, int]  # (distance, dataset, dim, k)


def load_cpp_results(path: Path) -> Dict[str, Dict[Key, Dict]]:
    """Returns dict keyed by simd_mode, each containing results keyed by (distance, dataset, dim, k)."""
    results: Dict[str, Dict[Key, Dict]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            simd_mode = row["simd_mode"]
            key: Key = (
                row["distance"],
                int(row["dataset"]),
                int(row["dim"]),
                int(row["k"]),
            )
            entry = {
                "build_s": float(row["build_s"]),
                "query_us": float(row["query_us"]),
                "brute_query_us": float(row["brute_query_us"]),
                "speedup": float(row["speedup"]),
                "recall": float(row["recall"]),
            }
            if simd_mode not in results:
                results[simd_mode] = {}
            results[simd_mode][key] = entry
    return results


def load_py_results(path: Path) -> Dict[Key, Dict]:
    results: Dict[Key, Dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key: Key = (
                row["distance"],
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


def pct_delta(test_val: float, base_val: float) -> float:
    """Percentage change from base to test. Negative means test is faster/smaller."""
    if base_val == 0:
        return float("inf")
    return (test_val / base_val - 1.0) * 100.0


def fmt_pct(delta: float, lower_is_better: bool = True) -> str:
    """Format percentage with indicator. For times, lower is better (green = negative delta)."""
    if lower_is_better:
        indicator = "[FASTER]" if delta < 0 else "[SLOWER]"
    else:
        indicator = "[BETTER]" if delta > 0 else "[WORSE]"
    return f"{delta:+.1f}% {indicator}"


def machine_info() -> str:
    lines = [
        f"- **OS**: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- **Processor**: {platform.processor() or 'N/A'}",
        f"- **Python**: {platform.python_version()}",
        f"- **Platform**: {platform.platform()}",
    ]
    return "\n".join(lines)


def compare_section(
    title: str,
    cpp_results: Dict[Key, Dict],
    py_results: Dict[Key, Dict],
    cpp_label: str,
) -> list:
    """Generate markdown comparison section."""
    lines = [
        f"## {title}",
        "",
        f"| Distance | Dataset | Dim | K | Python Build (s) | {cpp_label} Build (s) | Build Delta | Python Query (us) | {cpp_label} Query (us) | Query Delta | Python Recall | {cpp_label} Recall |",
        "|----------|---------|-----|---|-----------------|----------------------|-------------|-------------------|----------------------|-------------|---------------|-----------------|",
    ]

    common_keys = sorted(set(cpp_results.keys()) & set(py_results.keys()))

    for key in common_keys:
        dist, n, d, k = key
        cpp = cpp_results[key]
        py = py_results[key]

        build_delta = pct_delta(cpp["build_s"], py["build_s"])
        query_delta = pct_delta(cpp["query_us"], py["query_us"])

        lines.append(
            f"| {dist} | {n} | {d} | {k} "
            f"| {py['build_s']:.4f} | {cpp['build_s']:.4f} | {fmt_pct(build_delta)} "
            f"| {py['query_us']:.3f} | {cpp['query_us']:.3f} | {fmt_pct(query_delta)} "
            f"| {py['recall']:.4f} | {cpp['recall']:.4f} |"
        )

    return lines


def summary_table(
    cpp_by_mode: Dict[str, Dict[Key, Dict]],
    py_results: Dict[Key, Dict],
) -> list:
    """Generate a summary table showing average improvements across all scenarios."""
    lines = [
        "## Summary (Average Percentage Improvements)",
        "",
        "| Mode | Avg Build Delta | Avg Query Delta |",
        "|------|-----------------|-----------------|",
    ]

    mode_labels = {
        "scalar": "C++ Scalar",
        "simd": "C++ SIMD",
        "simd_mt": "C++ SIMD + MT",
    }

    for mode in ["scalar", "simd", "simd_mt"]:
        if mode not in cpp_by_mode:
            continue

        cpp_results = cpp_by_mode[mode]
        common_keys = set(cpp_results.keys()) & set(py_results.keys())

        if not common_keys:
            continue

        build_deltas = []
        query_deltas = []

        for key in common_keys:
            cpp = cpp_results[key]
            py = py_results[key]
            build_deltas.append(pct_delta(cpp["build_s"], py["build_s"]))
            query_deltas.append(pct_delta(cpp["query_us"], py["query_us"]))

        avg_build = sum(build_deltas) / len(build_deltas)
        avg_query = sum(query_deltas) / len(query_deltas)

        label = mode_labels.get(mode, mode)
        lines.append(f"| {label} | {fmt_pct(avg_build)} | {fmt_pct(avg_query)} |")

    return lines


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: compare.py <results_dir_name>")

    results_name = sys.argv[1]
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results" / results_name
    results_dir.mkdir(parents=True, exist_ok=True)

    cpp_path = base_dir / "cpp_results.csv"
    py_path = base_dir / "python_results.csv"

    if not cpp_path.exists():
        raise SystemExit(f"Missing C++ results: {cpp_path}")
    if not py_path.exists():
        raise SystemExit(f"Missing Python results file: {py_path}")

    cpp_by_mode = load_cpp_results(cpp_path)
    py = load_py_results(py_path)

    if not cpp_by_mode:
        raise SystemExit("No C++ benchmark results found.")
    if not py:
        raise SystemExit("No Python benchmark results found.")

    md_lines = [
        "# Benchmark Comparison Results",
        "",
        f"**Date**: {results_name}",
        "",
        "## Machine Info",
        "",
        machine_info(),
        "",
    ]

    md_lines.extend(summary_table(cpp_by_mode, py))
    md_lines.append("")

    if "scalar" in cpp_by_mode:
        md_lines.extend(compare_section(
            "Python vs C++ Scalar",
            cpp_by_mode["scalar"],
            py,
            "C++ Scalar"
        ))
        md_lines.append("")

    if "simd" in cpp_by_mode:
        md_lines.extend(compare_section(
            "Python vs C++ SIMD",
            cpp_by_mode["simd"],
            py,
            "C++ SIMD"
        ))
        md_lines.append("")

    if "simd_mt" in cpp_by_mode:
        md_lines.extend(compare_section(
            "Python vs C++ Multithreaded + SIMD",
            cpp_by_mode["simd_mt"],
            py,
            "C++ SIMD+MT"
        ))
        md_lines.append("")

    csv_rows = []
    for mode, cpp_results in cpp_by_mode.items():
        common_keys = sorted(set(cpp_results.keys()) & set(py.keys()))
        for key in common_keys:
            dist, n, d, k = key
            cpp = cpp_results[key]
            p = py[key]

            csv_rows.append({
                "distance": dist,
                "simd_mode": mode,
                "dataset": n,
                "dim": d,
                "k": k,
                "build_py_s": p["build_s"],
                "build_cpp_s": cpp["build_s"],
                "build_delta_pct": pct_delta(cpp["build_s"], p["build_s"]),
                "query_py_us": p["query_us"],
                "query_cpp_us": cpp["query_us"],
                "query_delta_pct": pct_delta(cpp["query_us"], p["query_us"]),
                "recall_py": p["recall"],
                "recall_cpp": cpp["recall"],
            })

    csv_path = results_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "distance", "simd_mode", "dataset", "dim", "k",
            "build_py_s", "build_cpp_s", "build_delta_pct",
            "query_py_us", "query_cpp_us", "query_delta_pct",
            "recall_py", "recall_cpp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    readme_path = results_dir / "README.md"
    readme_path.write_text("\n".join(md_lines))

    shutil.copy(cpp_path, results_dir / "cpp_results.csv")
    shutil.copy(py_path, results_dir / "python_results.csv")

    print(f"\nResults saved to {results_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {readme_path.name}")
    print(f"  - cpp_results.csv (copy)")
    print(f"  - python_results.csv (copy)")


if __name__ == "__main__":
    main()
