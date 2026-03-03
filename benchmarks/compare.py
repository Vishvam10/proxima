import csv
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

Key = Tuple[str, int, int, int]  # (distance, dataset, dim, k)


def load_cpp_results(path: Path) -> Tuple[Dict[Key, Dict], Dict[Key, Dict]]:
    """Returns (simd_results, scalar_results) keyed by (distance, dataset, dim, k)."""
    simd: Dict[Key, Dict] = {}
    scalar: Dict[Key, Dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key: Key = (
                row["distance"],
                int(row["dataset"]),
                int(row["dim"]),
                int(row["k"]),
            )
            entry = {
                "build_s": float(row["build_s"]),
                "query_us": float(row["query_us"]),
                "recall": float(row["recall"]),
            }
            if row["simd_mode"] == "simd":
                simd[key] = entry
            else:
                scalar[key] = entry
    return simd, scalar


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
                "recall": float(row["recall"]),
            }
    return results


def pct_delta(test_val: float, base_val: float) -> float:
    """Percentage change from base to test. Negative means test is faster/smaller."""
    if base_val == 0:
        return float("inf")
    return (test_val / base_val - 1.0) * 100.0


def fmt_pct(delta: float, lower_is_better: bool = True) -> str:
    """Format percentage with emoji. For times, lower is better (green = negative delta)."""
    if lower_is_better:
        emoji = "🟢" if delta < 0 else "🔴"
    else:
        emoji = "🟢" if delta > 0 else "🔴"
    return f"{emoji} {delta:+.1f}%"


def machine_info() -> str:
    lines = [
        f"- **OS**: {platform.system()} {platform.release()} ({platform.machine()})",
        f"- **Processor**: {platform.processor() or 'N/A'}",
        f"- **Python**: {platform.python_version()}",
        f"- **Platform**: {platform.platform()}",
    ]
    return "\n".join(lines)


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
        raise SystemExit(f"Missing Python results: {py_path}")

    shutil.copy2(cpp_path, results_dir / "cpp_results.csv")
    shutil.copy2(py_path, results_dir / "python_results.csv")

    simd, scalar = load_cpp_results(cpp_path)
    py = load_py_results(py_path)

    all_keys = sorted(set(simd.keys()) | set(scalar.keys()) | set(py.keys()))

    if not all_keys:
        raise SystemExit("No benchmark results found.")

    # --- CSV output ---
    csv_path = results_dir / "comparison.csv"
    csv_fields = [
        "distance", "dataset", "dim", "k",
        "build_simd_s", "build_scalar_s", "build_py_s",
        "build_simd_vs_py_pct", "build_scalar_vs_py_pct", "build_simd_vs_scalar_pct",
        "query_simd_us", "query_scalar_us", "query_py_us",
        "query_simd_vs_py_pct", "query_scalar_vs_py_pct", "query_simd_vs_scalar_pct",
        "recall_simd", "recall_scalar", "recall_py",
    ]

    csv_rows: List[Dict] = []

    for key in all_keys:
        dist, n, d, k = key
        s = simd.get(key, {})
        sc = scalar.get(key, {})
        p = py.get(key, {})

        row = {"distance": dist, "dataset": n, "dim": d, "k": k}

        row["build_simd_s"] = s.get("build_s", "")
        row["build_scalar_s"] = sc.get("build_s", "")
        row["build_py_s"] = p.get("build_s", "")

        if s.get("build_s") is not None and p.get("build_s") is not None:
            row["build_simd_vs_py_pct"] = round(pct_delta(s["build_s"], p["build_s"]), 1)
        else:
            row["build_simd_vs_py_pct"] = ""

        if sc.get("build_s") is not None and p.get("build_s") is not None:
            row["build_scalar_vs_py_pct"] = round(pct_delta(sc["build_s"], p["build_s"]), 1)
        else:
            row["build_scalar_vs_py_pct"] = ""

        if s.get("build_s") is not None and sc.get("build_s") is not None:
            row["build_simd_vs_scalar_pct"] = round(pct_delta(s["build_s"], sc["build_s"]), 1)
        else:
            row["build_simd_vs_scalar_pct"] = ""

        row["query_simd_us"] = s.get("query_us", "")
        row["query_scalar_us"] = sc.get("query_us", "")
        row["query_py_us"] = p.get("query_us", "")

        if s.get("query_us") is not None and p.get("query_us") is not None:
            row["query_simd_vs_py_pct"] = round(pct_delta(s["query_us"], p["query_us"]), 1)
        else:
            row["query_simd_vs_py_pct"] = ""

        if sc.get("query_us") is not None and p.get("query_us") is not None:
            row["query_scalar_vs_py_pct"] = round(pct_delta(sc["query_us"], p["query_us"]), 1)
        else:
            row["query_scalar_vs_py_pct"] = ""

        if s.get("query_us") is not None and sc.get("query_us") is not None:
            row["query_simd_vs_scalar_pct"] = round(pct_delta(s["query_us"], sc["query_us"]), 1)
        else:
            row["query_simd_vs_scalar_pct"] = ""

        row["recall_simd"] = s.get("recall", "")
        row["recall_scalar"] = sc.get("recall", "")
        row["recall_py"] = p.get("recall", "")

        csv_rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    # --- README.md output ---
    readme_path = results_dir / "README.md"

    md_lines = [
        "# Benchmark Results",
        "",
        f"**Date**: {results_name}",
        "",
        "## Machine Info",
        "",
        machine_info(),
        "",
        "## Build Time Comparison",
        "",
        "| Distance | Dataset | Dim | K | C++ SIMD (s) | C++ Scalar (s) | Python (s) | SIMD vs Py | Scalar vs Py | SIMD vs Scalar |",
        "|----------|---------|-----|---|-------------|----------------|------------|------------|--------------|----------------|",
    ]

    for row in csv_rows:
        bs = row["build_simd_s"]
        bsc = row["build_scalar_s"]
        bp = row["build_py_s"]

        d_sp = fmt_pct(row["build_simd_vs_py_pct"], lower_is_better=True) if row["build_simd_vs_py_pct"] != "" else "N/A"
        d_scp = fmt_pct(row["build_scalar_vs_py_pct"], lower_is_better=True) if row["build_scalar_vs_py_pct"] != "" else "N/A"
        d_ss = fmt_pct(row["build_simd_vs_scalar_pct"], lower_is_better=True) if row["build_simd_vs_scalar_pct"] != "" else "N/A"

        md_lines.append(
            f"| {row['distance']} | {row['dataset']} | {row['dim']} | {row['k']} "
            f"| {bs:.4f} | {bsc:.4f} | {bp:.4f} "
            f"| {d_sp} | {d_scp} | {d_ss} |"
            if bs != "" and bsc != "" and bp != ""
            else f"| {row['distance']} | {row['dataset']} | {row['dim']} | {row['k']} "
            f"| {bs} | {bsc} | {bp} "
            f"| {d_sp} | {d_scp} | {d_ss} |"
        )

    md_lines += [
        "",
        "## Query Time Comparison",
        "",
        "| Distance | Dataset | Dim | K | C++ SIMD (us) | C++ Scalar (us) | Python (us) | SIMD vs Py | Scalar vs Py | SIMD vs Scalar |",
        "|----------|---------|-----|---|--------------|-----------------|-------------|------------|--------------|----------------|",
    ]

    for row in csv_rows:
        qs = row["query_simd_us"]
        qsc = row["query_scalar_us"]
        qp = row["query_py_us"]

        d_sp = fmt_pct(row["query_simd_vs_py_pct"], lower_is_better=True) if row["query_simd_vs_py_pct"] != "" else "N/A"
        d_scp = fmt_pct(row["query_scalar_vs_py_pct"], lower_is_better=True) if row["query_scalar_vs_py_pct"] != "" else "N/A"
        d_ss = fmt_pct(row["query_simd_vs_scalar_pct"], lower_is_better=True) if row["query_simd_vs_scalar_pct"] != "" else "N/A"

        md_lines.append(
            f"| {row['distance']} | {row['dataset']} | {row['dim']} | {row['k']} "
            f"| {qs:.3f} | {qsc:.3f} | {qp:.3f} "
            f"| {d_sp} | {d_scp} | {d_ss} |"
            if qs != "" and qsc != "" and qp != ""
            else f"| {row['distance']} | {row['dataset']} | {row['dim']} | {row['k']} "
            f"| {qs} | {qsc} | {qp} "
            f"| {d_sp} | {d_scp} | {d_ss} |"
        )

    md_lines += [
        "",
        "## Recall Comparison",
        "",
        "| Distance | Dataset | Dim | K | C++ SIMD | C++ Scalar | Python |",
        "|----------|---------|-----|---|----------|------------|--------|",
    ]

    for row in csv_rows:
        rs = f"{row['recall_simd']:.4f}" if row["recall_simd"] != "" else "N/A"
        rsc = f"{row['recall_scalar']:.4f}" if row["recall_scalar"] != "" else "N/A"
        rp = f"{row['recall_py']:.4f}" if row["recall_py"] != "" else "N/A"
        md_lines.append(
            f"| {row['distance']} | {row['dataset']} | {row['dim']} | {row['k']} "
            f"| {rs} | {rsc} | {rp} |"
        )

    md_lines.append("")

    readme_path.write_text("\n".join(md_lines))

    print(f"\nResults saved to {results_dir}")
    print(f"  - {csv_path.name}")
    print(f"  - {readme_path.name}")
    print(f"  - cpp_results.csv (copy)")
    print(f"  - python_results.csv (copy)")


if __name__ == "__main__":
    main()
