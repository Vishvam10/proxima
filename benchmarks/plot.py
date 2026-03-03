import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_comparison(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(
    rows: list[dict],
    simd_col: str,
    scalar_col: str,
    py_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    labels = []
    simd_vals = []
    scalar_vals = []
    py_vals = []

    for row in rows:
        label = f"{row['distance']}\n{row['dataset']}x{row['dim']}"
        labels.append(label)

        simd_vals.append(float(row[simd_col]) if row[simd_col] else 0)
        scalar_vals.append(float(row[scalar_col]) if row[scalar_col] else 0)
        py_vals.append(float(row[py_col]) if row[py_col] else 0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 1.2), 7))

    bars_simd = ax.bar(x - width, simd_vals, width, label="C++ SIMD", color="#2196F3")
    bars_scalar = ax.bar(x, scalar_vals, width, label="C++ Scalar", color="#FF9800")
    bars_py = ax.bar(x + width, py_vals, width, label="Python (hnswlib)", color="#4CAF50")

    ax.set_xlabel("Scenario (distance / dataset x dim)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path.name}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: plot.py <results_dir_name>")

    results_name = sys.argv[1]
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results" / results_name
    csv_path = results_dir / "comparison.csv"

    if not csv_path.exists():
        raise SystemExit(f"Missing comparison CSV: {csv_path}")

    rows = load_comparison(csv_path)

    print("\nGenerating plots...")

    plot_metric(
        rows,
        simd_col="build_simd_s",
        scalar_col="build_scalar_s",
        py_col="build_py_s",
        ylabel="Build Time (seconds)",
        title="Build Time: C++ SIMD vs C++ Scalar vs Python",
        output_path=results_dir / "build_time.png",
    )

    plot_metric(
        rows,
        simd_col="query_simd_us",
        scalar_col="query_scalar_us",
        py_col="query_py_us",
        ylabel="Query Time (microseconds)",
        title="Query Time: C++ SIMD vs C++ Scalar vs Python",
        output_path=results_dir / "query_time.png",
    )

    print(f"\nPlots saved to {results_dir}\n")


if __name__ == "__main__":
    main()
