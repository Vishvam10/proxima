import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import catppuccin
import matplotlib.pyplot as plt
import pandas as pd

# Use Catppuccin Macchiato styling.
matplotlib.style.use(catppuccin.PALETTE.macchiato.identifier)


# --------------------------------------------------
# Input / Output
# --------------------------------------------------

if len(sys.argv) > 1:
    OUT = Path(sys.argv[1])
else:
    OUT = Path("benchmarks/results")

OUT.mkdir(parents=True, exist_ok=True)

CPP = OUT / "cpp_results.csv"
PY_BINDINGS = OUT / "python_bindings_results.csv"
PY_HNSWLIB = OUT / "python_hnswlib_results.csv"

PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

cpp = pd.read_csv(CPP)
python_bindings = pd.read_csv(PY_BINDINGS)
python_hnswlib = pd.read_csv(PY_HNSWLIB)

# Normalize implementation names so all datasets can
# be plotted together.
python_bindings["impl"] = "python_bindings"
python_hnswlib["impl"] = "python_hnswlib"

df = pd.concat(
    [
        cpp,
        python_bindings,
        python_hnswlib,
    ],
    ignore_index=True,
)


# --------------------------------------------------
# Display names
# --------------------------------------------------

DISPLAY_NAMES = {
    "cpp_scalar": "C++ Scalar",
    "cpp_simd": "C++ SIMD",
    "python_bindings": "Python Bindings",
    "python_bindings_scalar": "Python Bindings Scalar",
    "python_bindings_simd": "Python Bindings SIMD",
    "python_hnswlib": "Python hnswlib",
    "python": "Python hnswlib",
}


def display_name(impl):
    return DISPLAY_NAMES.get(impl, impl)


# --------------------------------------------------
# Plot
# --------------------------------------------------


def plot_metric(
    metric: str,
    y_label: str,
    title: str,
):
    # Every (N, K) combination gets its own subplot.
    groups = (
        df[["N", "K"]]
        .drop_duplicates()
        .sort_values(["N", "K"])
        .itertuples(index=False, name=None)
    )

    groups = list(groups)

    if not groups:
        print("No benchmark data found.")
        return

    # Dynamic grid.
    cols = min(3, len(groups))
    rows = (len(groups) + cols - 1) // cols

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(7 * cols, 5.5 * rows),
        squeeze=False,
    )

    axs = axs.flatten()

    # One legend entry per implementation.
    legend_handles = {}
    legend_labels = {}

    for i, (N, K) in enumerate(groups):
        ax = axs[i]

        subset = df[(df["N"] == N) & (df["K"] == K)]

        implementations = sorted(subset["impl"].unique())

        for impl in implementations:
            sub = subset[subset["impl"] == impl].sort_values("DIM")

            if sub.empty or metric not in sub.columns:
                continue

            (line,) = ax.plot(
                sub["DIM"],
                sub[metric],
                marker="o",
                linewidth=2,
                markersize=5,
                label=display_name(impl),
            )

            label = display_name(impl)

            if label not in legend_handles:
                legend_handles[label] = line
                legend_labels[label] = label

        ax.set_title(
            f"N = {N:,}, K = {K}",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        ax.set_xlabel(
            "Dimension",
            labelpad=8,
        )

        ax.set_ylabel(
            y_label,
            labelpad=8,
        )

        ax.set_xticks(sorted(subset["DIM"].unique()))

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.tick_params(
            axis="both",
            labelsize=10,
        )

    # Hide unused axes.
    for i in range(len(groups), len(axs)):
        axs[i].set_visible(False)

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    # Single shared legend for the entire figure.
    fig.legend(
        legend_handles.values(),
        legend_labels.values(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.87],
        h_pad=3,
        w_pad=2,
    )

    output = PLOTS / f"{metric}.png"

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot: {output}")


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    plot_metric(
        "build_s",
        "Build Time (s)",
        "Build Time Comparison",
    )

    plot_metric(
        "query_us",
        "Query Time (µs)",
        "Query Time Comparison",
    )


if __name__ == "__main__":
    main()
