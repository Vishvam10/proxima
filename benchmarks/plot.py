import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import catppuccin


matplotlib.style.use(
    catppuccin.PALETTE.macchiato.identifier
)


BASE = Path(__file__).parent

run_dir = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "."
)

DATA = BASE / "results" / run_dir
OUT = DATA / "plots"

OUT.mkdir(
    exist_ok=True,
    parents=True,
)


CPP = pd.read_csv(
    DATA / "cpp_results.csv"
)

PY_BINDINGS = pd.read_csv(
    DATA / "python_bindings_results.csv"
)

PY_HNSWLIB = pd.read_csv(
    DATA / "python_hnswlib_results.csv"
)


cpp = CPP.copy()
python_bindings = PY_BINDINGS.copy()
python_hnswlib = PY_HNSWLIB.copy()

df = pd.concat(
    [
        cpp,
        python_bindings,
        python_hnswlib,
    ],
    ignore_index=True,
)


DISPLAY_NAMES = {
    "cpp_scalar": "C++ Scalar",
    "cpp_simd": "C++ SIMD",
    "python_bindings": "Python Bindings",
    "python_hnswlib": "Python hnswlib",
    "python": "Python hnswlib",
}


def display_name(impl):
    return DISPLAY_NAMES.get(
        impl,
        impl,
    )


def plot_metric(
    metric: str,
    y_label: str,
    global_title: str,
):
    Ns = sorted(df.N.unique())

    # The benchmark currently has up to:
    #
    # 1K
    # 5K
    # 10K
    # 50K
    # 100K
    # 500K
    #
    # So 2 x 3 works nicely.
    rows = 2
    cols = 3

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(20, 12),
    )

    axs = axs.flatten()

    lines = []
    labels = []

    for i, N in enumerate(Ns):
        ax = axs[i]

        dfN = df[df.N == N]

        for impl in dfN.impl.unique():
            sub = dfN[dfN.impl == impl]

            for K in sorted(sub.K.unique()):
                dfK = (
                    sub[sub.K == K]
                    .sort_values("DIM")
                )

                if metric not in dfK:
                    continue

                label = (
                    f"{display_name(impl)} "
                    f"(K={K})"
                )

                line, = ax.plot(
                    dfK["DIM"],
                    dfK[metric],
                    marker="o",
                    label=label,
                )

                if label not in labels:
                    lines.append(line)
                    labels.append(label)

        ax.set_title(
            f"N = {N}",
            pad=15,
        )

        ax.set_xlabel(
            "DIM",
            labelpad=10,
        )

        ax.set_ylabel(
            y_label,
            labelpad=10,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    # Hide unused axes if the number of N values
    # is less than rows * cols.
    for i in range(len(Ns), len(axs)):
        axs[i].set_visible(False)

    fig.suptitle(
        global_title,
        fontsize=24,
        y=0.94,
        fontweight="bold",
    )

    fig.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.83]
    )

    output = OUT / f"{metric}.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot: {output}")


def main():
    plot_metric(
        "build_s",
        "Build Time (s)",
        "Build Time Comparison",
    )

    plot_metric(
        "query_us",
        "Query Time (us)",
        "Query Time Comparison",
    )


if __name__ == "__main__":
    main()