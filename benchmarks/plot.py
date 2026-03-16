import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import catppuccin

matplotlib.style.use(catppuccin.PALETTE.macchiato.identifier)

BASE = Path(__file__).parent

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
DATA = BASE / "results" / run_dir
OUT = DATA / "plots"
OUT.mkdir(exist_ok=True, parents=True)

cpp = pd.read_csv(DATA / "cpp_results.csv")
py = pd.read_csv(DATA / "python_results.csv")
df = pd.concat([cpp, py], ignore_index=True)

def plot_metric(metric: str, y_label: str, global_title: str):
    Ns = sorted(df.N.unique())

    rows = 2
    cols = 3

    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))
    axs = axs.flatten()

    lines = []
    labels = []

    for i, N in enumerate(Ns):
        ax = axs[i]

        dfN = df[df.N == N]

        for impl in dfN.impl.unique():
            sub = dfN[dfN.impl == impl]

            for K in sorted(sub.K.unique()):
                dfK = sub[sub.K == K].sort_values("DIM")

                label = f"{impl} (K={K})"
                line, = ax.plot(dfK["DIM"], dfK[metric], marker="o", label=label)

                if label not in labels:
                    lines.append(line)
                    labels.append(label)

        ax.set_title(f"N = {N}")
        ax.set_xlabel("DIM", labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        ax.grid(True)

    fig.subplots_adjust(wspace=0.30, hspace=0.35, top=0.85)

    fig.suptitle(global_title, fontsize=16)

    fig.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=4,
        frameon=False
    )

    plt.savefig(OUT / f"{metric}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {OUT / f'{metric}.png'}")

def main():
    plot_metric("build_s", "Build Time (s)", "Build Time Comparison")
    plot_metric("query_us", "Query Time (us)", "Query Time Comparison")


if __name__ == "__main__":
    main()