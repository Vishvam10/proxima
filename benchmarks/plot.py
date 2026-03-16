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
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    lines = []  # for common legend
    labels = []

    for impl in df.impl.unique():
        sub = df[df.impl == impl]

        # --- vs N (fix DIM=128, K=10) ---
        df_N = sub[(sub.DIM == 128) & (sub.K == 10)].sort_values("N")
        line, = axs[0].plot(df_N["N"], df_N[metric], marker="o", label=impl)
        if impl not in labels:
            lines.append(line)
            labels.append(impl)
        axs[0].set_title(f"{y_label} vs N (DIM=128, K=10)")
        axs[0].set_xlabel("N", labelpad=12)
        axs[0].set_ylabel(y_label, labelpad=12)
        axs[0].grid(True)

        # --- vs DIM (fix N=5000, K=10) ---
        df_DIM = sub[(sub.N == 5000) & (sub.K == 10)].sort_values("DIM")
        axs[1].plot(df_DIM["DIM"], df_DIM[metric], marker="o")
        axs[1].set_title(f"{y_label} vs DIM (N=5000, K=10)")
        axs[1].set_xlabel("DIM", labelpad=12)
        axs[1].set_ylabel(y_label, labelpad=12)
        axs[1].grid(True)

        # --- vs K (fix N=5000, DIM=128) ---
        df_K = sub[(sub.N == 5000) & (sub.DIM == 128)].sort_values("K")
        axs[2].plot(df_K["K"], df_K[metric], marker="o")
        axs[2].set_title(f"{y_label} vs K (N=5000, DIM=128)")
        axs[2].set_xlabel("K", labelpad=12)
        axs[2].set_ylabel(y_label, labelpad=12)
        axs[2].grid(True)

    # Horizontal spacing
    fig.subplots_adjust(wspace=0.35, top=0.78)

    # Global title
    fig.suptitle(global_title, fontsize=16)

    # One common legend above subplots
    fig.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),  # centered above plots
        ncol=len(labels),
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