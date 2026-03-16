import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl
import catppuccin

mpl.style.use(catppuccin.PALETTE.macchiato.identifier)

BASE=Path(__file__).parent

DATA=BASE/"results"

OUT=BASE/"plots"
OUT.mkdir(exist_ok=True)

cpp=pd.read_csv(DATA/"cpp_results.csv")
py=pd.read_csv(DATA/"python_results.csv")

df=pd.concat([cpp,py])


def plot_metric(metric,name):

    fig,axs=plt.subplots(1,3,figsize=(18,5))

    for impl in df.impl.unique():

        sub=df[df.impl==impl]

        axs[0].plot(sub["N"],sub[metric],marker="o",label=impl)
        axs[1].plot(sub["DIM"],sub[metric],marker="o",label=impl)
        axs[2].plot(sub["K"],sub[metric],marker="o",label=impl)

    axs[0].set_title(f"{name} vs N")
    axs[1].set_title(f"{name} vs DIM")
    axs[2].set_title(f"{name} vs K")

    for ax in axs:
        ax.legend()
        ax.grid(True)

    fig.tight_layout()

    plt.savefig(OUT/f"{metric}.png",dpi=300)

    print("saved",OUT/f"{metric}.png")


def main():

    plot_metric("build_us","Build Time")
    plot_metric("query_us","Query Time")

    plt.show()


if __name__=="__main__":
    main()