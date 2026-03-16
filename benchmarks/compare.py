import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

CPP = BASE / "results/cpp_results.csv"
PY = BASE / "results/python_results.csv"

OUT = BASE / "results/analysis"
OUT.mkdir(exist_ok=True)


def load():

    cpp = pd.read_csv(CPP)
    py = pd.read_csv(PY)

    df = pd.concat([cpp, py], ignore_index=True)

    return df


def compare(df):

    pivot = df.pivot_table(
        index=["N", "DIM", "K"], columns="impl", values=["build_us", "query_us"]
    )

    pivot.columns = ["_".join(c) for c in pivot.columns]

    pivot["build_delta_%"] = (
        (pivot["build_us_cpp_simd"] - pivot["build_us_python"])
        / pivot["build_us_python"]
        * 100
    )

    pivot["query_delta_%"] = (
        (pivot["query_us_cpp_simd"] - pivot["query_us_python"])
        / pivot["query_us_python"]
        * 100
    )

    pivot.reset_index(inplace=True)

    pivot.to_csv(OUT / "comparison.csv", index=False)

    print("\nC++ vs Python\n")
    print(pivot)


def scaling_tables(df):

    for impl in df.impl.unique():
        sub = df[df.impl == impl]

        for fixed in ["DIM", "N", "K"]:
            grp = sub.groupby([c for c in ["N", "DIM", "K"] if c != fixed])

            t = grp[["build_us", "query_us"]].mean().reset_index()

            path = OUT / f"{impl}_fix_{fixed}.csv"

            t.to_csv(path, index=False)

            print(f"\n{impl} fixed {fixed}\n")
            print(t)


def main():

    df = load()

    compare(df)

    scaling_tables(df)

    print("\nAnalysis saved to", OUT)


if __name__ == "__main__":
    main()
