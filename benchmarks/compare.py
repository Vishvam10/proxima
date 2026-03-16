import os
import platform
import subprocess
import sys

import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = BASE / "results" / run_dir
OUT.mkdir(exist_ok=True, parents=True)

CPP = OUT / "cpp_results.csv"
PY = OUT / "python_results.csv"


def load():
    cpp = pd.read_csv(CPP)
    py = pd.read_csv(PY)
    df = pd.concat([cpp, py], ignore_index=True)
    return df


def compare(df):
    pivot = df.pivot_table(
        index=["N", "DIM", "K"], columns="impl", values=["build_s", "query_us"]
    )

    pivot.columns = ["_".join(c) for c in pivot.columns]

    pivot["build_delta_%"] = (
        (pivot["build_s_cpp_simd"] - pivot["build_s_python"])
        / pivot["build_s_python"]
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

    return pivot


def scaling_tables(df):
    for impl in df.impl.unique():
        sub = df[df.impl == impl]

        for fixed in ["DIM", "N", "K"]:
            grp = sub.groupby([c for c in ["N", "DIM", "K"] if c != fixed])
            t = grp[["build_s", "query_us"]].mean().reset_index()
            path = OUT / f"{impl}_fix_{fixed}.csv"
            t.to_csv(path, index=False)

            print(f"\n{impl} fixed {fixed}\n")
            print(t)


def get_system_info():
    info = {}
    info["Operating System"] = f"{platform.system()} {platform.mac_ver()[0] or platform.release()}"
    info["Architecture"] = platform.machine()
    info["CPU"] = platform.processor() or platform.machine()

    logical = os.cpu_count() or "unknown"
    info["CPU Cores"] = str(logical)

    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["Memory"] = f"{mem_bytes / (1024 ** 3):.1f} GB"
    except (ValueError, AttributeError):
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True
            )
            mem_bytes = int(result.stdout.strip())
            info["Memory"] = f"{mem_bytes / (1024 ** 3):.1f} GB"
        except Exception:
            info["Memory"] = "unknown"

    info["Python Version"] = platform.python_version()
    return info


def df_to_markdown(df, float_fmt=".4f"):
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:{float_fmt}}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def generate_report(df, comparison):
    lines = []

    lines.append("## Benchmark Report")
    lines.append("")
    lines.append(f"Run: `{run_dir}`")
    lines.append("")

    lines.append("## System Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| --- | --- |")
    for key, val in get_system_info().items():
        lines.append(f"| {key} | {val} |")
    lines.append("")

    lines.append("## C++ Benchmark Results")
    lines.append("")
    cpp_df = df[df.impl.str.startswith("cpp")]
    lines.append(df_to_markdown(cpp_df))
    lines.append("")

    lines.append("## Python (hnswlib) Benchmark Results")
    lines.append("")
    py_df = df[df.impl == "python"]
    lines.append(df_to_markdown(py_df))
    lines.append("")

    lines.append("## Comparison (C++ SIMD vs Python)")
    lines.append("")
    lines.append(df_to_markdown(comparison))
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("### Build Time Comparison")
    lines.append("")
    lines.append("![Build Time](plots/build_s.png)")
    lines.append("")
    lines.append("### Query Time Comparison")
    lines.append("")
    lines.append("![Query Time](plots/query_us.png)")
    lines.append("")

    report_path = OUT / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to {report_path}")


def main():
    df = load()
    comparison = compare(df)
    scaling_tables(df)
    generate_report(df, comparison)
    print("\nAnalysis saved to", OUT)


if __name__ == "__main__":
    main()
