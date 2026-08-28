import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) > 1:
    OUT = Path(sys.argv[1])
else:
    OUT = Path("benchmarks/results")

OUT.mkdir(parents=True, exist_ok=True)

CPP = OUT / "cpp_results.csv"
PY_BINDINGS = OUT / "python_bindings_results.csv"
PY_HNSWLIB = OUT / "python_hnswlib_results.csv"

run_dir = OUT.name

def load():
    cpp = pd.read_csv(CPP)
    py_bindings = pd.read_csv(PY_BINDINGS)
    py_hnswlib = pd.read_csv(PY_HNSWLIB)

    return {
        "cpp": cpp,
        "python_bindings": py_bindings,
        "python_hnswlib": py_hnswlib,
    }


def compare(datasets):
    cpp = datasets["cpp"]
    py_bindings = datasets["python_bindings"]
    py_hnswlib = datasets["python_hnswlib"]

    # We compare against the SIMD C++ implementation since that is
    # the optimized native implementation.
    cpp_simd = cpp[cpp["impl"] == "cpp_simd"].copy()

    bindings = py_bindings.copy()
    hnswlib = py_hnswlib.copy()

    bindings["impl"] = "python_bindings"
    hnswlib["impl"] = "python_hnswlib"

    merged = cpp_simd.merge(
        bindings,
        on=["N", "DIM", "K"],
        suffixes=("_cpp", "_bindings"),
    )

    merged = merged.merge(
        hnswlib,
        on=["N", "DIM", "K"],
    )

    merged = merged.rename(
        columns={
            "build_s": "build_s_hnswlib",
            "query_us": "query_us_hnswlib",
            "speedup": "speedup_hnswlib",
            "recall": "recall_hnswlib",
        }
    )

    merged["build_s_bindings_delta_%"] = (
        (merged["build_s_bindings"] - merged["build_s_cpp"])
        / merged["build_s_cpp"]
        * 100
    )

    merged["query_bindings_delta_%"] = (
        (merged["query_us_bindings"] - merged["query_us_cpp"])
        / merged["query_us_cpp"]
        * 100
    )

    merged["build_s_hnswlib_delta_%"] = (
        (merged["build_s_hnswlib"] - merged["build_s_cpp"])
        / merged["build_s_cpp"]
        * 100
    )

    merged["query_hnswlib_delta_%"] = (
        (merged["query_us_hnswlib"] - merged["query_us_cpp"])
        / merged["query_us_cpp"]
        * 100
    )

    merged["bindings_vs_hnswlib_query_delta_%"] = (
        (merged["query_us_bindings"] - merged["query_us_hnswlib"])
        / merged["query_us_hnswlib"]
        * 100
    )

    merged["bindings_vs_hnswlib_build_delta_%"] = (
        (merged["build_s_bindings"] - merged["build_s_hnswlib"])
        / merged["build_s_hnswlib"]
        * 100
    )

    output_columns = [
        "N",
        "DIM",
        "K",
        "build_s_cpp",
        "build_s_bindings",
        "build_s_hnswlib",
        "query_us_cpp",
        "query_us_bindings",
        "query_us_hnswlib",
        "recall_cpp",
        "recall_bindings",
        "recall_hnswlib",
        "build_s_bindings_delta_%",
        "query_bindings_delta_%",
        "build_s_hnswlib_delta_%",
        "query_hnswlib_delta_%",
        "bindings_vs_hnswlib_build_delta_%",
        "bindings_vs_hnswlib_query_delta_%",
    ]

    # Only include columns that actually exist. This makes the comparison
    # tolerant of benchmarks that don't emit recall.
    output_columns = [
        column for column in output_columns if column in merged.columns
    ]

    comparison = merged[output_columns]

    comparison.to_csv(
        OUT / "comparison.csv",
        index=False,
    )

    print("\nC++ SIMD vs Python Bindings vs hnswlib\n")
    print(comparison.to_string(index=False))

    return comparison


def scaling_tables(datasets):
    for name, df in datasets.items():
        for fixed in ["DIM", "N", "K"]:
            group_columns = [
                column
                for column in ["N", "DIM", "K"]
                if column != fixed
            ]

            table = (
                df.groupby(group_columns)[["build_s", "query_us"]]
                .mean()
                .reset_index()
            )

            path = OUT / f"{name}_fix_{fixed}.csv"

            table.to_csv(
                path,
                index=False,
            )

            print(f"\n{name} fixed {fixed}\n")
            print(table.to_string(index=False))


def get_system_info():
    info = {}

    if platform.system() == "Darwin":
        os_version = platform.mac_ver()[0] or platform.release()
    else:
        os_version = platform.release()

    info["Operating System"] = f"{platform.system()} {os_version}"
    info["Architecture"] = platform.machine()
    info["CPU"] = platform.processor() or platform.machine()

    logical = os.cpu_count() or "unknown"
    info["CPU Cores"] = str(logical)

    try:
        mem_bytes = (
            os.sysconf("SC_PAGE_SIZE")
            * os.sysconf("SC_PHYS_PAGES")
        )

        info["Memory"] = (
            f"{mem_bytes / (1024 ** 3):.1f} GB"
        )

    except (ValueError, AttributeError, OSError):
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )

            mem_bytes = int(result.stdout.strip())

            info["Memory"] = (
                f"{mem_bytes / (1024 ** 3):.1f} GB"
            )

        except Exception:
            info["Memory"] = "unknown"

    info["Python Version"] = platform.python_version()

    return info


def df_to_markdown(df, float_fmt=".4f"):
    if df.empty:
        return "_No results._"

    cols = df.columns.tolist()

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(cols)
    ) + " |"

    rows = []

    for _, row in df.iterrows():
        cells = []

        for column in cols:
            value = row[column]

            if isinstance(value, float):
                cells.append(f"{value:{float_fmt}}")
            else:
                cells.append(str(value))

        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(
        [header, separator] + rows
    )


def generate_report(datasets, comparison):
    lines = []

    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Run: `{run_dir}`")
    lines.append("")

    lines.append("## System Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| --- | --- |")

    for key, value in get_system_info().items():
        lines.append(f"| {key} | {value} |")

    lines.append("")

    cpp = datasets["cpp"]
    bindings = datasets["python_bindings"]
    hnswlib = datasets["python_hnswlib"]

    lines.append("## C++ Benchmark Results")
    lines.append("")
    lines.append(df_to_markdown(cpp))
    lines.append("")

    lines.append("## Python Bindings Benchmark Results")
    lines.append("")
    lines.append(df_to_markdown(bindings))
    lines.append("")

    lines.append("## Python hnswlib Benchmark Results")
    lines.append("")
    lines.append(df_to_markdown(hnswlib))
    lines.append("")

    lines.append("## Comparison")
    lines.append("")
    lines.append(
        "The comparison uses the optimized `cpp_simd` "
        "implementation as the native baseline."
    )
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

    report_path.write_text(
        "\n".join(lines)
    )

    print(f"\nReport saved to {report_path}")


def main():
    datasets = load()

    comparison = compare(datasets)

    scaling_tables(datasets)

    generate_report(
        datasets,
        comparison,
    )

    print("\nAnalysis saved to", OUT)


if __name__ == "__main__":
    main()