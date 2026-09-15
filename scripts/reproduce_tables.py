"""Recompute the paper's main tables from scenario-level records.

Subcommands map paper tables to their source data files (see REPRODUCIBILITY.md).
This script delegates to compute_metrics.py and never fabricates records.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "results"


def _run(input_path: Path, group_by: str | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "compute_metrics.py"), "--input", str(input_path)]
    if group_by:
        cmd += ["--group-by", group_by]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq1", action="store_true")
    parser.add_argument("--rq2", action="store_true")
    parser.add_argument("--rq3", action="store_true")
    parser.add_argument("--topk", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--check-benchmark", action="store_true")
    args = parser.parse_args()

    targets = []
    if args.rq1:
        targets.append(DATA / "rq1_per_scenario.csv")
    if args.rq2:
        targets.append(DATA / "rq2_per_scenario.csv")
    if args.rq3:
        targets.append(DATA / "rq3_per_scenario.csv")
    if args.topk:
        targets.append(DATA / "topk_sensitivity.csv")
    if args.llm:
        targets.append(DATA / "llm_sensitivity.csv")

    if args.check_benchmark:
        print("Benchmark totals expected: NHTSA=2005 Synthetic=644 Total=2649 Coarse=1854 Fine=795")
        print("Validation requires data/benchmark/frozen_benchmark_2649.csv (see DATA_DICTIONARY.md).")
        return

    if not targets:
        parser.error("choose at least one of --rq1/--rq2/--rq3/--topk/--llm/--check-benchmark")

    for path in targets:
        print(f"### {path.relative_to(ROOT)}")
        _run(path, group_by="method_or_configuration")


if __name__ == "__main__":
    main()
