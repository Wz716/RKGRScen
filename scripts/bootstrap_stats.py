"""Percentile bootstrap for BRR_all and REM (and optionally ER/BRR_exec/MS).

Protocol declared in the manuscript:
    - 10,000 scenario-level resamples with replacement
    - statistical seed 42
    - primary CIs for BRR_all and REM (RQ1/RQ2)
    - RQ3 additionally reports ER, BRR_exec, REM, MS

The intervals quantify scenario-level sampling uncertainty conditional on the
recorded outputs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("rows", [])
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _int(row: Dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(float(value)) if value not in ("", None) else 0


def _bootstrap_ci(values: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    import random

    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(BOOTSTRAP_RESAMPLES * alpha / 2)]
    hi = means[int(BOOTSTRAP_RESAMPLES * (1 - alpha / 2)) - 1]
    return {"lower": lo, "upper": hi}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--metrics", default="brr_all,rem", help="Comma-separated: brr_all,rem,er,brr_exec,ms")
    args = parser.parse_args()

    rows = _load(args.input)
    n = len(rows)
    exec_flags = [_int(r, "exec") for r in rows]
    trig_flags = [_int(r, "trig") for r in rows]
    rem_flags = [_int(r, "rem") for r in rows]
    executed = sum(exec_flags)
    exec_and_trig = [e & t for e, t in zip(exec_flags, trig_flags)]

    metric_values: Dict[str, List[float]] = {}
    if "brr_all" in args.metrics:
        metric_values["brr_all"] = [float(e & t) for e, t in zip(exec_flags, trig_flags)]
    if "rem" in args.metrics:
        metric_values["rem"] = [float(r) for r in rem_flags]
    if "er" in args.metrics:
        metric_values["er"] = [float(e) for e in exec_flags]
    if "brr_exec" in args.metrics:
        metric_values["brr_exec"] = [
            float(e & t) / executed if executed else 0.0 for e, t in zip(exec_flags, trig_flags)
        ]

    print(f"Bootstrap: n={n}, resamples={BOOTSTRAP_RESAMPLES}, seed={BOOTSTRAP_SEED}")
    for name, values in metric_values.items():
        ci = _bootstrap_ci(values)
        point = sum(values) / len(values)
        print(f"{name}: point={point * 100:.2f}%  95% CI=[{ci['lower'] * 100:.1f}, {ci['upper'] * 100:.1f}]")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
