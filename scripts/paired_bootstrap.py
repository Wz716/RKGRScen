"""Paired bootstrap for between-method / between-configuration differences.

Resamples the *same* scenarios jointly for both arms, so the comparison is
paired. Reports the mean difference and a 95% percentile CI.

Protocol: 10,000 resamples, seed 42.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

RESAMPLES = 10_000
SEED = 42


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


def _by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {str(r[key]): r for r in rows if r.get(key) not in ("", None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--group-by", default="method_or_configuration")
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--metric", default="brr_all", choices=("brr_all", "rem", "er", "brr_exec"))
    args = parser.parse_args()

    rows = _load(args.input)
    a = _by_key(rows, args.group_by).get(args.arm_a, {})
    b = _by_key(rows, args.group_by).get(args.arm_b, {})
    common = sorted(set(a) & set(b))
    if not common:
        print(f"No shared scenario_ids between {args.arm_a} and {args.arm_b}", file=sys.stderr)
        sys.exit(2)

    def metric_value(row: Dict[str, Any]) -> float:
        if args.metric == "brr_all":
            return float(_int(row, "exec") and _int(row, "trig"))
        if args.metric == "rem":
            return float(_int(row, "rem"))
        if args.metric == "er":
            return float(_int(row, "exec"))
        raise NotImplementedError("brr_exec needs a pooled denominator; not implemented here")

    import random

    rng = random.Random(SEED)
    n = len(common)
    diffs = []
    for _ in range(RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(metric_value(a[common[i]]) for i in idx) / n
        db = sum(metric_value(b[common[i]]) for i in idx) / n
        diffs.append((da - db) * 100.0)
    diffs.sort()
    lo = diffs[int(RESAMPLES * 0.025)]
    hi = diffs[int(RESAMPLES * 0.975) - 1]
    point = sum(metric_value(a[s]) for s in common) / n - sum(metric_value(b[s]) for s in common) / n

    print(f"Paired {args.metric}: {args.arm_a} - {args.arm_b} over {n} shared scenarios")
    print(f"  mean diff = {point * 100:.2f} pp  95% CI = [{lo:.1f}, {hi:.1f}] pp")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
