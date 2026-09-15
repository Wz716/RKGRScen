"""Recompute RKGRScen metrics from scenario-level outcome records.

Reads per-scenario records (CSV or JSON) and recomputes:
    ER, BRR_all, BRR_exec, REM, MS
together with the underlying integer counts. This script never synthesizes
records; it exits with a clear error if the input file is missing.

Expected columns (see DATA_DICTIONARY.md):
    scenario_id, source, violation_type, difficulty,
    method_or_configuration, attempted, exec, trig, rem,
    ms_road, ms_env, ms_role, ms_lane, ms_action, ms_auto, ms_noctrl, ms_context,
    ms_total
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

MS_WEIGHTS = {
    "road": 0.18,
    "lane": 0.18,
    "role": 0.16,
    "action": 0.12,
    "auto": 0.12,
    "env": 0.10,
    "noctrl": 0.10,
    "context": 0.04,
}
MS_COMPONENT_COLS = ["ms_road", "ms_env", "ms_role", "ms_lane", "ms_action", "ms_auto", "ms_noctrl", "ms_context"]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Scenario-level records not found: {path}\n"
            "This file must be provided by the authors; it cannot be generated."
        )
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("rows", [])
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_int(row: Dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    return int(float(value)) if value not in ("", None) else 0


def _as_float(row: Dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    return float(value) if value not in ("", None) else 0.0


def _compute(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    executed = sum(_as_int(r, "exec") for r in rows)
    triggered = sum(_as_int(r, "trig") for r in rows)
    matched = sum(_as_int(r, "rem") for r in rows)
    exec_and_trig = sum(1 for r in rows if _as_int(r, "exec") and _as_int(r, "trig"))

    ms_total = 0.0
    for r in rows:
        if _as_int(r, "exec"):
            if "ms_total" in r and r.get("ms_total") not in ("", None):
                ms_total += _as_float(r, "ms_total")
            else:
                ms_total += sum(MS_WEIGHTS[k] * _as_float(r, c) for k, c in zip(MS_WEIGHTS, MS_COMPONENT_COLS))

    return {
        "n": n,
        "executed": executed,
        "triggered": triggered,
        "road_matched": matched,
        "exec_and_trig": exec_and_trig,
        "er": (executed / n * 100.0) if n else 0.0,
        "brr_all": (exec_and_trig / n * 100.0) if n else 0.0,
        "brr_exec": (exec_and_trig / executed * 100.0) if executed else 0.0,
        "rem": (matched / n * 100.0) if n else 0.0,
        "ms": (ms_total / executed) if executed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Per-scenario CSV/JSON file")
    parser.add_argument("--group-by", default=None, help="Optional column to group by (e.g. method_or_configuration)")
    args = parser.parse_args()

    rows = _load_records(args.input)

    if args.group_by:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(args.group_by, "unknown")), []).append(row)
        for name in sorted(groups):
            result = _compute(groups[name])
            print(f"== {name} ==")
            print(f"  N={result['n']} Exec={result['executed']} Trigger={result['triggered']} "
                  f"RoadMatched={result['road_matched']} ExecAndTrig={result['exec_and_trig']}")
            print(f"  ER={result['er']:.2f}% BRR_all={result['brr_all']:.2f}% "
                  f"BRR_exec={result['brr_exec']:.2f}% REM={result['rem']:.2f}% MS={result['ms']:.4f}")
    else:
        result = _compute(rows)
        print(f"N={result['n']} Exec={result['executed']} Trigger={result['triggered']} "
              f"RoadMatched={result['road_matched']} ExecAndTrig={result['exec_and_trig']}")
        print(f"ER={result['er']:.2f}% BRR_all={result['brr_all']:.2f}% "
              f"BRR_exec={result['brr_exec']:.2f}% REM={result['rem']:.2f}% MS={result['ms']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
