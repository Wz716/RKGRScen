"""Recompute human-audit statistics (extraction audit and REM validation).

Reads the released audit records and recomputes:
    - per-field extraction accuracy over individual expert judgments
    - majority-vote adjudication counts
    - system/human REM agreement, Cohen's kappa

This script never fabricates expert labels; it fails clearly if the input
files are missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Audit file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("rows", [])
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = sum(a) / n
    p2 = sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def extraction_audit(path: Path) -> None:
    rows = _load(path)
    fields = sorted({r["field"] for r in rows})
    print("== Extraction audit (individual judgments) ==")
    for field in fields:
        field_rows = [r for r in rows if r["field"] == field]
        correct = sum(1 for r in field_rows if r.get("judgment") == "correct")
        partial = sum(1 for r in field_rows if r.get("judgment") == "partial")
        total = len(field_rows)
        print(f"{field}: correct={correct}/{total}={correct / total * 100:.1f}%  partial={partial}/{total}={partial / total * 100:.1f}%")


def rem_validation(path: Path) -> None:
    rows = _load(path)
    n = len(rows)
    system = [int(r.get("system_rem", 0)) for r in rows]
    human = [int(r.get("human_majority", 0)) for r in rows]
    agree = sum(1 for s, h in zip(system, human) if s == h)
    fp = sum(1 for s, h in zip(system, human) if s == 1 and h == 0)
    fn = sum(1 for s, h in zip(system, human) if s == 0 and h == 1)
    print("== Independent REM validation ==")
    print(f"n={n} system_match={sum(system)} human_match={sum(human)}")
    print(f"agreement={agree}/{n}={agree / n * 100:.1f}%  Cohen kappa={_cohens_kappa(system, human):.3f}")
    print(f"false_positives={fp} false_negatives={fn}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, help="extraction_audit_350 CSV/JSON")
    parser.add_argument("--rem", type=Path, help="rem_validation_350 CSV/JSON")
    args = parser.parse_args()

    if not args.extraction and not args.rem:
        parser.error("provide --extraction and/or --rem")

    if args.extraction:
        extraction_audit(args.extraction)
    if args.rem:
        rem_validation(args.rem)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
