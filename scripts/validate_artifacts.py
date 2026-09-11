"""Validate that all manuscript-referenced artifacts are present.

Produces a PASS/MISSING report per artifact. Exit code 1 if any required
artifact is missing. This is a read-only check; it never creates data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    "RKGRScen/config/violation_map.json",
    "RKGRScen/config/detector_thresholds.json",
    "RKGRScen/prompts/index_prompt.txt",
    "RKGRScen/prompts/expand_prompt.txt",
    "RKGRScen/prompts/extract_prompt.txt",
    "RKGRScen/prompts/semantic_match_prompt.txt",
    "RKGRScen/schemas/index_schema.json",
    "RKGRScen/schemas/expand_schema.json",
    "RKGRScen/schemas/extract_schema.json",
    "RKGRScen/extraction/scenario_extractor.py",
    "data/benchmark/frozen_benchmark_2649.csv",
    "data/audit/extraction_audit_350.csv",
    "data/audit/rem_validation_350.csv",
    "data/results/rq1_per_scenario.csv",
    "data/results/rq2_per_scenario.csv",
    "data/results/rq3_per_scenario.csv",
    "data/results/topk_sensitivity.csv",
    "data/results/llm_sensitivity.csv",
    "arise/ARISE/db/database_v1_scenic3.pkl",
    "arise/ARISE/prompts/generation/extraction.txt",
    "arise/eval_prompt.txt",
    "arise/LICENSE.txt",
    "requirements.txt",
    "README.md",
    "DATA_DICTIONARY.md",
    "REPRODUCIBILITY.md",
    "SEEDS.md",
    "CITATION.cff",
    "LICENSE",
]

OPTIONAL = [
    "data/benchmark/frozen_benchmark_2649.json",
    "data/audit/extraction_audit_350.json",
    "data/audit/rem_validation_350.json",
    "data/results/rq1_per_scenario.json",
    "data/results/rq2_per_scenario.json",
    "data/results/rq3_per_scenario.json",
]


def main() -> None:
    missing: list[str] = []
    for rel in REQUIRED:
        path = ROOT / rel
        status = "PASS" if path.exists() else "MISSING"
        print(f"[{status}] {rel}")
        if not path.exists():
            missing.append(rel)

    print()
    print(f"Required artifacts: {len(REQUIRED) - len(missing)}/{len(REQUIRED)} present")

    if missing:
        print("\nMissing (provide real files; do not fabricate):")
        for rel in missing:
            print(f"  - {rel}")
        sys.exit(1)


if __name__ == "__main__":
    main()
