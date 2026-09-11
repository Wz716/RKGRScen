# Reproducibility

This document explains how to reproduce the experiments reported in the
manuscript and how the released artifacts map to the reported numbers.

## Environment

| Item | Value |
|---|---|
| OS | Ubuntu 20.04.6 |
| CPU | AMD Ryzen 9 3950X |
| RAM | 62 GB |
| GPU | NVIDIA RTX 3090 |
| CARLA | 0.9.13 |
| Maps | Town01--Town05 |
| Apollo | 8.0 (RQ3 only) |
| LLM | DeepSeek-V3, version 2024-12-25 |
| LLM decoding | temperature 0.2, top-p 0.9, requested seed 42 where supported |
| Sentence embedding | all-MiniLM-L6-v2, dimension 384 |

## Randomness and seeds

See [SEEDS.md](SEEDS.md). The benchmark / RQ1/RQ2 sampling seed is `20260714`;
the bootstrap seed is `42`; hosted-LLM decoding requests seed `42` where the
API supports it.

## Artifact map

| Paper result | Source data | Reproduction script |
|---|---|---|
| RQ1 Table (full benchmark) | `data/results/rq1_per_scenario.*` | `scripts/reproduce_tables.py --rq1` |
| RQ1 coarse/fine split | same as above | `scripts/reproduce_tables.py --rq1` |
| RQ1 Global-K sensitivity | `data/results/topk_sensitivity.*` | `scripts/reproduce_tables.py --topk` |
| RQ2 ablation | `data/results/rq2_per_scenario.*` | `scripts/reproduce_tables.py --rq2` |
| RQ2 fallback analysis | `data/results/rq2_per_scenario.*` | `scripts/reproduce_tables.py --rq2` |
| RQ3 pooled accounting | `data/results/rq3_per_scenario.*` | `scripts/reproduce_tables.py --rq3` |
| LLM sensitivity | `data/results/llm_sensitivity.*` | `scripts/reproduce_tables.py --llm` |
| Extraction audit | `data/audit/extraction_audit_350.*` | `scripts/audit_metrics.py` |
| Independent REM validation | `data/audit/rem_validation_350.*` | `scripts/audit_metrics.py` |
| Bootstrap / paired-bootstrap CIs | scenario-level records above | `scripts/bootstrap_stats.py` |

> **Note:** the `data/` and `scripts/` paths listed above are the *target*
> layout declared by `README.md`. See `ARTIFACT_AUDIT.md` (if present) or the
> repository status below for which artifacts are currently available.

## Failure accounting

- RQ1/RQ2: all selected scenarios remain in the accounting; an incomplete run
  is recorded as `exec(s)=0`.
- RQ3: a pre-execution abort remains in the all-scenario ER denominator but is
  excluded from the `ER_att` denominator. A run that enters scenario execution
  but does not complete is a post-attempt execution failure.

## Statistical protocol

- 95% percentile bootstrap, 10,000 scenario-level resamples, seed 42.
- RQ1/RQ2 primary CIs: `BRR_all` and `REM`.
- Paired comparisons resample the same scenarios jointly.
- McNemar's exact test is applied to the illegal-overtaking category.
- RQ3 pooled CIs: ER, `BRR_exec`, REM, MS.
- These CIs quantify scenario-level sampling uncertainty conditional on the
  recorded outputs only; they do **not** include hosted-LLM decoding or
  simulator/co-simulation run-to-run stochasticity.
