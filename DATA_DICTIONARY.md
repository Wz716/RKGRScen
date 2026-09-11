# Data Dictionary

This document describes the fields of each released artifact. Binary fields
use `1` for true and `0` for false. Missing / not-applicable values use an
empty string (`""`) or `null` as documented per field.

> **Status:** the schema below is authoritative. The corresponding data files
> under `data/` are released separately; see `ARTIFACT_AUDIT.md` for which
> files are present.

## 1. Frozen benchmark (`data/benchmark/frozen_benchmark_2649.*`)

| Field | Type | Description |
|---|---|---|
| `scenario_id` | string | Stable unique scenario identifier |
| `source` | string | `nhtsa` or `synthetic` |
| `source_report_id` | string | NHTSA report id; empty for synthetic |
| `violation_type` | string | One of the seven violation types (see below) |
| `difficulty` | string | `coarse` or `fine` |
| `is_synthetic` | int | `1` if synthetic, `0` otherwise |
| `parent_scenario_id` | string | Source scenario for synthetic extension; empty for NHTSA |

Violation types (manuscript names): `inattention`, `failure_to_yield`,
`speeding`, `following_distance`, `wrong_way`, `illegal_lane_change`,
`illegal_overtaking`.

## 2. Extraction audit (`data/audit/extraction_audit_350.*`)

One record per (scenario, field, annotator) judgment.

| Field | Type | Description |
|---|---|---|
| `scenario_id` | string | Benchmark scenario id |
| `field` | string | `violation_type`, `actor_behavior`, `road_requirement` |
| `annotator` | string | Annotator id (1..3) |
| `judgment` | string | `correct`, `partial`, `incorrect` (see below) |

- `violation_type`: single-label, `correct` / `incorrect`.
- `actor_behavior` and `road_requirement`: three-level `correct` / `partial` / `incorrect`.

## 3. Independent REM validation (`data/audit/rem_validation_350.*`)

| Field | Type | Description |
|---|---|---|
| `scenario_id` | string | Benchmark scenario id |
| `system_rem` | int | RKGRScen REM decision (`1` match, `0` non-match) |
| `expert_1` / `expert_2` / `expert_3` | string | `match`, `partial`, `incorrect` |
| `human_majority` | int | Majority-vote match (`1` / `0`; partial counts as non-match) |

## 4. RQ1 / RQ2 per-scenario outcomes

| Field | Type | Description |
|---|---|---|
| `scenario_id` | string | Benchmark scenario id |
| `source` | string | `nhtsa` / `synthetic` |
| `violation_type` | string | Violation type |
| `difficulty` | string | `coarse` / `fine` |
| `method_or_configuration` | string | `rkqrscen`, `template_mapping`, `arise` (RQ1); `full`, `without_community`, `without_expansion`, `without_constraint` (RQ2) |
| `attempted` | int | `1` if execution was attempted |
| `exec` | int | `1` if executed to completion |
| `trig` | int | `1` if target violation reproduced |
| `rem` | int | `1` if road-environment matched |
| `ms_road` ... `ms_context` | float | Eight MS component values |
| `ms_total` | float | Weighted MS |
| `failure_stage` | string | Failure stage label (empty if executed) |

## 5. RQ3 per-scenario outcomes

Adds:

| Field | Type | Description |
|---|---|---|
| `batch_id` | int | Batch 1..3 |
| `pre_exec_abort` | int | `1` if aborted before scenario logic began |
| `post_attempt_failure` | int | `1` if entered execution but did not complete |

## Metric definitions

- `ER = sum(exec) / N`
- `BRR_all = sum(exec & trig) / N`
- `BRR_exec = sum(exec & trig) / sum(exec)`
- `REM = sum(rem) / N`
- `MS = sum(weight_i * component_i)` over executed scenarios
