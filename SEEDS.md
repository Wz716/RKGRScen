# Randomness and Seeds

This document records every random seed used in the RKGRScen codebase and its
purpose. Do **not** claim that "all experiments use one fixed seed"; different
stages use different seeds for different purposes.

## Seeds actually present in the code

| Seed | Where used | Purpose |
|---|---:|---|
| `20260714` | `RKGRScen/experiments/prepare_rq1_rq2_shared.py` (`FIXED_SEED`) | Frozen shared-manifest / benchmark-sampling seed for the RQ1/RQ2 single-seed protocol (`protocol_version = "rq1-rq2-single-seed-v3"`). All manifest rows carry this seed. |
| `20260714` | `RKGRScen/experiments/run_rq2_ablation.py` (`--seed` default) | RQ2 balanced-subset sampling / case-level seed. `stable_case_seed()` derives a per-scenario seed from this global seed, the scenario id, and the source path. |
| `42` | `RKGRScen/indexing/community_detector.py` (`seed=42`) | Community-detection partitioning seed. The detector uses Leiden (`igraph.community_leiden` / `leidenalg`), falling back to networkx Louvain (`louvain_communities(..., seed=42)`) when Leiden is unavailable. |

## Seeds declared in the manuscript / README but not hard-coded in RKGRScen

| Seed | Declared purpose | Status |
|---|---:|---|
| `42` | Hosted-LLM decoding `random seed` (where supported by the API) | Implemented in `RKGRScen/llm_client.py` (`DeepSeekClient.SEED = 42`, passed in the request payload). |
| `42` | Statistical bootstrap resampling seed | Declared in the manuscript and README; the recomputation scripts that implement this are not yet present in this checkout (see `scripts/`). |

## Notes

- `20260714` is the benchmark / RQ1/RQ2 sampling seed, **not** the LLM or
  bootstrap seed.
- The bootstrap protocol declared in the manuscript is: 10,000 scenario-level
  percentile resamples, statistical seed 42.
