# RKGRScen: Road-Network Knowledge Graph Retrieval-Based Scenario Generation for Autonomous Driving Testing

## Overview

RKGRScen is a framework for grounding logical autonomous-driving test scenarios onto concrete road locations and generating executable simulation scenarios. It formalizes road matching as semantic location retrieval over a **road-network knowledge graph** and combines **weighted Leiden community detection**, **LLM-assisted semantic annotation**, **global-to-local retrieval**, **deterministic graph-constraint matching**, and **constraint-based parameter solving**.

The repository supports the experiments reported in the paper:

- **RQ1:** comparison of RKGRScen with Template Mapping and ARISE on a frozen 2,649-scenario benchmark;
- **RQ2:** ablation of community retrieval, semantic expansion, and constraint solving on a balanced 420-scenario subset;
- **RQ3:** cross-stack simulation transfer to Apollo 8.0 in CARLA--Apollo co-simulation.

The repository is intended to expose both the implementation and the scenario-level evidence required to regenerate the reported metrics and statistical analyses.

---

## Important Reproducibility Note

The ARISE baseline used in the paper is **not a simplified reimplementation**.

RKGRScen evaluates ARISE using the **original Scenic pipeline released by the ARISE authors without reimplementation**. The reported configuration uses **top-2 snippet retrieval per section** and the **original test-and-repair procedure**, with at most **10 repairs** and **5 attempts**.

Any older README text describing the baseline as an "ARISE-derived simplified reimplementation" or stating that it "does NOT include Test-and-Repair" is obsolete and inconsistent with the experiments reported in the manuscript.

---

## Experimental Environment

The main experiments reported in the manuscript use:

- **Operating system:** Ubuntu 20.04.6
- **CPU:** AMD Ryzen 9 3950X
- **Memory:** 62 GB RAM
- **GPU:** NVIDIA RTX 3090
- **CARLA:** 0.9.13
- **Maps:** Town01--Town05
- **Apollo:** 8.0 for RQ3 cross-stack co-simulation
- **LLM:** DeepSeek-V3, version 2024-12-25
- **LLM decoding:** temperature 0.2, top-p 0.9, requested seed 42 where supported by the hosted API
- **Sentence embedding model:** all-MiniLM-L6-v2
- **Embedding dimension:** 384
- **Bootstrap resamples:** 10,000
- **Bootstrap seed:** 42

Hosted-API decoding and simulator/co-simulation execution are not assumed to be bit-reproducible across independent complete reruns.

---

## Repository Structure

A complete release should contain the implementation, configuration, prompts/schemas, frozen benchmark, audit records, scenario-level outcomes, and recomputation scripts.

```text
RKGRScen_experiments/
├── RKGRScen/
│   ├── config.py
│   ├── models.py
│   ├── pipeline.py
│   ├── llm_client.py
│   ├── config/
│   │   ├── detector_thresholds.json
│   │   └── violation_map.json
│   ├── prompts/
│   │   ├── index_prompt.*
│   │   ├── extract_prompt.*
│   │   └── expand_prompt.*
│   ├── schemas/
│   │   ├── index_schema.json
│   │   ├── extract_schema.json
│   │   └── expand_schema.json
│   ├── execution/
│   │   ├── carla_runner.py
│   │   └── violation_detector.py
│   ├── query/
│   │   ├── scene_expander.py
│   │   ├── semantic_matcher.py
│   │   ├── retriever.py
│   │   ├── retrieval_adapter.py
│   │   ├── constraint_solver.py
│   │   ├── constraint_validator.py
│   │   └── scenario_match_evaluator.py
│   ├── indexing/
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── community_tagger.py
│   ├── experiments/
│   │   ├── batch_utils.py
│   │   ├── prepare_rq1_rq2_shared.py
│   │   ├── run_rq1_carla_full_execution.py
│   │   ├── run_rq1_baseline_execution.py
│   │   ├── run_rq2_ablation.py
│   │   ├── run_rq3_apollo_pilot.py
│   │   ├── watch_rq2_carla.py
│   │   └── watch_rq1_sequence.py
│   └── tests/
│       ├── test_pipeline.py
│       ├── test_rq1_rq2_shared.py
│       └── test_rq2_ablation.py
├── arise/
│   ├── main.py
│   ├── evaluate_scenarios.py
│   └── ARISE/
│       ├── retrieve.py
│       ├── logger_config.py
│       └── db/
│           ├── pickle_db.py
│           └── unpickle_db.py
├── data/
│   ├── benchmark/
│   │   └── frozen_benchmark_2649.*
│   ├── audit/
│   │   ├── extraction_audit_350.*
│   │   └── rem_validation_350.*
│   └── results/
│       ├── rq1_per_scenario.*
│       ├── rq2_per_scenario.*
│       ├── rq3_per_scenario.*
│       ├── topk_sensitivity.*
│       └── llm_sensitivity.*
├── scripts/
│   ├── compute_metrics.py
│   ├── bootstrap_stats.py
│   ├── audit_metrics.py
│   └── reproduce_tables.py
├── carla_control/
│   └── mkz_keyboard_control_web_2.py
├── DATA_DICTIONARY.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── README.md
└── LICENSE
```

If your local checkout predates the final reproducibility release, ensure that every file named above that is referenced by this README is actually present before publishing the repository.

---

## Installation

### Python environment

Python 3.8 or later is recommended.

```bash
cd RKGRScen_experiments
pip install -r requirements.txt
```

If `requirements.txt` is not yet included in your local checkout, the minimum dependencies used by the current code include packages such as:

```bash
pip install numpy opencv-python flask jsonschema matplotlib
```

Install the CARLA Python API matching the simulator version:

```bash
pip install carla==0.9.13
```

Additional graph, embedding, LLM-client, and statistical dependencies should be listed in `requirements.txt` in the final release.

---

## LLM API Configuration

The code uses environment variables for hosted-LLM access. API keys must not be committed to the repository.

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

Relevant variables include:

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | Hosted LLM API key |
| `DEEPSEEK_BASE_URL` | API base URL |
| `DEEPSEEK_MODEL` | Model identifier |
| `DEEPSEEK_TIMEOUT_S` | Request timeout |
| `DEEPSEEK_MAX_RETRIES` | Maximum request retries |

The paper reports DeepSeek-V3 (version 2024-12-25), temperature 0.2, top-p 0.9, and a requested random seed of 42 where supported by the hosted API.

---

## Core Pipeline

### 1. Road-Network Graph Construction

`RKGRScen/indexing/graph_builder.py`

RKGRScen samples CARLA waypoints at 20 m intervals along drivable lanes and groups them by `(road_id, section_id, lane_id)` into lane-segment nodes. Graph edges follow successor/predecessor relations and junction connectivity.

The graph stores road, lane, junction, traffic-rule, and lane-change attributes used by downstream retrieval and validation.

### 2. Weighted Leiden Community Detection

`RKGRScen/indexing/community_detector.py`

The directed graph is converted to an undirected weighted graph before Leiden community detection.

The paper reports the following fixed edge-affinity coefficients:

| Parameter | Value |
|---|---:|
| `alpha_0` | 1.0 |
| `alpha_road` | 2.5 |
| `alpha_junction` | 1.5 |
| `alpha_section` | 1.0 |
| `alpha_lane` | 1.0 |
| `alpha_count` | 0.5 |
| `alpha_type` | 0.5 |

These values should be exposed in the released configuration rather than hard-coded only in the manuscript.

### 3. Community Annotation

`RKGRScen/indexing/community_tagger.py`

`LLM_index` generates:

- a natural-language road-network summary for each community;
- applicable violation tags selected from the predefined seven violation categories.

Outputs are schema validated before being added to the semantic community index.

### 4. Structured Scenario Extraction

`LLM_extract` converts NHTSA crash narratives into structured logical scenarios.

After initial filtering, **2,052** candidate reports entered structured extraction. **47** reports failed schema validation after all allowed retries and were excluded, leaving **2,005** successfully extracted NHTSA-derived logical scenarios.

The extraction format records the target violation type, actors, coarse road requirements, and environmental conditions.

### 5. Semantic Expansion

`RKGRScen/query/scene_expander.py`

`LLM_expand` converts coarse road requirements into graph-queryable node-attribute and topology constraints and, when applicable, conflict-point and timing requirements.

If semantic expansion remains invalid after the allowed retry procedure, the pipeline can use a violation-type-specific fallback query template. The RQ2 fallback analysis is reported separately from the full ablation.

### 6. Global-to-Local Retrieval

`RKGRScen/query/retriever.py`

Global retrieval first filters communities by violation tag and ranks them by embedding similarity.

The default RKGRScen setting retains:

- **top-3 communities** at the global retrieval level;
- **top-3 locally matched nodes/subgraphs** for downstream instantiation.

Local retrieval then deterministically verifies node attributes and graph topology.

### 7. Constraint Solver

`RKGRScen/query/constraint_solver.py`

The solver determines executable actor parameters such as:

- spawn positions;
- legal lanes;
- initial speeds;
- headings;
- conflict-point timing when an interaction window is defined.

RKGRScen applies the deterministic constraint solver **without a repair loop**.

### 8. CARLA Execution and Violation Detection

`RKGRScen/execution/carla_runner.py`

`RKGRScen/execution/violation_detector.py`

Completed traces are evaluated by a unified rule-based detector for seven violation types:

1. inattention to the road ahead;
2. failure to yield;
3. wrong-way driving;
4. failure to maintain safe following distance;
5. illegal lane change;
6. illegal overtaking;
7. speeding.

The released `detector_thresholds.json` should match the thresholds reported in the manuscript.

---

## Baseline Methods

### Template Mapping

Template Mapping assigns each violation type to a fixed map/road template and selects a matching node without RKGRScen's graph-structured retrieval or constraint solver.

The baseline uses the same frozen scenarios, Town01--Town05 candidate pool, unified detector, and CARLA environment as RKGRScen.

### ARISE

The ARISE baseline is evaluated using the **original Scenic pipeline released by its authors without reimplementation**.

The experimental configuration used in the manuscript is:

- top-2 snippet retrieval per section;
- original Scenic grounding pipeline;
- original test-and-repair procedure;
- at most 10 repairs;
- at most 5 attempts.

The `arise/` directory should therefore contain or invoke the author-released ARISE/Scenic implementation used in the experiments. It must not be described as a simplified reimplementation if the reported experiments use the original released implementation.

#### ARISE API keys

ARISE reads its LLM credentials from two local files that are intentionally
**not committed** to the repository (see `.gitignore`):

- `arise/ARISE/openai_key.txt`
- `arise/ARISE/genai_key.txt`

Create these files yourself and put the corresponding key in each one before
running the ARISE baseline. Without them, ARISE exits with a "Please provide
the ... API key" message.

---

## Frozen Benchmark

RQ1 uses a frozen benchmark of **2,649 logical scenarios**:

- **2,005 NHTSA-derived scenarios**
- **644 synthetic fine-grained scenarios**

Per-type counts:

| Violation type | NHTSA | Synthetic | Total |
|---|---:|---:|---:|
| Inattention to the road ahead | 646 | 217 | 863 |
| Failure to yield | 517 | 140 | 657 |
| Speeding | 349 | 90 | 439 |
| Failure to maintain safe following distance | 302 | 100 | 402 |
| Wrong-way driving | 84 | 60 | 144 |
| Illegal lane change | 55 | 22 | 77 |
| Illegal overtaking | 52 | 15 | 67 |
| **Total** | **2,005** | **644** | **2,649** |

Difficulty split:

- **Coarse:** 1,854
- **Fine:** 795
  - NHTSA-derived fine: 151
  - synthetic fine: 644

The frozen benchmark artifact should expose, at minimum:

```text
scenario_id
source
source_report_id
violation_type
difficulty
is_synthetic
parent_scenario_id
```

If a field is not applicable, it may be empty, but identifiers and source labels should remain stable across all result files.

---

## Human Extraction Audit

The extraction audit contains **350 NHTSA-derived scenarios**, with **50 scenarios per violation type**, independently judged by three autonomous-driving-testing experts.

The three audited fields are:

- violation type;
- actor behavior;
- road requirement.

The reported extraction accuracies are computed over the individual expert judgments:

- violation type: **998/1,050 = 95.0%**
- actor behavior: **939/1,050 = 89.4%**
- road requirement: **913/1,050 = 87.0%**

Majority-vote labels are retained separately as adjudicated scenario-level audit outcomes.

The released audit data should preserve the individual expert labels and not only the aggregated percentages.

---

## Independent REM Validation

REM is independently validated on **350 stratified scenarios**, with **50 scenarios per violation type**.

Experts judge each grounded site using only:

- the original functional-scenario description;
- a visualization of the grounded road segment.

They do **not** see the expanded constraint set `R_req*` or the system REM decision.

Reported results:

- system-side REM: **328/350 = 93.7%**
- independent human match: **322/350 = 92.0%**
- agreement: **344/350 = 98.3%**
- Cohen's kappa: **0.87**
- false positives: **6**
- false negatives: **0**
- overall Fleiss' kappa across expert labels: **0.79**

The released validation artifact should preserve the individual expert labels, majority vote, and system REM decision for each scenario.

---

## Evaluation Metrics

For each scenario `s`:

- `exec(s) = 1` if the instantiated scenario completes successfully;
- `trig(s) = 1` if the target violation is detected in the execution trace;
- `match(s) = 1` if the grounded site satisfies the road-environment constraints.

### Executability Rate

```text
ER = executed scenarios / all evaluated scenarios
```

### End-to-End Behavior Reproduction Rate

```text
BRR_all = executed-and-triggered scenarios / all evaluated scenarios
```

### Behavior Reproduction Rate Among Executed Scenarios

```text
BRR_exec = executed-and-triggered scenarios / executed scenarios
```

### Road-Environment Matching Rate

```text
REM = road-matched scenarios / all evaluated scenarios
```

The repository and scripts should use the term **REM**, not the obsolete `RMA` wording from older drafts.

### Semantic Match Score

MS is a weighted sum of eight binary components:

| Component | Weight |
|---|---:|
| road | 0.18 |
| lane | 0.18 |
| role | 0.16 |
| action | 0.12 |
| autonomous control | 0.12 |
| environment | 0.10 |
| no direct manual control | 0.10 |
| context | 0.04 |

MS is averaged over successfully executed scenarios.

The released scenario-level results should retain the eight component values in addition to the final MS.

---

## Statistical Analysis

For RQ1 and RQ2:

- 95% percentile bootstrap confidence intervals are reported for `BRR_all` and `REM`;
- 10,000 scenario-level resamples are used;
- paired method/configuration differences are assessed by paired bootstrap resampling of scenario pairs;
- McNemar's exact test is additionally applied to the illegal-overtaking comparison;
- ER, `BRR_exec`, and MS are reported as point estimates.

For RQ3:

- pooled 95% bootstrap confidence intervals are additionally reported for ER, `BRR_exec`, REM, and MS.

The bootstrap seed is **42**.

These confidence intervals quantify **scenario-level sampling uncertainty conditional on the recorded pipeline outputs**. They do not include independent rerun variability caused by hosted-LLM decoding or simulator/co-simulation execution.

---

## RQ1: Full-Benchmark Comparison

RQ1 compares RKGRScen, Template Mapping, and ARISE on the same 2,649-scenario benchmark.

Main aggregate results:

| Method | ER (%) | BRR_all (%) | BRR_exec (%) | MS | REM (%) |
|---|---:|---:|---:|---:|---:|
| RKGRScen | 93.88 | 72.59 | 77.32 | 0.969 | 93.43 |
| Template Mapping | 77.65 | 55.23 | 71.12 | 0.883 | 87.62 |
| ARISE | 94.90 | 67.35 | 70.96 | 0.856 | 82.14 |

Underlying aggregate counts:

| Method | Executed | Triggered | Road matched |
|---|---:|---:|---:|
| RKGRScen | 2,487 | 1,923 | 2,475 |
| Template Mapping | 2,057 | 1,463 | 2,321 |
| ARISE | 2,514 | 1,784 | 2,176 |

The per-scenario RQ1 file should allow all values above, all coarse/fine results, and all per-violation-type results to be recomputed directly.

---

## Global-K Sensitivity

The manuscript varies the number of retained global communities over:

```text
K = 1, 2, 3, 4, 5, 6
```

with the local top-K fixed to 3.

Reported results:

| Global-K | ER | BRR_all | REM | MS |
|---:|---:|---:|---:|---:|
| 1 | 85.13 | 62.78 | 83.58 | 0.884 |
| 2 | 91.05 | 70.22 | 90.11 | 0.946 |
| 3 | 93.88 | 72.59 | 93.43 | 0.969 |
| 4 | 93.92 | 72.44 | 93.58 | 0.968 |
| 5 | 93.77 | 72.10 | 93.28 | 0.966 |
| 6 | 93.58 | 71.84 | 93.05 | 0.964 |

The main conclusion is a strong improvement from K=1 to K=3 followed by a practical plateau. Because each K setting is evaluated through a complete pipeline run, small K=4--6 differences may include hosted-LLM and simulator variability and should not be interpreted as a clean causal effect of K alone.

---

## RQ2: Ablation Study

RQ2 uses a balanced subset of **420 scenarios**, with **60 scenarios per violation type**.

The formal manuscript ablations are:

- `full`
- `without_community`
- `without_expansion`
- `without_constraint`

If the codebase contains additional diagnostic variants such as `without_semantic_summaries`, they should be clearly labeled as **auxiliary engineering analyses not included in the primary manuscript RQ2 table**.

Reported aggregate results:

| Configuration | ER | BRR_all | BRR_exec | MS | REM |
|---|---:|---:|---:|---:|---:|
| Full RKGRScen | 88.57 | 68.81 | 77.69 | 0.9113 | 89.52 |
| w/o Community | 50.71 | 31.67 | 62.44 | 0.8162 | 69.76 |
| w/o Expansion | 48.81 | 29.52 | 60.49 | 0.7793 | 81.43 |
| w/o Constraint | 49.76 | 23.33 | 46.89 | 0.7298 | 82.86 |

Underlying counts:

| Configuration | Executed | Triggered | Road matched |
|---|---:|---:|---:|
| Full RKGRScen | 372 | 289 | 376 |
| w/o Community | 213 | 133 | 293 |
| w/o Expansion | 205 | 124 | 342 |
| w/o Constraint | 209 | 98 | 348 |

---

## Semantic-Expansion Fallback Analysis

On the 420-scenario RQ2 subset:

- normal expansion: 400 scenarios;
- fallback expansion: 20 scenarios.

Reported values:

| Expansion mode | N | ER | BRR_all | BRR_exec | MS | REM |
|---|---:|---:|---:|---:|---:|---:|
| Normal expansion | 400 | 89.00 | 70.25 | 78.93 | 0.913 | 90.00 |
| Fallback | 20 | 80.00 | 40.00 | 50.00 | 0.870 | 80.00 |
| Overall Full | 420 | 88.57 | 68.81 | 77.69 | 0.9113 | 89.52 |

The fallback subset contains only scenarios that specifically failed normal semantic expansion, so its degradation should not be interpreted as directly equivalent in magnitude to the full `without_expansion` ablation.

---

## LLM Sensitivity

The paper additionally replaces DeepSeek-V3 with Qwen2.5-72B-Instruct on the 420-scenario RQ2 subset while keeping the remaining pipeline fixed.

| Backbone | ER | BRR_all | BRR_exec | MS | REM |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 88.57 | 68.81 | 77.69 | 0.9113 | 89.52 |
| Qwen2.5-72B-Instruct | 89.05 | 67.38 | 75.67 | 0.8992 | 87.86 |

---

## RQ3: Apollo 8.0 Cross-Stack Simulation Transfer

RQ3 evaluates RKGRScen-generated scenarios with Apollo 8.0 in CARLA--Apollo co-simulation.

This experiment evaluates **cross-stack simulation transfer**. It should not be described as physical-road validation or full real-world external validation.

Three balanced batches of 420 scenarios are evaluated.

Pooled accounting:

```text
Total sampled scenarios              1260
Pre-execution aborts                   49
Attempted scenarios                  1211
Post-attempt execution failures       120
Valid executions                     1091
Detected violations                   557
Road-matched scenarios               1101
```

Batch accounting:

| Status | Batch 1 | Batch 2 | Batch 3 |
|---|---:|---:|---:|
| Total sampled | 420 | 420 | 420 |
| Pre-execution aborts | 10 | 18 | 21 |
| Attempted | 410 | 402 | 399 |
| Post-attempt execution failures | 38 | 42 | 40 |
| Valid executions | 372 | 360 | 359 |
| Detected violations | 179 | 192 | 186 |
| Road matched | 363 | 367 | 371 |

Reported pooled metrics:

| Metric | Pooled value |
|---|---:|
| ER | 86.59% |
| ER_att | 90.09% |
| BRR_exec | 51.05% |
| REM | 87.38% |
| MS | 0.9024 |

### Failure accounting

All selected scenarios remain in the experimental accounting.

- In RQ1 and RQ2, an incomplete run is recorded as `exec(s)=0`.
- In RQ3, a failure before scenario logic begins is recorded as a **pre-execution abort**. It remains in the all-scenario ER denominator but is excluded from the `ER_att` denominator.
- A run that enters scenario execution but does not complete is recorded as a **post-attempt execution failure**.

Therefore, older README wording stating that CARLA/co-simulation failures are automatically isolated and "do not count toward formal results" must not be used.

---

## Scenario-Level Result Schema

To make the quantitative tables independently auditable, scenario-level result files should preserve one row per scenario and method/configuration.

Recommended minimum fields:

```text
scenario_id
source
violation_type
difficulty
method_or_configuration
batch_id
attempted
exec
trig
rem
ms_road
ms_env
ms_role
ms_lane
ms_action
ms_auto
ms_noctrl
ms_context
ms_total
failure_stage
```

Not every field is required for every experiment, but the released records should be sufficient to regenerate the integer numerators and all aggregate metrics reported in the paper.

---

## Recommended Reproduction Workflow

### 1. Validate the frozen benchmark

```bash
python scripts/reproduce_tables.py --check-benchmark
```

Expected totals:

```text
NHTSA-derived = 2005
Synthetic     = 644
Total         = 2649
Coarse        = 1854
Fine          = 795
```

### 2. Recompute RQ1 metrics

```bash
python scripts/reproduce_tables.py --rq1
```

### 3. Recompute RQ2 ablation and fallback results

```bash
python scripts/reproduce_tables.py --rq2
```

### 4. Recompute RQ3 pooled accounting

```bash
python scripts/reproduce_tables.py --rq3
```

### 5. Recompute bootstrap and paired-bootstrap statistics

```bash
python scripts/bootstrap_stats.py
```

### 6. Recompute human-audit statistics

```bash
python scripts/audit_metrics.py
```

If the actual script names in the final release differ from the examples above, update this README so that every command is executable exactly as written.

---

## Data Dictionary

`DATA_DICTIONARY.md` should describe every field in:

- the frozen benchmark;
- the extraction audit;
- the independent REM validation;
- RQ1 per-scenario outcomes;
- RQ2 per-scenario outcomes;
- RQ3 per-scenario outcomes.

Binary fields should explicitly define the meaning of `0` and `1`, and missing/not-applicable values should use a documented representation.

---

## Reproducibility Checklist

Before creating the final public release, verify that:

- [ ] the repository is publicly accessible without authentication;
- [ ] the code used for RKGRScen is present;
- [ ] the original author-released ARISE/Scenic pipeline used in the experiments is present or reproducibly invoked;
- [ ] no README text describes ARISE as a simplified reimplementation;
- [ ] the original ARISE test-and-repair procedure is represented correctly;
- [ ] all three task-specific LLM prompts are present;
- [ ] all three JSON schemas are present;
- [ ] the frozen 2,649-scenario benchmark is present;
- [ ] the 350-scenario extraction-audit records are present;
- [ ] the 350-scenario independent REM-validation records are present;
- [ ] RQ1, RQ2, and RQ3 scenario-level result files are present;
- [ ] the eight MS component fields are preserved;
- [ ] metric recomputation scripts run without manual editing;
- [ ] bootstrap/paired-bootstrap scripts use 10,000 resamples and seed 42;
- [ ] README metric names use REM rather than obsolete RMA wording;
- [ ] RQ3 failure accounting matches the manuscript;
- [ ] the Global-K results match the manuscript;
- [ ] the Data Availability Statement in the manuscript matches the files actually visible in the repository;
- [ ] no API keys, tokens, passwords, or private credentials are committed.

---

## Scope and Limitations

The reported Apollo experiment is a CARLA--Apollo co-simulation study. Physical-road testing and validation on a full real-world ADS deployment are outside the scope of the current release.

The statistical bootstrap intervals quantify scenario-level sampling uncertainty conditional on the recorded outputs. They should not be interpreted as confidence intervals over independent hosted-LLM and simulator reruns.

---

## Citation

If you use RKGRScen in academic work, please cite the corresponding paper:

> Zhe Wang, Yaqing Shi, Tongtong Bai, Song Huang, Changyou Zheng, Kui Yao, Kunyuan Li, and Yao He.  
> **RKGRScen: Road-Network Knowledge Graph Retrieval-Based Scenario Generation for Autonomous Driving Testing.**

A `CITATION.cff` file is recommended for the final public release.

---

## License

Use of the code and data should follow the license included in the final public release.

If the repository does not yet contain a `LICENSE` file, add an explicit license before public release rather than relying only on the phrase "for research purposes only."
