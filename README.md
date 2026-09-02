# RKGRScen: Road-Network Knowledge Graph Retrieval-Based Scenario Generation for Autonomous Driving Testing

## Overview

RKGRScen is a framework for generating executable traffic scenarios from logical violation descriptions. It formalizes road matching as a semantic location retrieval problem over a **road-network knowledge graph**, combining **community detection**, **LLM-assisted semantic annotation**, **global-to-local retrieval**, and **constraint solving** to convert natural language traffic violation scenarios into concrete, executable CARLA simulator scenarios.

This repository contains the cleaned experimental code used in our research paper, including the core RKGRScen pipeline, baseline implementations (Template Mapping and ARISE-derived), CARLA execution infrastructure, and all experiment scripts for RQ1 (comparison), RQ2 (ablation), and RQ3 (Apollo+CARLA integration).

---

## Repository Structure

```
RKGRScen_experiments/
├── RKGRScen/                          # Core RKGRScen framework
│   ├── config.py                    # Configuration loading (LLM settings, paths)
│   ├── models.py                    # Data models (CommunityRecord, ScenarioConfiguration, etc.)
│   ├── pipeline.py                  # Main RKGRScenPipeline orchestration
│   ├── llm_client.py                # LLM API client (DeepSeek)
│   ├── config/
│   │   ├── detector_thresholds.json  # Violation detection thresholds
│   │   └── violation_map.json        # Violation type mapping
│   ├── execution/
│   │   ├── carla_runner.py          # CARLA scenario execution runner
│   │   └── violation_detector.py    # Rule-based violation detection (7 violation types)
│   ├── query/
│   │   ├── scene_expander.py        # LLM-based structured scenario expansion
│   │   ├── semantic_matcher.py      # Semantic matching with LLM assistance
│   │   ├── retriever.py             # Graph retriever (global + local search)
│   │   ├── retrieval_adapter.py     # Retrieval result adaptation
│   │   ├── constraint_solver.py     # Constraint solving for scenario configuration
│   │   ├── constraint_validator.py  # Constraint validation
│   │   └── scenario_match_evaluator.py  # Scene match quality evaluation
│   ├── indexing/
│   │   ├── graph_builder.py         # Road-network graph construction from OpenDRIVE maps
│   │   ├── community_detector.py    # Community detection via Leiden algorithm
│   │   └── community_tagger.py     # Community annotation with LLM-generated summaries
│   ├── experiments/
│   │   ├── batch_utils.py           # Shared utilities for batch experiment execution
│   │   ├── prepare_rq1_rq2_shared.py  # Prepare shared manifest for RQ1/RQ2
│   │   ├── run_rq1_carla_full_execution.py  # RQ1 Full RKGRScen execution
│   │   ├── run_rq1_baseline_execution.py    # RQ1 baseline (Template Mapping) execution
│   │   ├── run_rq2_ablation.py      # RQ2 ablation study execution
│   │   ├── run_rq3_apollo_pilot.py  # RQ3 Apollo+CARLA integration pilot
│   │   ├── watch_rq2_carla.py       # RQ2 CARLA watchdog with auto-restart
│   │   └── watch_rq1_sequence.py    # RQ1 sequence watchdog
│   └── tests/
│       ├── test_pipeline.py         # Pipeline unit tests
│       ├── test_rq1_rq2_shared.py   # RQ1/RQ2 shared manifest tests
│       └── test_rq2_ablation.py     # RQ2 ablation variant tests
├── arise/                           # ARISE baseline implementation
│   ├── main.py                      # ARISE main entry point
│   ├── evaluate_scenarios.py       # ARISE scenario evaluation
│   └── ARISE/
│       ├── retrieve.py              # ARISE retrieval logic
│       ├── logger_config.py         # Logging configuration
│       └── db/
│           ├── pickle_db.py        # Pickle database utilities
│           └── unpickle_db.py      # Database loading utilities
├── carla_control/
│   └── mkz_keyboard_control_web_2.py  # Web-based CARLA manual controller (Flask)
└── README.md
```

---

## Prerequisites

- **Python**: 3.8 or higher
- **CARLA**: 0.9.13 (must be running for scenario execution)
- **LLM API**: For LLM-based components (scene expansion, community tagging, structured extraction)
- **System dependencies**:
  - `numpy`, `opencv-python`, `flask` (for CARLA control)
  - `jsonschema` (for LLM response validation)
  - `matplotlib` (for result visualization)

## Installation

```bash
# Clone or copy the repository
cd RKGRScen_experiments

# Install Python dependencies
pip install numpy opencv-python flask jsonschema matplotlib

# Install CARLA Python API (from CARLA installation)
pip install carla==0.9.13

# Set up environment variables
export DEEPSEEK_API_KEY="your-api-key-here"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

## API Key Configuration

The code uses environment variables for API keys. No secrets are hardcoded. Set the following variables before running:

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | Your LLM API key |
| `DEEPSEEK_BASE_URL` | API base URL (default: `https://api.deepseek.com`) |
| `DEEPSEEK_MODEL` | Model name (default: `deepseek-chat`) |
| `DEEPSEEK_TIMEOUT_S` | Request timeout in seconds (default: `60`) |
| `DEEPSEEK_MAX_RETRIES` | Max retry attempts (default: `2`) |

---

## Core Components

### 1. RKGRScenPipeline (`RKGRScen/pipeline.py`)

The main orchestration pipeline that chains all components together:

```python
from RKGRScen.pipeline import RKGRScenPipeline

pipeline = RKGRScenPipeline()

# Step 1: Build road-network knowledge graph and community index
index = pipeline.build_index("Town03", lane_records)

# Step 2: Generate concrete scenario from logical description
result = pipeline.generate(logical_scenario, index["graph"], index["communities"])
```

### 2. Road-Network Indexing (`RKGRScen/indexing/`)

Constructs a searchable semantic index from OpenDRIVE maps:
- **Graph Builder**: Extracts lane-segment nodes and connectivity edges from CARLA waypoint sampling
- **Community Detector**: Partitions the road-network graph into topologically coherent communities using the Leiden algorithm
- **Community Tagger**: Uses LLM to generate natural-language summaries and violation tags for each community

### 3. Scene Expander (`RKGRScen/query/scene_expander.py`)

LLM-based structured expansion that enriches logical scenarios with:
- Violation type analysis and refinement
- Actor role classification (violator, priority, witness)
- Conflict point specification
- Road requirement extraction (type, lanes, traffic lights, topology)

### 4. Graph Retriever (`RKGRScen/query/retriever.py`)

Two-stage global-to-local retrieval:
- **Global search**: Searches across pre-built semantic community index using embedding similarity to find relevant map regions
- **Local search**: Performs subgraph matching within selected communities to locate exact road segments satisfying topological constraints

### 5. Constraint Solver (`RKGRScen/query/constraint_solver.py`)

Solves the constraint satisfaction problem to produce a complete concrete scenario configuration:
- Ego vehicle spawn position and initial speed
- NPC vehicle placement and behavior
- Conflict point calculation and timing
- Environment settings (weather, time of day)

### 6. CARLA Runner (`RKGRScen/execution/carla_runner.py`)

Executes the generated scenario in CARLA:
- Loads the target map
- Spawns vehicles at computed positions
- Controls ego via autopilot and NPCs via scripted behaviors
- Records full execution trace (vehicle states, sensor data)

### 7. Violation Detector (`RKGRScen/execution/violation_detector.py`)

Rule-based detection for 7 violation types:
- **Inattention to the road ahead**: Forward collision checking
- **Failure to yield**: TTC-based right-of-way detection
- **Wrong-way driving**: Road direction compliance checking
- **Failure to maintain safe following distance**: RSS-based safe distance checking
- **Illegal lane change**: Trajectory-based lane position analysis
- **Illegal overtaking**: Longitudinal/lateral conflict detection
- **Speeding**: Speed limit enforcement

---

## Baseline Methods

### Template Mapping (`RKGRScen/experiments/run_rq1_baseline_execution.py`)

A controlled baseline that uses coarse road type + violation type lookup tables to select map locations and spawn points. Reuses the same CARLA runner and violation detector for fair comparison.

### ARISE-derived (`arise/`)

Simplified re-implementation of the ARISE approach using:
- Logical-to-NL template conversion
- Flat text retrieval from scenario fragment libraries
- Coarse geometric filtering
- Does NOT include Test-and-Repair or adversarial search

---

## Experiments

### RQ1: Scenario Generation Quality Comparison

**Objective**: Compare RKGRScen against baseline methods on scenario generation quality.

**Execution**:
```bash
# Prepare shared manifest
python RKGRScen/experiments/prepare_rq1_rq2_shared.py

# Run Full RKGRScen
python RKGRScen/experiments/run_rq1_carla_full_execution.py

# Run Template Mapping baseline
python RKGRScen/experiments/run_rq1_baseline_execution.py

# Monitor execution with auto-restart
python RKGRScen/experiments/watch_rq1_sequence.py
```

**Metrics**:
- **ER (Executability Rate)**: Proportion of logical scenarios successfully converted to executable scenarios
- **BRR_all (End-to-End Behavior Reproduction Rate)**: Proportion where target violation was reproduced across all attempts
- **BRR_exec (Behavior Reproduction Rate among Executed)**: Violation reproduction rate among successfully executed scenarios
- **MS (Match Score)**: Semantic/topological alignment between logical and concrete scenarios
- **RMA (Road Matching Accuracy)**: Proportion where retrieved road satisfies type + topology constraints

### RQ2: Ablation Study

**Objective**: Analyze the contribution of each RKGRScen component.

**Variants**:
- `full`: Complete RKGRScen
- `without_community`: Disable community-level graph retrieval (single-hop search)
- `without_constraint`: Disable constraint solver (default parameter assignment)
- `without_expansion`: Disable LLM-based scene expansion (use raw logical scenario)
- `without_semantic_summaries`: Disable LLM-generated community summaries (use structural features only)

**Execution**:
```bash
python RKGRScen/experiments/run_rq2_ablation.py --variants full without_community without_constraint without_expansion without_semantic_summaries

# Monitor with auto-restart
python RKGRScen/experiments/watch_rq2_carla.py
```

### RQ3: Apollo+CARLA Integration

**Objective**: Validate RKGRScen-generated scenarios with Apollo autonomous driving system.

**Execution**:
```bash
python RKGRScen/experiments/run_rq3_apollo_pilot.py
```

---

## CARLA Control

The `carla_control/mkz_keyboard_control_web_2.py` provides a Flask-based web interface for manual CARLA vehicle control:

```bash
python carla_control/mkz_keyboard_control_web_2.py
# Open http://localhost:5000 in browser
```

Features:
- Keyboard-based throttle/steer/brake control
- Real-time camera view
- GNSS/IMU data display
- Spectator camera tracking

---

## Notes

- All experiments use a fixed random seed (`20260714`) for reproducibility
- CARLA environment failures (crashes, unhealthy simulator) are automatically isolated and do not count toward formal results
- The watchdog scripts (`watch_rq1_sequence.py`, `watch_rq2_carla.py`) provide automatic CARLA restart and breakpoint recovery
- Experiments are conducted on CARLA Town01–Town05 map pool
- Scenarios are classified by topological complexity; results are reported for different subsets

---

## License

This code is provided for research purposes only.
