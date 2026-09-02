import json
from pathlib import Path

from RKGRScen.models import ExecutionTrace
from RKGRScen.pipeline import RKGRScenPipeline

def test_yield_pipeline_end_to_end(monkeypatch) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    map_path = base_dir / "data" / "maps" / "town03_sample_lanes.json"
    scenario_path = base_dir / "data" / "logical_scenarios" / "yield_case_001.json"

    with map_path.open("r", encoding="utf-8") as handle:
        lane_records = json.load(handle)
    with scenario_path.open("r", encoding="utf-8") as handle:
        logical_scenario = json.load(handle)

    pipeline = RKGRScenPipeline()

    def fake_run(scenario) -> ExecutionTrace:
        return ExecutionTrace(
            scenario_id=scenario.scenario_id,
            ticks=[
                {
                    "timestamp_s": 0.0,
                    "ego": {"speed_mps": 0.0, "distance_to_conflict_m": 10.0},
                    "npcs": [],
                }
            ],
        )

    monkeypatch.setattr(pipeline.runner, "run", fake_run)
    index_bundle = pipeline.build_index("Town03", lane_records)
    result = pipeline.generate(logical_scenario, index_bundle["graph"], index_bundle["communities"])

    assert result["scene_spec"]["violation_type"] == "未按规定让行"
    assert result["scenario_config"]["map_name"] == "Town03"
    assert result["violation_result"]["violation_type"] == "未按规定让行"
    assert "detected" in result["violation_result"]
