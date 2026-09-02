from pathlib import Path

import pytest

from RKGRScen.experiments import run_rq2_ablation as rq2
from RKGRScen.query.retriever import GraphRetriever

def organized(violation_type, actors=None, road=None):
    return {
        "violation_type": violation_type,
        "dsl": {
            "road_network": road or {"type": "Straight", "lanes": 2},
            "actors": actors or [],
        },
    }

@pytest.mark.parametrize(
    ("source", "expected_type", "expected_action"),
    [
        (
            organized(
                "未保持安全距离",
                [
                    {"initial_position": "S2N", "actions": "Move Forward", "speed_limit": 36},
                    {"initial_position": "S2N", "actions": "Brake", "speed_limit": 18},
                ],
            ),
            "未保持安全距离",
            "Move Forward",
        ),
        (
            organized(
                "违规变道",
                [
                    {"initial_position": "W2E", "actions": "Change Lane Right", "speed_limit": 45},
                    {"initial_position": "W2E", "actions": "Move Forward", "speed_limit": 40},
                ],
            ),
            "违规变道",
            "Change Lane Right",
        ),
        (
            organized(
                "未注意前方路况",
                [{"initial_position": "E2W", "actions": "Turn Left", "speed_limit": 25}],
                {"type": "Intersection", "lanes": 4},
            ),
            "未注意前方路况",
            "Turn Left",
        ),
        (
            organized(
                "超速行驶",
                [{"initial_position": "N2S", "actions": "Move Forward", "speed_limit": 66}],
                {"type": "Straight", "lanes": 1, "speed_limit_kmh": 40},
            ),
            "超速",
            "Move Forward",
        ),
    ],
)
def test_build_skeleton_spec_maps_dsl(source, expected_type, expected_action):
    spec, diagnostics = rq2.build_skeleton_spec(source)

    assert spec["violation_type"] == expected_type
    assert spec["actors"][0]["action"] == expected_action
    assert spec["actors"][0]["path"] == source["dsl"]["actors"][0]["initial_position"]
    assert "defaulted_fields" in diagnostics
    if expected_type == "违规变道":
        assert spec["road_requirement"]["same_direction_multi_lane"] is True
        assert spec["road_requirement"]["lane_change_direction"] == "right"
    if expected_type == "未注意前方路况":
        assert "needs_long_straight" not in spec["road_requirement"]
        assert spec["actors"][1]["action"] == "Static Obstacle"
    if expected_type == "超速":
        assert spec["speed_requirement"]["target_speed_kmh"] == 66
        assert spec["road_requirement"]["speed_limit_kmh"] == 40

def test_stable_case_seed_is_repeatable_and_case_specific():
    first = rq2.stable_case_seed(17, "scenario_00001", "/tmp/场景.json")
    assert first == rq2.stable_case_seed(17, "scenario_00001", "/tmp/场景.json")
    assert first != rq2.stable_case_seed(17, "scenario_00002", "/tmp/场景.json")
    assert first != rq2.stable_case_seed(18, "scenario_00001", "/tmp/场景.json")

def test_scenario_id_is_shared_across_variants():
    assert rq2.scenario_id_for("full", 7) == "scenario_00007"
    assert rq2.scenario_id_for("without_expansion", 7) == "scenario_00007"
    assert "without_semantic_summaries" in rq2.VARIANTS

def test_retrieval_without_semantic_summaries_has_audit_fallback(monkeypatch, tmp_path):
    class DummyRetriever:
        def __init__(self, base):
            self.base = base

        def retrieve(self, spec, top_k=3):
            return {"local_top_k": [{"score": 1.0}], "mode": "community"}

    monkeypatch.setattr(rq2, "GraphRetriever", DummyRetriever)

    result, diagnostics = rq2.retrieval_for({"violation_type": "超速"}, "/tmp/source.json", "without_semantic_summaries", tmp_path, tmp_path)

    assert result["mode"] == "community"
    assert diagnostics["mode"] == "community_without_semantic_summaries_fallback"
    assert diagnostics["llm_summary_used"] is False

def test_full_graph_retrieval_keeps_only_top_town(monkeypatch):
    retriever = GraphRetriever(Path("/nonexistent"))
    town_nodes = {
        "Town01": {"a": node("a", 1)},
        "Town02": {"b": node("b", 2)},
        "Town03": {},
        "Town04": {},
        "Town05": {},
    }
    monkeypatch.setattr(retriever, "_graph_nodes_by_id", lambda town: town_nodes[town])
    monkeypatch.setattr(retriever, "_lane_index_by_road", lambda town: {item["road_id"]: [item["lane_id"]] for item in town_nodes[town].values()})
    monkeypatch.setattr(retriever, "_score_node", lambda attrs, requirement, lane_index: (True, {"matched": [], "failed": []}, 10.0 if attrs["id"] == "b" else 5.0))

    result = retriever.retrieve_without_community({"violation_type": "超速", "road_requirement": {"type": "Straight"}}, top_k=3)

    assert result["global_top_k"] == []
    assert result["mode"] == "full_graph"
    assert {item["map_name"] for item in result["local_top_k"]} == {"Town02"}

def node(node_id, road_id):
    return {
        "id": node_id,
        "road_id": road_id,
        "lane_id": -1,
        "section_id": 0,
        "lane_count": 1,
        "curvature": 0.0,
        "speed_limit": 40.0,
        "is_junction": False,
        "lane_change": "None",
        "start": {"x": 0.0, "y": 0.0},
        "end": {"x": 40.0, "y": 0.0},
        "heading": 0.0,
    }

def test_summarize_variant_rates():
    rows = [
        {"status": "ok", "detected": True, "constraint_satisfied": True, "total_generation_time_s": 3.0, "stage_timing_s": {"retrieval": 1.0}},
        {"status": "ok", "detected": False, "constraint_satisfied": False, "total_generation_time_s": 5.0, "stage_timing_s": {"retrieval": 3.0}},
        {"status": "failed", "failure_type": "method_failure", "total_generation_time_s": 1.0, "stage_timing_s": {"retrieval": 2.0}},
        {"status": "failed", "failure_type": "simulator_unhealthy_before"},
    ]

    summary = rq2.summarize_variant(rows, "full")

    assert summary["total"] == 3
    assert summary["executed"] == 2
    assert summary["failed"] == 1
    assert summary["execution_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["behavior_reproduction_rate"] == 0.5
    assert summary["constraint_satisfaction_rate"] == 0.5
    assert summary["avg_total_generation_time_s"] == 3.0
    assert summary["avg_retrieval_time_s"] == 2.0

def test_full_expansion_rejects_silent_llm_fallback(monkeypatch, tmp_path):
    class DisabledExpander:
        def __init__(self, use_llm):
            assert use_llm is True
            self.use_llm = False

    monkeypatch.setattr(rq2, "SceneExpander", DisabledExpander)

    with pytest.raises(RuntimeError, match="effective use_llm=False"):
        rq2.expansion_for("/tmp/source.json", organized("超速行驶"), "full", tmp_path)
