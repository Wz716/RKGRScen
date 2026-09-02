import csv
import hashlib
import json
from pathlib import Path
from urllib import error

import pytest
from jsonschema.exceptions import ValidationError

from RKGRScen.experiments.prepare_rq1_rq2_shared import (
    DERIVED_FINE_SCENARIOS,
    EXPECTED_SCENARIOS,
    FIXED_SEED,
    build_granularity_labels,
    build_manifest,
    build_map_stats,
    classify_granularity,
    generate,
)
from RKGRScen.indexing.community_tagger import CommunityTagger
from RKGRScen.llm_client import DeepSeekClient
from RKGRScen.query.scene_expander import SceneExpander

BASE = Path(__file__).resolve().parents[2]

def test_manifest_has_shared_ids_and_fixed_seed():
    rows = [
        {"source_path": str(BASE / "organized_scenarios/未保持安全距离/未保持安全距离_100117976.json"), "source_violation_type": "未保持安全距离"},
        {"source_path": str(BASE / "organized_scenarios/逆行/逆行_431466074.json"), "source_violation_type": "逆行"},
    ]
    manifest = build_manifest(rows)

    assert [row["scenario_id"] for row in manifest["rows"]] == ["scenario_00001", "scenario_00002"]
    assert {row["seed"] for row in manifest["rows"]} == {FIXED_SEED}
    assert all("method" not in row and "variant" not in row for row in manifest["rows"])
    assert len(manifest["schema_sha256"]) == len(manifest["protocol_sha256"]) == len(manifest["rows_sha256"]) == 64
    assert manifest["scenario_count"] == 2
    assert manifest["protocol_version"] == "rq1-rq2-single-seed-v3"

def test_granularity_uses_only_explicit_g3_fields(tmp_path):
    coarse = {"dsl": {"road_network": {"type": "Straight", "lanes": 4, "stem_direction": "Not applicable"}}}
    fine = {"dsl": {"road_network": {"type": "T-intersection", "lanes": 2, "stem_direction": "South"}}}
    assert classify_granularity(coarse) == ("coarse", "未触发G3细粒度规则")
    assert classify_granularity(fine) == ("fine", "复合节点约束")

    paths = []
    for index, payload in enumerate((coarse, fine), 1):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    manifest = {"rows": [{"scenario_id": f"scenario_{index:05d}", "source_path": str(path)} for index, path in enumerate(paths, 1)]}
    labels, diagnostics = build_granularity_labels(manifest)
    assert [row["granularity"] for row in labels] == ["coarse", "fine"]
    assert diagnostics["synthetic_count"] == 0
    assert all(row["source_sha256"] and row["触发规则"] for row in labels)
    assert all(row["synthetic"] is False and row["parent_scenario_id"] == "" for row in labels)

def test_map_stats_has_five_maps_and_required_columns():
    rows = build_map_stats(BASE / "RKGRScen/data/maps", BASE / "RKGRScen/data/community_index")
    assert len(rows) == 5
    assert {row["map"] for row in rows} == {f"Town0{i}" for i in range(1, 6)}
    required = {"map", "node_count", "edge_count", "community_count", "junction_community_count", "avg_community_size"}
    traceability = {
        "graph_schema", "graph_builder", "graph_directed", "graph_multigraph", "graph_waypoint_step",
        "graph_source_sha256", "community_algorithm", "community_implementation", "community_resolution",
        "community_seed", "community_min_size", "community_merge_policy", "community_source_sha256",
        "graph_source_file", "community_source_file", "community_index_source", "community_index_map_name",
    }
    assert all(required | traceability == set(row) for row in rows)
    assert all(row["node_count"] > 0 and row["edge_count"] > 0 for row in rows)
    assert all(len(row["graph_source_sha256"]) == 64 and len(row["community_source_sha256"]) == 64 for row in rows)

def _api_response(content):
    return json.dumps({"choices": [{"message": {"content": content}}]})

def test_deepseek_retries_schema_failure_and_audits_without_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-log-this")
    monkeypatch.setenv("DEEPSEEK_RETRY_BACKOFF_S", "0")
    audit = tmp_path / "audit.jsonl"
    client = DeepSeekClient(max_retries=1, audit_jsonl=audit)
    responses = iter([_api_response('{"value": "bad"}'), _api_response('{"value": 3}')])
    monkeypatch.setattr(client, "_request", lambda *args: next(responses))
    schema = {"type": "object", "additionalProperties": False, "required": ["value"], "properties": {"value": {"type": "integer"}}}

    assert client.generate_json(
        "system",
        "user",
        schema=schema,
        audit_metadata={"scenario_id": "scenario_00001", "source_sha256": "a" * 64},
    ) == {"value": 3}
    records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert [record["status"] for record in records] == ["schema_error", "success"]
    assert all(record["scenario_id"] == "scenario_00001" and record["source_sha256"] == "a" * 64 for record in records)
    assert "never-log-this" not in audit.read_text(encoding="utf-8")

def test_scene_expander_forwards_traceable_audit_context(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-log-this")
    expander = SceneExpander(use_llm=True)
    captured = {}
    response = {
        "violation_type": "逆行",
        "subject_mode": "dual",
        "actors": [
            {"id": "A", "role": "violator", "path": "S2N", "action": "Move Forward", "speed_kmh": 30},
            {"id": "B", "role": "priority", "path": "N2S", "action": "Move Forward", "speed_kmh": 30},
        ],
        "conflict": {"type": "x", "location": "Straight", "trigger_condition": "x", "timing": {"time_gap_to_conflict_s": [0, 1]}},
        "road_requirement": {"type": "Straight", "min_lanes": 2, "has_traffic_light": False, "needs_opposing_lanes": True},
    }

    def fake_generate_json(*args, **kwargs):
        captured.update(kwargs["audit_metadata"])
        expander.client.last_metadata = dict(kwargs["audit_metadata"])
        return response

    monkeypatch.setattr(expander.client, "generate_json", fake_generate_json)
    expander.expand({"violation_type": "逆行", "road_network": {"type": "Straight"}}, scenario_id="scenario_00009", source_sha256="b" * 64)
    assert captured["scenario_id"] == "scenario_00009"
    assert captured["source_sha256"] == "b" * 64
    assert "key" not in captured

def test_deepseek_network_retry_is_bounded(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setenv("DEEPSEEK_RETRY_BACKOFF_S", "0")
    client = DeepSeekClient(max_retries=2)
    attempts = {"count": 0}

    def fail(*args):
        attempts["count"] += 1
        raise error.URLError("offline")

    monkeypatch.setattr(client, "_request", fail)
    with pytest.raises(RuntimeError, match="3 次有界尝试"):
        client.generate_json("system", "user", schema={"type": "object"})
    assert attempts["count"] == 3

def test_component_schemas_reject_unapproved_values():
    tagger = CommunityTagger(use_llm=False)
    expander = SceneExpander(use_llm=False)
    with pytest.raises(ValidationError):
        from jsonschema import validate
        validate({"summary": "x", "applicable_violations": ["虚构标签"]}, tagger.output_schema)
    with pytest.raises(ValidationError):
        from jsonschema import validate
        validate(
            {
                "violation_type": "逆行", "subject_mode": "dual",
                "actors": [{"id": "A", "role": "invented", "path": "N2S", "action": "Move Forward", "speed_kmh": 30}],
                "conflict": {"type": "x", "location": "x", "trigger_condition": "x", "timing": {"time_gap_to_conflict_s": [0, 1]}},
                "road_requirement": {"type": "Straight", "min_lanes": 2, "has_traffic_light": False, "needs_opposing_lanes": True},
            },
            expander.output_schema,
        )

@pytest.fixture(scope="module")
def generated_shared(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("rq1_rq2_shared")
    organized_root = BASE / "organized_scenarios"
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in organized_root.glob("**/*.json")
    }
    args = (
        output_dir,
        BASE / "RKGRScen/data/retrieval/p0_graphrag_retrieval_full/retrieval_results.json",
        BASE / "RKGRScen/data/maps",
        BASE / "RKGRScen/data/community_index",
    )
    metadata = generate(*args)
    first_hashes = {
        str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_dir.glob("**/*") if path.is_file()
    }
    second_metadata = generate(*args)
    second_hashes = {
        str(path.relative_to(output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output_dir.glob("**/*") if path.is_file()
    }
    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in organized_root.glob("**/*.json")
    }
    return output_dir, metadata, second_metadata, first_hashes, second_hashes, before, after

def test_generated_shared_set_preserves_2005_and_appends_minimum_fine_scenarios(generated_shared):
    output_dir, metadata, _, _, _, before, after = generated_shared
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    with (output_dir / "granularity_labels.csv").open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))

    final_count = EXPECTED_SCENARIOS + DERIVED_FINE_SCENARIOS
    assert manifest["scenario_count"] == final_count
    assert manifest["original_scenario_count"] == EXPECTED_SCENARIOS
    assert manifest["synthetic_scenario_count"] == DERIVED_FINE_SCENARIOS
    assert len(manifest["rows"]) == len(labels) == final_count
    assert [row["scenario_id"] for row in manifest["rows"]] == [f"scenario_{index:05d}" for index in range(1, final_count + 1)]
    assert {row["seed"] for row in manifest["rows"]} == {FIXED_SEED}
    assert all(not row["synthetic"] for row in manifest["rows"][:EXPECTED_SCENARIOS])
    assert sum(row["granularity"] == "fine" for row in labels) / final_count >= 0.30
    assert (sum(row["granularity"] == "fine" for row in labels) - 1) / (final_count - 1) < 0.30
    assert metadata["dataset_extension"]["synthetic_fine_count"] == DERIVED_FINE_SCENARIOS
    assert metadata["dataset_extension"]["final_count"] == final_count
    assert metadata["fixed_seed"] == FIXED_SEED
    assert metadata["repetitions_per_method_variant"] == 1
    assert metadata["k_fold_or_multi_seed"] is False
    assert "rq1_rq2" in metadata["scenario_id_scope"]
    assert before == after

def test_generation_is_deterministic_and_idempotent(generated_shared):
    _, metadata, second_metadata, first_hashes, second_hashes, _, _ = generated_shared
    assert metadata == second_metadata
    assert first_hashes == second_hashes

def test_synthetic_manifest_rows_are_fine_and_traceable(generated_shared):
    output_dir, _, _, _, _, _, _ = generated_shared
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    original_by_id = {row["scenario_id"]: row for row in manifest["rows"][:EXPECTED_SCENARIOS]}
    synthetic = manifest["rows"][EXPECTED_SCENARIOS:]
    assert len(synthetic) == DERIVED_FINE_SCENARIOS
    for row in synthetic:
        source = Path(row["source_path"])
        parent = original_by_id[row["parent_scenario_id"]]
        assert row["synthetic"] is True
        assert source.parent == output_dir / "synthetic_fine_scenarios"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
        assert row["original_source_path"] == parent["source_path"]
        assert row["original_source_sha256"] == parent["source_sha256"]
        payload = json.loads(source.read_text(encoding="utf-8"))
        parent_payload = json.loads(Path(parent["source_path"]).read_text(encoding="utf-8"))
        assert classify_granularity(payload) == ("fine", "多跳拓扑约束")
        assert len(payload["dsl"]["road_network"]["topology_sequence"]) >= 2
        assert payload["violation_type"] == parent_payload["violation_type"]
        assert payload["dsl"].get("actors") == parent_payload["dsl"].get("actors")
        assert payload["provenance"]["parent_scenario_id"] == row["parent_scenario_id"]
        assert payload["provenance"]["synthetic"] is True
        assert payload["provenance"]["generation_rule"] == row["generation_rule"]

def test_granularity_csv_has_required_traceability_columns(generated_shared):
    output_dir, _, _, _, _, _, _ = generated_shared
    with (output_dir / "granularity_labels.csv").open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert set(labels[0]) == {"scenario_id", "source_sha256", "synthetic", "parent_scenario_id", "granularity", "触发规则"}
    assert all(len(row["source_sha256"]) == 64 for row in labels)
    assert all(row["parent_scenario_id"] for row in labels[EXPECTED_SCENARIOS:])

def test_generated_artifact_hashes_and_five_map_stats_are_traceable(generated_shared):
    output_dir, metadata, _, _, _, _, _ = generated_shared
    for key in ("manifest", "granularity_labels", "map_graph_stats"):
        artifact = metadata[key]
        assert hashlib.sha256((output_dir / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    with (output_dir / "map_graph_stats.csv").open(encoding="utf-8", newline="") as handle:
        map_rows = list(csv.DictReader(handle))
    assert {row["map"] for row in map_rows} == {f"Town0{i}" for i in range(1, 6)}
    assert all(row["community_algorithm"] and row["graph_source_sha256"] and row["community_source_sha256"] for row in map_rows)
