import argparse
import copy
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

FIXED_SEED = 20260714
EXPECTED_SCENARIOS = 2005
EXPECTED_ORIGINAL_FINE = 151
TARGET_FINE_NUMERATOR = 3
TARGET_FINE_DENOMINATOR = 10
DERIVED_FINE_SCENARIOS = 644
EXPECTED_FINE = EXPECTED_ORIGINAL_FINE + DERIVED_FINE_SCENARIOS
SCHEMA_VERSION = "rq1-rq2-manifest-v3"
PROTOCOL_VERSION = "rq1-rq2-single-seed-v3"
GENERATION_RULE = "deterministic_topology_sequence_extension_v3"
TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05")
SOURCE_TYPES = {
    "未保持安全距离",
    "未按规定让行",
    "未注意前方路况",
    "违规变道",
    "违规超车",
    "逆行",
    "超速行驶",
    "超速",
}
TYPE_NORMALIZATION = {"超速行驶": "超速"}
NOT_APPLICABLE = {"", "not applicable", "n/a", "none", "unknown"}
MANIFEST_ROW_FIELDS = (
    "scenario_id",
    "seed",
    "source_path",
    "source_sha256",
    "original_source_path",
    "original_source_sha256",
    "violation_type",
    "source_violation_type",
    "synthetic",
    "parent_scenario_id",
    "generation_rule",
)

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

def dump_json(payload: Dict[str, Any], path: Path, *, atomic: bool = False) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if atomic:
        atomic_write_text(path, content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

def scenario_id_for(index: int) -> str:
    return f"scenario_{index:05d}"

def source_rows(retrieval_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in retrieval_payload.get("results", []):
        if row.get("status") != "ok" or row.get("source_violation_type") not in SOURCE_TYPES:
            continue
        if not (row.get("scenario_spec") or row.get("retrieval", {}).get("scenario_spec")):
            continue
        selected.append(row)
    if len(selected) != EXPECTED_SCENARIOS:
        raise ValueError(f"统一输入应为 {EXPECTED_SCENARIOS} 条，实际为 {len(selected)} 条")
    return selected

def _protocol_descriptor(seed: int, scenario_count: int) -> Dict[str, Any]:
    return {
        "name": "rq1_rq2_shared_single_seed",
        "version": PROTOCOL_VERSION,
        "seed": seed,
        "repetitions_per_method_variant": 1,
        "scenario_count": scenario_count,
        "scenario_id_scope": "shared_across_all_rq1_rq2_methods_and_variants",
        "method_or_variant_in_scenario_id": False,
    }

def _schema_descriptor() -> Dict[str, Any]:
    return {
        "name": "rq1_rq2_shared_manifest",
        "version": SCHEMA_VERSION,
        "required_row_fields": list(MANIFEST_ROW_FIELDS),
    }

def build_manifest(rows: Sequence[Dict[str, Any]], seed: int = FIXED_SEED) -> Dict[str, Any]:
    manifest_rows: List[Dict[str, Any]] = []
    seen_paths = set()
    for index, row in enumerate(rows, 1):
        source_path = str(Path(row["source_path"]).resolve())
        if source_path in seen_paths:
            raise ValueError(f"manifest 中存在重复 source_path: {source_path}")
        seen_paths.add(source_path)
        original_source_path = str(Path(row.get("original_source_path", source_path)).resolve())
        source_type = str(row["source_violation_type"])
        synthetic = bool(row.get("synthetic", False))
        parent_scenario_id = str(row.get("parent_scenario_id", ""))
        if synthetic and not parent_scenario_id:
            raise ValueError(f"synthetic row 缺少 parent_scenario_id: {source_path}")
        manifest_rows.append(
            {
                "scenario_id": scenario_id_for(index),
                "seed": seed,
                "source_path": source_path,
                "source_sha256": sha256_file(Path(source_path)),
                "original_source_path": original_source_path,
                "original_source_sha256": row.get("original_source_sha256") or sha256_file(Path(original_source_path)),
                "violation_type": TYPE_NORMALIZATION.get(source_type, source_type),
                "source_violation_type": source_type,
                "synthetic": synthetic,
                "parent_scenario_id": parent_scenario_id,
                "generation_rule": row.get("generation_rule", "") if synthetic else "",
            }
        )
    protocol = _protocol_descriptor(seed, len(manifest_rows))
    schema = _schema_descriptor()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "schema_sha256": sha256_bytes(_canonical_json(schema)),
        "protocol_sha256": sha256_bytes(_canonical_json(protocol)),
        "protocol": protocol["name"],
        "seed": seed,
        "scenario_count": len(manifest_rows),
        "original_scenario_count": sum(not row["synthetic"] for row in manifest_rows),
        "synthetic_scenario_count": sum(row["synthetic"] for row in manifest_rows),
        "frozen": True,
        "scenario_id_scope": protocol["scenario_id_scope"],
        "scenario_id_derivation": "fixed_manifest_row_order_only; method/variant excluded",
        "rows_sha256": sha256_bytes(_canonical_json(manifest_rows)),
        "rows": manifest_rows,
    }

def _synthetic_payload(parent_payload: Dict[str, Any], parent: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(parent_payload)
    dsl = payload.setdefault("dsl", {})
    road = dsl.setdefault("road_network", {})
    road_type = str(road.get("type") or "RoadSegment")
    next_type = "StraightExit" if road_type.lower() in {"intersection", "t-intersection", "junction"} else "IntersectionApproach"
    road["topology_sequence"] = [
        {"order": 1, "element_type": road_type, "relation_to_next": "immediately_connects_to"},
        {"order": 2, "element_type": next_type, "relation_to_previous": "immediately_follows"},
    ]
    payload["provenance"] = {
        "parent_scenario_id": parent["scenario_id"],
        "parent_source_path": parent["source_path"],
        "parent_source_sha256": parent["source_sha256"],
        "synthetic": True,
        "generation_rule": GENERATION_RULE,
        "seed": FIXED_SEED,
    }
    return payload

def required_synthetic_count(original_total: int, original_fine: int) -> int:
    deficit = TARGET_FINE_NUMERATOR * original_total - TARGET_FINE_DENOMINATOR * original_fine
    if deficit <= 0:
        return 0
    return math.ceil(deficit / (TARGET_FINE_DENOMINATOR - TARGET_FINE_NUMERATOR))

def generate_fine_scenarios(
    original_manifest: Dict[str, Any], output_dir: Path, synthetic_count: int
) -> List[Dict[str, Any]]:
    coarse_parents = [
        row
        for row in original_manifest["rows"]
        if classify_granularity(load_json(Path(row["source_path"])))[0] == "coarse"
    ]
    if len(coarse_parents) < synthetic_count:
        raise ValueError(f"coarse parent 不足: {len(coarse_parents)}")

    generated_dir = output_dir / "synthetic_fine_scenarios"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Dict[str, Any]] = []
    for offset, parent in enumerate(coarse_parents[:synthetic_count], 1):
        original_path = Path(parent["source_path"])
        parent_payload = load_json(original_path)
        payload = _synthetic_payload(parent_payload, parent)
        if payload.get("violation_type") != parent_payload.get("violation_type"):
            raise ValueError(f"扩展改变了违规类别: {parent['scenario_id']}")
        if payload.get("dsl", {}).get("actors") != parent_payload.get("dsl", {}).get("actors"):
            raise ValueError(f"扩展改变了 actors: {parent['scenario_id']}")
        if classify_granularity(payload)[0] != "fine":
            raise ValueError(f"扩展场景未判为 fine: {parent['scenario_id']}")
        synthetic_id = scenario_id_for(len(original_manifest["rows"]) + offset)
        generated_path = generated_dir / f"{synthetic_id}.json"
        dump_json(payload, generated_path, atomic=True)
        generated.append(
            {
                "source_path": str(generated_path.resolve()),
                "source_violation_type": parent["source_violation_type"],
                "original_source_path": parent["source_path"],
                "original_source_sha256": parent["source_sha256"],
                "synthetic": True,
                "parent_scenario_id": parent["scenario_id"],
                "generation_rule": GENERATION_RULE,
            }
        )
    return generated

def _multi_hop_rule(road: Dict[str, Any]) -> bool:
    for key in ("elements", "element_sequence", "road_sequence", "topology_sequence"):
        value = road.get(key)
        if isinstance(value, list) and len(value) >= 2:
            return True
    return False

def _composite_node_rule(road: Dict[str, Any]) -> bool:
    constraints = road.get("node_constraints")
    if isinstance(constraints, list) and len(constraints) >= 3 and all(isinstance(item, dict) and item.get("attribute") for item in constraints):
        return True
    explicit_attributes = 0
    if isinstance(road.get("type"), str) and road["type"].strip():
        explicit_attributes += 1
    if isinstance(road.get("lanes"), (int, float)) and road["lanes"] > 0:
        explicit_attributes += 1
    stem = str(road.get("stem_direction", "")).strip().lower()
    if stem not in NOT_APPLICABLE:
        explicit_attributes += 1
    return explicit_attributes >= 3

def _role_conflict_rule(dsl: Dict[str, Any]) -> bool:
    geometry = dsl.get("conflict_geometry")
    if not isinstance(geometry, dict):
        return False
    participants = geometry.get("participants")
    return bool(
        isinstance(participants, list)
        and len(participants) >= 2
        and geometry.get("conflict_point")
        and all(isinstance(item, dict) and item.get("role") and item.get("lane_connection") for item in participants[:2])
    )

def classify_granularity(payload: Dict[str, Any]) -> Tuple[str, str]:
    dsl = payload.get("dsl", payload)
    road = dsl.get("road_network", {}) if isinstance(dsl, dict) else {}
    if _multi_hop_rule(road):
        return "fine", "多跳拓扑约束"
    if _composite_node_rule(road):
        return "fine", "复合节点约束"
    if isinstance(dsl, dict) and _role_conflict_rule(dsl):
        return "fine", "带角色的冲突几何"
    return "coarse", "未触发G3细粒度规则"

def build_granularity_labels(manifest: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    trigger_counts: Counter = Counter()
    synthetic_count = 0
    for row in manifest["rows"]:
        source_path = Path(row["source_path"])
        granularity, trigger = classify_granularity(load_json(source_path))
        labels.append(
            {
                "scenario_id": row["scenario_id"],
                "source_sha256": row.get("source_sha256") or sha256_file(source_path),
                "synthetic": bool(row.get("synthetic", False)),
                "parent_scenario_id": row.get("parent_scenario_id", ""),
                "granularity": granularity,
                "触发规则": trigger,
            }
        )
        counts[granularity] += 1
        trigger_counts[trigger] += 1
        synthetic_count += int(bool(row.get("synthetic")))
    total = len(labels)
    fine = counts["fine"]
    target = math.ceil(total * 0.30)
    return labels, {
        "total": total,
        "fine_count": fine,
        "coarse_count": counts["coarse"],
        "fine_ratio": fine / total if total else 0.0,
        "fine_ratio_percent": round(100.0 * fine / total, 4) if total else 0.0,
        "target_ratio": 0.30,
        "target_reached": fine / total >= 0.30 if total else False,
        "target_fine_count": target,
        "fine_count_gap": max(0, target - fine),
        "trigger_counts": dict(trigger_counts),
        "synthetic_count": synthetic_count,
    }

def build_map_stats(maps_dir: Path, community_dir: Path) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for town in TOWNS:
        graph_path = maps_dir / f"Carla_Maps_{town}_graph.json"
        community_path = community_dir / f"{town}_community_detection_summary.json"
        graph = load_json(graph_path)
        summary = load_json(community_path)
        communities = summary.get("communities", [])
        community_count = len(communities)
        junction_count = sum(int(item.get("structure", {}).get("junctions", 0)) > 0 for item in communities)
        node_count = len(graph.get("nodes", []))
        edge_count = len(graph.get("links", graph.get("edges", [])))
        graph_metadata = graph.get("graph", {})
        community_method = str(summary.get("community_method", "unknown"))
        stats.append(
            {
                "map": town,
                "node_count": node_count,
                "edge_count": edge_count,
                "community_count": community_count,
                "junction_community_count": junction_count,
                "avg_community_size": round(node_count / community_count, 6) if community_count else 0.0,
                "graph_schema": "networkx_node_link_json",
                "graph_builder": "RKGRScen.indexing.graph_builder.RoadGraphBuilder",
                "graph_directed": bool(graph.get("directed")),
                "graph_multigraph": bool(graph.get("multigraph")),
                "graph_waypoint_step": graph_metadata.get("waypoint_step"),
                "graph_source_sha256": sha256_file(graph_path),
                "community_algorithm": community_method,
                "community_implementation": "networkx.algorithms.community",
                "community_resolution": summary.get("resolution", ""),
                "community_seed": summary.get("community_seed", summary.get("seed", "")),
                "community_min_size": summary.get("min_community_size", ""),
                "community_merge_policy": "merge_small_communities_by_weighted_affinity",
                "community_source_sha256": sha256_file(community_path),
                "graph_source_file": str(graph_path.resolve()),
                "community_source_file": str(community_path.resolve()),
                "community_index_source": "communities[].sample_nodes",
                "community_index_map_name": summary.get("map_name", ""),
            }
        )
    return stats

def generate(output_dir: Path, retrieval_path: Path, maps_dir: Path, community_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = source_rows(load_json(retrieval_path))
    original_manifest = build_manifest(rows)
    _, original_granularity = build_granularity_labels(original_manifest)
    if original_granularity["fine_count"] != EXPECTED_ORIGINAL_FINE:
        raise ValueError(
            f"原始 fine 数应为 {EXPECTED_ORIGINAL_FINE}，实际为 {original_granularity['fine_count']}；禁止在输入漂移时静默补造"
        )

    synthetic_count = required_synthetic_count(len(rows), original_granularity["fine_count"])
    if synthetic_count != DERIVED_FINE_SCENARIOS:
        raise ValueError(f"最小扩展数应为 {DERIVED_FINE_SCENARIOS}，实际计算为 {synthetic_count}")
    synthetic_rows = generate_fine_scenarios(original_manifest, output_dir, synthetic_count)
    manifest = build_manifest([*rows, *synthetic_rows])
    labels, granularity = build_granularity_labels(manifest)
    if len(manifest["rows"]) != EXPECTED_SCENARIOS + synthetic_count or granularity["fine_count"] != EXPECTED_FINE:
        raise ValueError(f"扩展后统计异常: total={len(manifest['rows'])}, fine={granularity['fine_count']}")
    if granularity["fine_ratio"] < 0.30:
        raise ValueError(f"fine 比例不足30%: {granularity['fine_ratio']:.6f}")
    if synthetic_count and (EXPECTED_FINE - 1) / (len(manifest["rows"]) - 1) >= 0.30:
        raise ValueError("扩展场景数量不是达到30%所需的最小值")

    map_stats = build_map_stats(maps_dir, community_dir)
    manifest_path = output_dir / "manifest.json"
    labels_path = output_dir / "granularity_labels.csv"
    stats_path = output_dir / "map_graph_stats.csv"
    dump_json(manifest, manifest_path, atomic=True)
    write_csv(
        labels_path,
        labels,
        ["scenario_id", "source_sha256", "synthetic", "parent_scenario_id", "granularity", "触发规则"],
    )
    stats_columns = list(map_stats[0]) if map_stats else []
    write_csv(stats_path, map_stats, stats_columns)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol": "rq1_rq2_shared_single_seed",
        "protocol_sha256": manifest["protocol_sha256"],
        "fixed_seed": FIXED_SEED,
        "repetitions_per_method_variant": 1,
        "execution_protocol": "fixed_seed_single_execution_per_scenario_method_or_variant",
        "k_fold_or_multi_seed": False,
        "scenario_id_scope": manifest["scenario_id_scope"],
        "manifest": {"path": "manifest.json", "sha256": sha256_file(manifest_path), "rows_sha256": manifest["rows_sha256"]},
        "granularity_labels": {"path": "granularity_labels.csv", "sha256": sha256_file(labels_path)},
        "map_graph_stats": {"path": "map_graph_stats.csv", "sha256": sha256_file(stats_path)},
        "old_results_overwritten": False,
        "output_directory": str(output_dir.resolve()),
        "source_rows_validated": EXPECTED_SCENARIOS,
        "dataset_extension": {
            "original_count": EXPECTED_SCENARIOS,
            "synthetic_fine_count": synthetic_count,
            "final_count": manifest["scenario_count"],
            "original_fine_count": original_granularity["fine_count"],
            "final_fine_count": granularity["fine_count"],
            "final_fine_ratio": granularity["fine_ratio"],
            "minimum_extension_for_30_percent": True,
            "generation_rule": GENERATION_RULE,
            "generated_directory": "synthetic_fine_scenarios",
            "original_scenarios_preserved": True,
            "llm_used": False,
        },
        "granularity_before_extension": original_granularity,
        "granularity_final": granularity,
    }
    dump_json(metadata, output_dir / "protocol_metadata.json", atomic=True)
    return metadata

def main() -> None:
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate shared RQ1/RQ2 single-seed infrastructure without CARLA or LLM calls.")
    parser.add_argument("--output-dir", default="RKGRScen/data/evaluation/rq1_rq2_20260714_shared")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = base / output
    metadata = generate(
        output,
        base / "RKGRScen/data/retrieval/p0_graphrag_retrieval_full/retrieval_results.json",
        base / "RKGRScen/data/maps",
        base / "RKGRScen/data/community_index",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
