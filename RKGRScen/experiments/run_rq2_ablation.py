import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from RKGRScen.experiments.batch_utils import carla_health_check, classify_failure, dump_json
from RKGRScen.experiments.run_rq1_carla_full_execution import (
    DEFAULT_SHARED_MANIFEST,
    enrich_manifest_rows,
    load_shared_manifest,
    local_to_retrieval_result,
    manifest_metadata,
    manifest_rows as select_manifest_rows,
    normalize_node,
    normalize_type,
    remove_invalid_environment_result,
    retrieval_index,
)
from RKGRScen.execution.carla_runner import CarlaScenarioRunner
from RKGRScen.execution.violation_detector import detect_violation
from RKGRScen.models import RetrievalResult, ScenarioConfiguration
from RKGRScen.query.constraint_solver import ConstraintSolver
from RKGRScen.query.constraint_validator import ConstraintValidator
from RKGRScen.query.retriever import GraphRetriever
from RKGRScen.query.scenario_match_evaluator import ScenarioMatchEvaluator
from RKGRScen.query.scene_expander import SceneExpander

VARIANTS = ("full", "without_community", "without_expansion", "without_constraint", "without_semantic_summaries")
RQ2_SOURCE_TYPES = ("未保持安全距离", "未按规定让行", "未注意前方路况", "违规变道", "违规超车", "逆行", "超速行驶", "超速")
ENVIRONMENT_FAILURES = {"simulator_unhealthy_before", "simulator_unhealthy_after"}
RECOVERABLE_FAILURES = ENVIRONMENT_FAILURES | {"llm_expansion_failed"}
STAGE_NAMES = ("expansion", "retrieval", "configuration", "constraint_validation", "execution", "violation_detection", "scene_match_evaluation")

class RandomSamplingUnsatisfied(RuntimeError):
    pass

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    columns = list(fieldnames or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def cache_path(cache_dir: Path, source_path: str) -> Path:
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"

def stable_case_seed(global_seed: int, scenario_id: str, source_path: str) -> int:
    material = f"{global_seed}\0{scenario_id}\0{source_path}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

def scenario_id_for(variant: str, original_case_number: int) -> str:
    return f"scenario_{original_case_number:05d}"

def normalize_llm_spec(spec: Dict[str, Any], source: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    normalized = dict(spec)
    source_violation_type = normalize_type(str(source.get("violation_type", "")))
    model_violation_type = normalize_type(str(normalized.get("violation_type", "")))
    violation_type = source_violation_type or model_violation_type
    normalized["violation_type"] = violation_type
    normalized.setdefault("subject_mode", "dual" if len(normalized.get("actors", [])) > 1 else "single")
    normalized.setdefault("actors", [])
    conflict = dict(normalized.get("conflict") or {})
    conflict.setdefault("type", f"{violation_type} conflict")
    conflict.setdefault("location", source.get("dsl", {}).get("road_network", {}).get("type", "RoadSegment"))
    conflict.setdefault("trigger_condition", f"trigger {violation_type}")
    timing = dict(conflict.get("timing") or {})
    if not isinstance(timing.get("time_gap_to_conflict_s"), list) or len(timing["time_gap_to_conflict_s"]) != 2:
        timing["time_gap_to_conflict_s"] = [0.5, 2.0]
    conflict["timing"] = timing
    normalized["conflict"] = conflict
    road = dict(normalized.get("road_requirement") or {})
    source_road = source.get("dsl", {}).get("road_network", {})
    road.setdefault("type", source_road.get("type", "RoadSegment"))
    road.setdefault("min_lanes", max(1, int(source_road.get("min_lanes", source_road.get("lanes", 1)) or 1)))
    road.setdefault("has_traffic_light", False)
    road.setdefault("needs_opposing_lanes", violation_type in {"未按规定让行", "逆行", "违规超车"})
    filled: List[str] = []
    if model_violation_type and model_violation_type != violation_type:
        filled.append("violation_type")
    if violation_type in {"违规变道", "违规超车"}:
        action = str((normalized.get("actors") or [{}])[0].get("action", "Change Lane Left"))
        defaults = {
            "same_direction_multi_lane": True,
            "lane_change_direction": "right" if "right" in action.lower() else "left",
            "min_lanes": max(2, int(road.get("min_lanes", 2))),
        }
        for key, value in defaults.items():
            if key not in road or key == "min_lanes" and int(road[key]) < 2:
                road[key] = value
                filled.append(f"road_requirement.{key}")
    if violation_type == "未保持安全距离" or violation_type == "未注意前方路况" and road.get("type") in {"RoadSegment", "Straight"}:
        defaults = {"needs_long_straight": True, "min_segment_length_m": 25.0}
        for key, value in defaults.items():
            if key not in road:
                road[key] = value
                filled.append(f"road_requirement.{key}")
    if violation_type == "超速":
        speed = dict(normalized.get("speed_requirement") or {})
        speed_limit = float(speed.get("speed_limit_kmh", source_road.get("speed_limit_kmh", source.get("dsl", {}).get("speed_limit_kmh", 40))) or 40)
        target_speed = float(speed.get("target_speed_kmh", max(speed_limit + 15, 55)))
        speed.update({"speed_limit_kmh": speed_limit, "target_speed_kmh": target_speed})
        normalized["speed_requirement"] = speed
        road["speed_limit_kmh"] = speed_limit
        conflict["target_speed_kmh"] = target_speed
        filled.extend(["road_requirement.speed_limit_kmh", "conflict.target_speed_kmh"])
    normalized["road_requirement"] = road
    return normalized, sorted(set(filled))

def _actor_defaults(violation_type: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    roles = {
        "未保持安全距离": ["ego", "lead"],
        "未按规定让行": ["violator", "priority"],
        "未注意前方路况": ["ego"],
        "违规变道": ["violator", "priority"],
        "违规超车": ["violator", "priority"],
        "逆行": ["violator", "priority"],
        "超速": ["ego"],
    }[violation_type]
    fallback = {
        "未保持安全距离": [
            {"id": "ego", "role": "ego", "path": "straight", "action": "Move Forward", "speed_kmh": 35},
            {"id": "lead", "role": "lead", "path": "ahead", "action": "Lead Brake", "speed_kmh": 20},
        ],
        "未按规定让行": [
            {"id": "A", "role": "violator", "path": "S2N", "action": "Turn Left", "speed_kmh": 35},
            {"id": "B", "role": "priority", "path": "N2S", "action": "Move Forward", "speed_kmh": 35},
        ],
        "未注意前方路况": [{"id": "ego", "role": "ego", "path": "straight", "action": "Move Forward", "speed_kmh": 35}],
        "违规变道": [
            {"id": "A", "role": "violator", "path": "straight", "action": "Change Lane Left", "speed_kmh": 45},
            {"id": "B", "role": "priority", "path": "straight", "action": "Move Forward", "speed_kmh": 40},
        ],
        "违规超车": [
            {"id": "A", "role": "violator", "path": "straight", "action": "Overtake Left", "speed_kmh": 55},
            {"id": "B", "role": "priority", "path": "straight", "action": "Move Forward", "speed_kmh": 35},
        ],
        "逆行": [
            {"id": "A", "role": "violator", "path": "S2N", "action": "Move Forward", "speed_kmh": 35},
            {"id": "B", "role": "priority", "path": "N2S", "action": "Move Forward", "speed_kmh": 35},
        ],
        "超速": [{"id": "A", "role": "ego", "path": "straight", "action": "Move Forward", "speed_kmh": 60}],
    }[violation_type]
    return roles, fallback

def build_skeleton_spec(source: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    violation_type = normalize_type(str(source.get("violation_type", "")))
    if violation_type not in {"未保持安全距离", "未按规定让行", "未注意前方路况", "违规变道", "违规超车", "逆行", "超速"}:
        raise ValueError(f"without_expansion 不支持类型: {violation_type}")
    dsl = dict(source.get("dsl") or {})
    source_actors = list(dsl.get("actors") or [])
    roles, fallback = _actor_defaults(violation_type)
    actors: List[Dict[str, Any]] = []
    defaulted_fields: List[str] = []
    for index, role in enumerate(roles):
        raw = source_actors[index] if index < len(source_actors) else {}
        default = fallback[index]
        action = raw.get("action", raw.get("actions", default["action"]))
        speed = raw.get("speed_kmh", raw.get("speed", raw.get("init_speed_kmh", raw.get("speed_limit", default["speed_kmh"]))))
        path = raw.get("path", raw.get("initial_position", default["path"]))
        actors.append({"id": raw.get("id", default["id"]), "role": role, "path": path, "action": action, "speed_kmh": float(speed)})
        for key, present in (("id", "id" in raw), ("role", "role" in raw), ("path", "path" in raw or "initial_position" in raw), ("action", "action" in raw or "actions" in raw), ("speed_kmh", any(item in raw for item in ("speed_kmh", "speed", "init_speed_kmh", "speed_limit")))):
            if not present:
                defaulted_fields.append(f"actors[{index}].{key}")
    if violation_type == "未注意前方路况":
        actors.append({"id": "obstacle", "role": "obstacle", "path": "ahead", "action": "Static Obstacle", "speed_kmh": 0.0})
        defaulted_fields.append("actors[1]")
    road_source = dict(dsl.get("road_network") or {})
    road_type = road_source.get("type", "Intersection" if violation_type == "未按规定让行" else "Straight")
    min_lanes = max(2 if violation_type in {"未按规定让行", "违规变道", "违规超车", "逆行"} else 1, int(road_source.get("min_lanes", road_source.get("lanes", 1)) or 1))
    road_requirement = {
        "type": road_type,
        "min_lanes": min_lanes,
        "has_traffic_light": bool(road_source.get("has_traffic_light", False)),
        "needs_opposing_lanes": violation_type in {"未按规定让行", "逆行", "违规超车"},
    }
    if violation_type in {"违规变道", "违规超车"}:
        action = str(actors[0]["action"])
        road_requirement.update({"same_direction_multi_lane": True, "lane_change_direction": "right" if "right" in action.lower() else "left"})
    if violation_type == "未保持安全距离":
        road_requirement.update({"needs_long_straight": True, "min_segment_length_m": 25.0})
    elif violation_type == "未注意前方路况" and road_type in {"RoadSegment", "Straight"}:
        road_requirement.update({"needs_long_straight": True, "min_segment_length_m": 25.0})
    timing = {
        "未保持安全距离": [0.0, 3.0], "未按规定让行": [0.5, 2.0], "未注意前方路况": [0.0, 3.0],
        "违规变道": [0.3, 1.2], "违规超车": [0.2, 1.5], "逆行": [0.5, 2.0], "超速": [0.0, 0.0],
    }[violation_type]
    spec: Dict[str, Any] = {
        "violation_type": violation_type,
        "subject_mode": "single" if violation_type == "超速" else "dual",
        "actors": actors,
        "conflict": {"type": f"{violation_type} default conflict", "location": road_type, "trigger_condition": f"default {violation_type} trigger", "timing": {"time_gap_to_conflict_s": timing}},
        "road_requirement": road_requirement,
    }
    if violation_type == "超速":
        speed_limit = float(road_source.get("speed_limit_kmh", dsl.get("speed_limit_kmh", 40)) or 40)
        target_speed = max(float(actors[0]["speed_kmh"]), speed_limit + 15)
        spec["speed_requirement"] = {"speed_limit_kmh": speed_limit, "target_speed_kmh": target_speed}
        spec["road_requirement"]["speed_limit_kmh"] = speed_limit
        spec["conflict"]["target_speed_kmh"] = target_speed
    diagnostics = {"mode": "fixed_skeleton", "defaulted_fields": sorted(defaulted_fields), "source_actor_count": len(source_actors)}
    return spec, diagnostics

def expansion_for(
    source_path: str,
    source: Dict[str, Any],
    variant: str,
    output_root: Path,
    *,
    scenario_id: str = "",
    source_sha256: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if variant == "without_expansion":
        return build_skeleton_spec(source)
    path = cache_path(output_root / "shared_cache" / "expansions", source_path)
    legacy_path = cache_path(Path("/home/zxy/apollo/data/test/point2/RKGRScen/data/evaluation/rq2_ablation_full_execution_invalid_mixed_schema_20260719_155218") / "shared_cache" / "expansions", source_path)
    if not path.exists() and legacy_path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(load_json(legacy_path), path)
    if path.exists():
        payload = load_json(path)
        if not payload.get("requested_use_llm") or not payload.get("effective_use_llm"):
            raise RuntimeError(f"无效 LLM expansion 缓存（requested/effective 必须均为 true）: {path}")
        spec, filled_fields = normalize_llm_spec(dict(payload["scenario_spec"]), source)
        if spec != payload.get("scenario_spec") or filled_fields != payload.get("filled_fields", []):
            payload["scenario_spec"] = spec
            payload["filled_fields"] = filled_fields
            dump_json(payload, path)
        return spec, {"cache": "hit", "cache_path": str(path), "filled_fields": filled_fields}
    expander = SceneExpander(use_llm=True)
    if not expander.use_llm:
        raise RuntimeError("Full 消融要求真实 LLM expansion，但 SceneExpander effective use_llm=False；请配置 DEEPSEEK_API_KEY，禁止静默降级")
    spec, filled_fields = normalize_llm_spec(
        expander.expand(
            source,
            scenario_id=scenario_id,
            source_sha256=source_sha256 or hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        ),
        source,
    )
    payload = {
        "source_path": source_path,
        "requested_use_llm": True,
        "effective_use_llm": bool(expander.use_llm),
        "scenario_spec": spec,
        "filled_fields": filled_fields,
    }
    dump_json(payload, path)
    return spec, {"cache": "miss", "cache_path": str(path), "filled_fields": filled_fields}

def retrieval_for(spec: Dict[str, Any], source_path: str, variant: str, output_root: Path, base: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    retriever = GraphRetriever(base / "RKGRScen")
    if variant == "without_community":
        result = retriever.retrieve_without_community(spec, top_k=3)
        return result, {"mode": "full_graph", "cache": "disabled"}
    if variant == "without_semantic_summaries":
        if hasattr(retriever, "retrieve_without_semantic_summaries"):
            result = retriever.retrieve_without_semantic_summaries(spec, top_k=3)
            return result, {"mode": "community_without_semantic_summaries", "cache": "disabled", "llm_summary_used": False}
        result = retriever.retrieve(spec, top_k=3)
        return result, {"mode": "community_without_semantic_summaries_fallback", "cache": "disabled", "llm_summary_used": False}
    if variant in {"full", "without_constraint"}:
        path = cache_path(output_root / "shared_cache" / "retrievals", source_path)
        if path.exists():
            payload = load_json(path)
            if payload.get("source_path") != source_path:
                raise RuntimeError(f"retrieval 缓存 source_path 不匹配: {path}")
            cached_spec = payload.get("scenario_spec") or payload.get("retrieval", {}).get("scenario_spec")
            if cached_spec != spec:
                result = retriever.retrieve(spec, top_k=3)
                dump_json({"source_path": source_path, "scenario_spec": spec, "retrieval": result}, path)
                return result, {"mode": "community", "cache": "refreshed", "cache_path": str(path)}
            return dict(payload["retrieval"]), {"mode": "community", "cache": "hit", "cache_path": str(path)}
        result = retriever.retrieve(spec, top_k=3)
        dump_json({"source_path": source_path, "scenario_spec": spec, "retrieval": result}, path)
        return result, {"mode": "community", "cache": "miss", "cache_path": str(path)}
    return retriever.retrieve(spec, top_k=3), {"mode": "community", "cache": "disabled"}

def retrieval_to_solver_result(retrieval: Dict[str, Any], source_violation_type: str) -> RetrievalResult:
    local_top = list(retrieval.get("local_top_k") or [])
    if not local_top:
        raise RuntimeError("检索结果没有 local_top_k")
    top = local_top[0]
    top_town = str(top.get("map_name", "")).split("/")[-1]
    same_town = [item for item in local_top if str(item.get("map_name", "")).split("/")[-1] == top_town]
    row = {"retrieval": {"local_top_k": same_town}, "source_violation_type": source_violation_type}
    result = local_to_retrieval_result(row)
    result.community.map_name = top_town
    if any(str(item.get("map_name", "")).split("/")[-1] != top_town for item in same_town):
        raise RuntimeError("matched_nodes 出现跨 Town 污染")
    return result

def _point(node: Dict[str, Any]) -> Dict[str, float]:
    point = node.get("end") or node.get("start") or {"x": 0.0, "y": 0.0}
    return {"x": float(point.get("x", 0.0)), "y": float(point.get("y", 0.0))}

def _wp(node: Dict[str, Any], s_value: float) -> Dict[str, Any]:
    return {"road_id": node.get("road_id"), "lane_id": node.get("lane_id"), "s": round(s_value, 2)}

def random_configuration(spec: Dict[str, Any], retrieval: Dict[str, Any], scenario_id: str, seed: int) -> Tuple[ScenarioConfiguration, Dict[str, Any]]:
    rng = random.Random(seed)
    local_top = list(retrieval.get("local_top_k") or [])
    if not local_top:
        raise RandomSamplingUnsatisfied("随机采样没有 local_top_k")
    top_town = str(local_top[0].get("map_name", "")).split("/")[-1]
    candidates = [item for item in local_top if str(item.get("map_name", "")).split("/")[-1] == top_town]
    if not candidates:
        raise RandomSamplingUnsatisfied("随机采样没有同 Town 候选")
    chosen = rng.choice(candidates)
    nodes = []
    if chosen.get("match_type") == "lane_pair":
        nodes = [normalize_node(chosen["source_node"], chosen), normalize_node(chosen["target_node"], chosen)]
    else:
        nodes = [normalize_node(item) for item in candidates]
    violation_type = spec["violation_type"]
    ego_s = rng.uniform(8.0, 24.0)
    trigger = rng.uniform(0.0, 1.5)
    ego = {"role": "ego", "spawn_waypoint": _wp(nodes[0], ego_s), "init_speed_kmh": round(rng.uniform(28.0, 42.0), 1), "behavior": "Move Forward (autopilot)"}
    npcs: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {}
    conflict = _point(nodes[0])
    if violation_type in {"未保持安全距离", "未注意前方路况"}:
        gap = rng.uniform(15.0, 30.0)
        behavior = "Lead Brake" if violation_type == "未保持安全距离" else "Static Obstacle"
        npcs = [{"id": "lead" if behavior == "Lead Brake" else "obstacle", "role": "lead" if behavior == "Lead Brake" else "obstacle", "spawn_waypoint": _wp(nodes[0], ego_s + gap), "init_speed_kmh": round(rng.uniform(18.0, 25.0), 1) if behavior == "Lead Brake" else 0.0, "behavior": behavior, "trigger_time_s": trigger, "brake_after_s": rng.uniform(4.0, 9.0)}]
        params = {"thw_threshold_s": 1.0, "min_gap_m": 8.0} if behavior == "Lead Brake" else {"ttc_threshold_s": 3.0, "danger_distance_m": 15.0}
    elif violation_type == "超速":
        speed = spec.get("speed_requirement", {})
        limit = float(speed.get("speed_limit_kmh", 40.0))
        target = float(speed.get("target_speed_kmh", limit + 15.0))
        npcs = [{"id": "speeding_npc", "role": "violator", "spawn_waypoint": _wp(nodes[0], ego_s + rng.uniform(15.0, 30.0)), "init_speed_kmh": target, "target_speed_kmh": target, "behavior": "Move Forward Speeding", "trigger_time_s": 0.0}]
        params = {"speed_limit_kmh": limit, "target_speed_kmh": target, "subject": "npc", "violator_role": "violator"}
    elif violation_type in {"违规变道", "违规超车"}:
        if len(nodes) < 2:
            raise RandomSamplingUnsatisfied(f"{violation_type} 随机采样缺少 lane_pair")
        direction = str(chosen.get("lane_pair", {}).get("direction", spec.get("road_requirement", {}).get("lane_change_direction", "left")))
        ego["role"] = "priority"
        ego["spawn_waypoint"] = _wp(nodes[1], ego_s + rng.uniform(3.0, 10.0))
        behavior = f"Overtake {direction.title()}" if violation_type == "违规超车" else f"Change Lane {direction.title()}"
        npc_speed = rng.uniform(42.0, 60.0)
        npc = {"id": "violator", "role": "violator", "spawn_waypoint": _wp(nodes[0], ego_s), "init_speed_kmh": round(npc_speed, 1), "behavior": behavior, "trigger_time_s": trigger, "target_lane_change": direction, "source_lane_id": nodes[0]["lane_id"], "target_lane_id": nodes[1]["lane_id"], "longitudinal_gap_m": round(rng.uniform(4.0, 10.0), 1)}
        if violation_type == "违规超车":
            npc["target_speed_kmh"] = round(npc_speed + rng.uniform(12.0, 25.0), 1)
        npcs = [npc]
        params = {"lane_change_direction": direction, "source_lane_id": nodes[0]["lane_id"], "target_lane_id": nodes[1]["lane_id"], "min_lateral_shift_m": 1.8, "danger_gap_m": 10.0 if violation_type == "违规超车" else 8.0}
    elif violation_type in {"未按规定让行", "逆行"}:
        if len(nodes) < 2:
            raise RandomSamplingUnsatisfied(f"{violation_type} 随机采样至少需要两个同 Town 节点")
        second = rng.choice(nodes[1:])
        ego["role"] = "priority"
        ego["spawn_waypoint"] = _wp(second, rng.uniform(6.0, 18.0))
        npc = {"id": "violator", "role": "violator", "spawn_waypoint": _wp(nodes[0], rng.uniform(6.0, 20.0)), "init_speed_kmh": round(rng.uniform(30.0, 48.0), 1), "behavior": "Move Forward", "trigger_time_s": trigger}
        if violation_type == "逆行":
            npc["reverse_spawn_heading"] = True
            params = {"danger_distance_m": 12.5, "heading_opposition_deg": 120.0, "requires_ego_forward_wrong_lane": True}
        else:
            npc["behavior"] = "Turn Left"
            params = {"time_gap_range_s": spec.get("conflict", {}).get("timing", {}).get("time_gap_to_conflict_s", [0.5, 2.0])}
        npcs = [npc]
    else:
        raise RandomSamplingUnsatisfied(f"不支持的随机配置类型: {violation_type}")
    detectors = {
        "未保持安全距离": "following_distance_detector", "未按规定让行": "yield_violation_detector",
        "未注意前方路况": "inattention_front_condition_detector", "违规变道": "lane_change_violation_detector",
        "违规超车": "overtake_violation_detector", "逆行": "wrong_way_detector", "超速": "npc_speeding_violation_detector",
    }
    config = ScenarioConfiguration(scenario_id, violation_type, top_town, ego, npcs, conflict, {"weather": "clear", "time": "day"}, {"type": violation_type, "detector": detectors[violation_type], "params": params})
    diagnostics = {"mode": "random_sampling", "seed": seed, "candidate_count": len(candidates), "selected_map": top_town, "selected_match_type": chosen.get("match_type"), "sampled": {"ego_s": round(ego_s, 2), "trigger_time_s": round(trigger, 3)}}
    return config, diagnostics

def method_failure_type(exc: Exception, health_before: Dict[str, Any], health_after: Dict[str, Any]) -> str:
    if not health_before.get("healthy", False):
        return "simulator_unhealthy_before"
    if not health_after.get("healthy", False):
        return "simulator_unhealthy_after"
    if isinstance(exc, RandomSamplingUnsatisfied):
        return "random_sampling_unsatisfied"
    text = str(exc)
    if "effective use_llm=False" in text or "LLM expansion" in text or "DeepSeek" in text:
        return "llm_expansion_failed"
    classified = classify_failure(1, repr(exc), health_before, health_after)
    return classified if classified != "unknown_failure" else "method_failure"

def run_one(row: Dict[str, Any], variant: str, output_root: Path, base: Path, timeout_s: float, global_seed: int, prepare_only: bool) -> Dict[str, Any]:
    source_path = str(row.get("source_path", ""))
    source = load_json(Path(source_path))
    scenario_id = str(row["scenario_id"])
    variant_dir = output_root / variant
    result_dir = variant_dir / ("preparation_results" if prepare_only else "case_results")
    result_path = result_dir / f"{scenario_id}.json"
    if result_path.exists():
        prior = load_json(result_path)
        summary_row = dict(prior.get("summary_row", {}))
        failure_type = summary_row.get("failure_type")
        if not prepare_only and failure_type in ENVIRONMENT_FAILURES:
            remove_invalid_environment_result(summary_row, variant_dir)
        elif not prepare_only and failure_type == "llm_expansion_failed":
            target = variant_dir / "recoverable_failure_attempts" / result_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            result_path.replace(target)
        else:
            summary_row["_result_cached"] = True
            return summary_row
    health_before = {"healthy": True, "reason": "prepare_only"} if prepare_only else carla_health_check(timeout_s=5.0)
    stage_timing_s: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {"variant": variant}
    row_metadata = manifest_metadata(row)
    case_seed = int(row.get("seed", global_seed))
    spec: Dict[str, Any] = {}
    retrieval: Dict[str, Any] = {}
    config: Optional[ScenarioConfiguration] = None
    validation: Dict[str, Any] = {}
    try:
        if not prepare_only and not health_before.get("healthy", False):
            raise RuntimeError("CARLA simulator unhealthy before case")
        start = time.perf_counter()
        spec, diagnostics["expansion"] = expansion_for(source_path, source, variant, output_root, scenario_id=scenario_id, source_sha256=row.get("source_sha256", ""))
        stage_timing_s["expansion"] = round(time.perf_counter() - start, 6)
        start = time.perf_counter()
        retrieval, diagnostics["retrieval"] = retrieval_for(spec, source_path, variant, output_root, base)
        stage_timing_s["retrieval"] = round(time.perf_counter() - start, 6)
        start = time.perf_counter()
        if variant == "without_constraint":
            config, diagnostics["randomization"] = random_configuration(spec, retrieval, scenario_id, case_seed)
        else:
            solver_retrieval = retrieval_to_solver_result(retrieval, str(row.get("source_violation_type", "")))
            config = ConstraintSolver().solve(spec, [solver_retrieval])
            config.scenario_id = scenario_id
        stage_timing_s["configuration"] = round(time.perf_counter() - start, 6)
        start = time.perf_counter()
        validation = ConstraintValidator().validate(config.to_dict())
        stage_timing_s["constraint_validation"] = round(time.perf_counter() - start, 6)
        if prepare_only:
            summary_row = {
                "variant": variant, "scenario_id": scenario_id, "source_path": source_path,
                "source_violation_type": row.get("source_violation_type", ""), "violation_type": spec["violation_type"],
                "status": "prepared", "detected": False, "constraint_satisfied": validation.get("satisfied"),
                "constraint_satisfaction_rate": validation.get("satisfaction_rate"), "seed": case_seed,
                "stage_timing_s": stage_timing_s, "total_generation_time_s": round(sum(stage_timing_s.values()), 6),
                "health_before": health_before, "health_after": health_before, "output_path": str(result_path),
                **row_metadata,
            }
            dump_json({"source": source, "source_path": source_path, "manifest_row": row, "scenario_spec": spec, "retrieval": retrieval, "scenario_config": config.to_dict(), "constraint_validation": validation, "stage_timing_s": stage_timing_s, "diagnostics": diagnostics, "summary_row": summary_row}, result_path)
            return summary_row
        start = time.perf_counter()
        trace = CarlaScenarioRunner(timeout_s=timeout_s).run(config)
        stage_timing_s["execution"] = round(time.perf_counter() - start, 6)
        if not trace.ticks:
            raise RuntimeError("trace empty: CARLA runner 返回空执行轨迹")
        start = time.perf_counter()
        violation = detect_violation(config.violation_type, trace.ticks, config.expected_violation.get("params", {}))
        stage_timing_s["violation_detection"] = round(time.perf_counter() - start, 6)
        health_after = carla_health_check(timeout_s=5.0)
        if not health_after.get("healthy", False):
            raise RuntimeError("CARLA simulator unhealthy after case")
        payload = {
            "source": source, "source_path": source_path, "manifest_row": row, "scenario_spec": spec, "retrieval": retrieval,
            "scenario_config": config.to_dict(), "execution_trace": trace.to_dict(), "violation_result": violation,
            "constraint_validation": validation, "stage_timing_s": stage_timing_s, "diagnostics": diagnostics,
            "health_before": health_before, "health_after": health_after,
        }
        start = time.perf_counter()
        match = ScenarioMatchEvaluator(base).evaluate_result(payload, source)
        stage_timing_s["scene_match_evaluation"] = round(time.perf_counter() - start, 6)
        payload["scene_match_evaluation"] = match
        payload["stage_timing_s"] = stage_timing_s
        summary_row = {
            "variant": variant, "scenario_id": scenario_id, "source_path": source_path,
            "source_violation_type": row.get("source_violation_type", ""), "violation_type": config.violation_type,
            "status": "ok", "detected": bool(violation.get("detected")), "constraint_satisfied": validation.get("satisfied"),
            "constraint_satisfaction_rate": validation.get("satisfaction_rate"), "match_score": match.get("match_score"),
            "grade": match.get("grade"), "map": config.map_name, "seed": case_seed,
            "stage_timing_s": stage_timing_s, "total_generation_time_s": round(sum(stage_timing_s[stage] for stage in ("expansion", "retrieval", "configuration")), 6),
            "health_before": health_before, "health_after": health_after, "output_path": str(result_path),
            **row_metadata,
        }
        payload["summary_row"] = summary_row
        dump_json(payload, result_path)
        return summary_row
    except Exception as exc:
        health_after = health_before if prepare_only else carla_health_check(timeout_s=5.0)
        failure_type = method_failure_type(exc, health_before, health_after)
        summary_row = {
            "variant": variant, "scenario_id": scenario_id, "source_path": source_path,
            "source_violation_type": row.get("source_violation_type", ""), "violation_type": spec.get("violation_type", normalize_type(str(row.get("source_violation_type", "")))),
            "status": "failed", "detected": False, "constraint_satisfied": validation.get("satisfied", False),
            "constraint_satisfaction_rate": validation.get("satisfaction_rate", 0.0), "failure_type": failure_type,
            "recoverable": failure_type == "llm_expansion_failed", "error": repr(exc), "seed": case_seed,
            "stage_timing_s": stage_timing_s, "total_generation_time_s": round(sum(stage_timing_s.get(stage, 0.0) for stage in ("expansion", "retrieval", "configuration")), 6),
            "health_before": health_before, "health_after": health_after, "output_path": str(result_path),
            **row_metadata,
        }
        dump_json({"source": source, "source_path": source_path, "manifest_row": row, "scenario_spec": spec, "retrieval": retrieval, "scenario_config": config.to_dict() if config else None, "execution_trace": None, "violation_result": None, "scene_match_evaluation": None, "constraint_validation": validation, "stage_timing_s": stage_timing_s, "diagnostics": diagnostics, "summary_row": summary_row}, result_path)
        return summary_row

def summarize_variant(rows: Sequence[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    formal = [row for row in rows if row.get("failure_type") not in ENVIRONMENT_FAILURES and row.get("status") != "prepared"]
    executed_rows = [row for row in formal if row.get("status") == "ok"]
    failed_rows = [row for row in formal if row.get("status") == "failed"]
    detected = sum(bool(row.get("detected")) for row in executed_rows)
    satisfied = sum(bool(row.get("constraint_satisfied")) for row in executed_rows)
    stage_averages: Dict[str, float] = {}
    for stage in STAGE_NAMES:
        values = [float(row.get("stage_timing_s", {}).get(stage)) for row in formal if row.get("stage_timing_s", {}).get(stage) is not None]
        stage_averages[stage] = round(sum(values) / len(values), 6) if values else 0.0
    generation_times = [float(row.get("total_generation_time_s", 0.0)) for row in formal if row.get("timing_comparable", True)]
    match_scores = [float(row.get("match_score", 0.0) or 0.0) for row in executed_rows]
    high_grade = sum(row.get("grade") == "high" for row in executed_rows)
    total = len(formal)
    executed = len(executed_rows)
    failed = len(failed_rows)
    return {
        "variant": variant, "total": total, "executed": executed, "failed": failed,
        "execution_rate": round(executed / total, 6) if total else 0.0,
        "failure_rate": round(failed / total, 6) if total else 0.0,
        "detected": detected,
        "behavior_reproduction_rate": round(detected / executed, 6) if executed else 0.0,
        "behavior_reproduction_rate_all": round(detected / total, 6) if total else 0.0,
        "constraint_satisfied": satisfied, "constraint_satisfaction_rate": round(satisfied / executed, 6) if executed else 0.0,
        "avg_match_score": round(sum(match_scores) / len(match_scores), 6) if match_scores else 0.0,
        "high_grade": high_grade, "high_grade_rate": round(high_grade / executed, 6) if executed else 0.0,
        "avg_total_generation_time_s": round(sum(generation_times) / len(generation_times), 6) if generation_times else 0.0,
        **{f"avg_{stage}_time_s": value for stage, value in stage_averages.items()},
        "failure_types": dict(Counter(row.get("failure_type", "method_failure") for row in failed_rows)),
        "results": list(formal),
    }

def load_formal_rows(variant_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((variant_dir / "case_results").glob("*.json")):
        row = load_json(path).get("summary_row")
        if row and row.get("failure_type") not in ENVIRONMENT_FAILURES:
            rows.append(row)
    return rows

def import_rq1_full_baseline(output_root: Path, rq1_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    source_results: Dict[str, Dict[str, Any]] = {}
    for path in (rq1_dir / "case_results").glob("*.json"):
        payload = load_json(path)
        row = dict(payload.get("summary_row") or {})
        source_path = str(row.get("source_path") or payload.get("source_path") or "")
        if source_path:
            source_results[source_path] = row

    full_dir = output_root / "full"
    imported = 0
    missing: List[str] = []
    for item in manifest["rows"]:
        source_path = str(item["source_path"])
        source_row = source_results.get(source_path)
        if source_row is None:
            missing.append(source_path)
            continue
        scenario_id = item["scenario_id"]
        summary_row = {
            **source_row,
            "variant": "full",
            "scenario_id": scenario_id,
            "baseline_source": "rq1_carla_fixed_full_execution",
            "rq1_scenario_id": source_row.get("scenario_id"),
            "timing_comparable": False,
            "output_path": str(full_dir / "case_results" / f"{scenario_id}.json"),
        }
        dump_json({"summary_row": summary_row, "rq1_result_path": source_row.get("output_path")}, Path(summary_row["output_path"]))
        imported += 1
    if missing:
        raise RuntimeError(f"RQ1 Full 基线缺少 {len(missing)} 个配对场景，示例: {missing[:3]}")
    summary = write_variant_outputs(output_root, "full")
    dump_json({"source_dir": str(rq1_dir), "imported": imported, "missing": missing, "summary": summary}, full_dir / "rq1_import_meta.json")
    return {"imported": imported, "missing": len(missing)}

def write_variant_outputs(output_root: Path, variant: str, rows: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    variant_dir = output_root / variant
    formal_rows = list(rows) if rows is not None else load_formal_rows(variant_dir)
    summary = summarize_variant(formal_rows, variant)
    dump_json(summary, variant_dir / "summary.json")
    write_csv(variant_dir / "execution_rows.csv", formal_rows)
    return summary

def wilcoxon_rows(output_root: Path) -> List[Dict[str, Any]]:
    variant_rows = {variant: {row["source_path"]: row for row in load_formal_rows(output_root / variant)} for variant in VARIANTS}
    results: List[Dict[str, Any]] = []
    metrics = ("executed", "detected", "constraint_satisfied", "match_score", "total_generation_time_s")
    for ablation in VARIANTS[1:]:
        shared = sorted(set(variant_rows["full"]) & set(variant_rows[ablation]))
        for metric in metrics:
            full_values: List[float] = []
            ablation_values: List[float] = []
            for source_path in shared:
                left = variant_rows["full"][source_path]
                right = variant_rows[ablation][source_path]
                if metric == "executed":
                    full_values.append(float(left.get("status") == "ok"))
                    ablation_values.append(float(right.get("status") == "ok"))
                elif metric in {"detected", "constraint_satisfied"}:
                    full_values.append(float(bool(left.get(metric, False))))
                    ablation_values.append(float(bool(right.get(metric, False))))
                else:
                    full_values.append(float(left.get(metric, 0.0) or 0.0))
                    ablation_values.append(float(right.get(metric, 0.0) or 0.0))
            row: Dict[str, Any] = {"comparison": f"full_vs_{ablation}", "metric": metric, "paired_n": len(shared)}
            differences = [left - right for left, right in zip(full_values, ablation_values)]
            if not shared:
                row.update({"status": "unavailable", "reason": "no_paired_source_path", "statistic": "", "p_value": ""})
            elif all(abs(value) <= 1e-12 for value in differences):
                row.update({"status": "identical", "reason": "no_difference", "statistic": 0.0, "p_value": 1.0})
            else:
                try:
                    from scipy.stats import wilcoxon
                    test = wilcoxon(full_values, ablation_values)
                    row.update({"status": "ok", "reason": "", "statistic": float(test.statistic), "p_value": float(test.pvalue)})
                except ImportError:
                    row.update({"status": "unavailable", "reason": "scipy_not_installed", "statistic": "", "p_value": ""})
                except ValueError as exc:
                    row.update({"status": "unavailable", "reason": str(exc), "statistic": "", "p_value": ""})
            results.append(row)
    return results

def write_root_outputs(output_root: Path) -> None:
    summaries = []
    for variant in VARIANTS:
        if (output_root / variant / "case_results").exists():
            summaries.append(write_variant_outputs(output_root, variant))
    columns = ["variant", "total", "executed", "failed", "execution_rate", "failure_rate", "detected", "behavior_reproduction_rate", "behavior_reproduction_rate_all", "constraint_satisfaction_rate", "avg_match_score", "high_grade_rate", "avg_total_generation_time_s"] + [f"avg_{stage}_time_s" for stage in STAGE_NAMES]
    write_csv(output_root / "rq2_ablation.csv", summaries, columns)
    write_csv(output_root / "rq2_wilcoxon.csv", wilcoxon_rows(output_root), ["comparison", "metric", "paired_n", "status", "reason", "statistic", "p_value"])
    if summaries:
        try:
            import matplotlib.pyplot as plt
            labels = [item["variant"] for item in summaries]
            x_values = range(len(labels))
            width = 0.25
            plt.figure(figsize=(11, 6))
            plt.bar([item - width for item in x_values], [row["execution_rate"] for row in summaries], width, label="execution")
            plt.bar(list(x_values), [row["behavior_reproduction_rate"] for row in summaries], width, label="detected")
            plt.bar([item + width for item in x_values], [row["constraint_satisfaction_rate"] for row in summaries], width, label="constraint")
            plt.xticks(list(x_values), labels, rotation=20)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_root / "rq2_ablation_rates.png", dpi=180)
            plt.close()
        except ImportError:
            pass

def parse_variants(values: Sequence[str], parser: argparse.ArgumentParser) -> List[str]:
    if not values or "all" in values:
        return list(VARIANTS)
    invalid = [value for value in values if value not in VARIANTS]
    if invalid:
        parser.error(f"未知 variant: {invalid}")
    return list(dict.fromkeys(values))

def child_command(script: Path, variant: str, case_index: int, args: argparse.Namespace, output_root: Path) -> List[str]:
    command = [sys.executable, str(script), "--variant", variant, "--case-index", str(case_index), "--offset", str(args.offset), "--timeout-s", str(args.timeout_s), "--output-dir", str(output_root), "--seed", str(args.seed), "--manifest", str(args.manifest)]
    if args.cases_per_type is not None:
        command.extend(["--cases-per-type", str(args.cases_per_type)])
    if args.max_cases is not None:
        command.extend(["--max-cases", str(args.max_cases)])
    if args.prepare_only:
        command.append("--prepare-only")
    if args.rq1_full_dir:
        command.extend(["--rq1-full-dir", str(args.rq1_full_dir)])
    return command

def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal RQ2 ablation experiments with isolated per-case execution.")
    parser.add_argument("--variant", action="append", default=[], help="Repeat one of full/without_community/without_expansion/without_constraint/without_semantic_summaries, or use all.")
    parser.add_argument("--cases-per-type", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output-dir", default="RKGRScen/data/evaluation/rq2_ablation")
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manifest", default=DEFAULT_SHARED_MANIFEST)
    parser.add_argument("--rq1-full-dir", default=None, help="Import the paired Full baseline from an existing RQ1 result directory.")
    args = parser.parse_args()
    if args.cases_per_type is not None and args.cases_per_type < 0:
        parser.error("--cases-per-type 必须是非负整数")
    variants = parse_variants(args.variant, parser)
    if args.case_index is not None and len(variants) != 1:
        parser.error("--case-index 必须且只能指定一个 variant")
    base = Path(__file__).resolve().parents[2]
    retrieval_path = base / "RKGRScen" / "data" / "retrieval" / "p0_graphrag_retrieval_full" / "retrieval_results.json"
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = base / manifest_path
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = base / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = load_shared_manifest(manifest_path, base)
    rows_to_run = enrich_manifest_rows(
        select_manifest_rows(source_manifest, args.max_cases, args.offset, list(RQ2_SOURCE_TYPES), args.cases_per_type),
        retrieval_index(retrieval_path),
    )
    manifest = {**source_manifest, "manifest_path": str(manifest_path), "retrieval_path": str(retrieval_path), "selected_total": len(rows_to_run), "source_path_order": [row["source_path"] for row in rows_to_run], "rows": rows_to_run}
    dump_json(manifest, output_root / "manifest.json")
    selected_meta = {"manifest_path": str(manifest_path), "retrieval_path": str(retrieval_path), "selected_total": len(rows_to_run), "cases_per_type": args.cases_per_type, "max_cases": args.max_cases, "offset": args.offset, "variants": variants, "prepare_only": args.prepare_only, "type_counts": dict(Counter(row["source_violation_type"] for row in rows_to_run))}
    dump_json(selected_meta, output_root / "selected_input_meta.json")
    for variant in variants:
        dump_json({**selected_meta, "variant": variant, "source_path_order": manifest["source_path_order"]}, output_root / variant / "selected_input_meta.json")
    if args.rq1_full_dir and args.case_index is None:
        rq1_dir = Path(args.rq1_full_dir)
        if not rq1_dir.is_absolute():
            rq1_dir = base / rq1_dir
        imported = import_rq1_full_baseline(output_root, rq1_dir, manifest)
        print(json.dumps({"baseline": "full", **imported}, ensure_ascii=False), flush=True)
    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(rows_to_run):
            raise SystemExit(f"case-index 越界: {args.case_index}, total={len(rows_to_run)}")
        result = run_one(rows_to_run[args.case_index], variants[0], output_root, base, args.timeout_s, args.seed, args.prepare_only)
        if not args.prepare_only and not result.get("_result_cached", False):
            remove_invalid_environment_result(result, output_root / variants[0])
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    for variant in variants:
        if variant == "full" and args.rq1_full_dir:
            continue
        rows: List[Dict[str, Any]] = []
        for index, source_row in enumerate(rows_to_run):
            completed = subprocess.run(child_command(Path(__file__).resolve(), variant, index, args, output_root), cwd=str(base), capture_output=True, text=True)
            json_lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
            if completed.returncode == 0 and json_lines:
                result = json.loads(json_lines[-1])
            else:
                source_path = str(source_row.get("source_path", ""))
                scenario_id = str(source_row["scenario_id"])
                health_before = {"healthy": True, "reason": "prepare_only"} if args.prepare_only else carla_health_check(timeout_s=5.0)
                health_after = health_before if args.prepare_only else carla_health_check(timeout_s=5.0)
                result = {"variant": variant, "scenario_id": scenario_id, "source_path": source_path, "source_violation_type": source_row.get("source_violation_type", ""), "status": "failed", "detected": False, "failure_type": classify_failure(completed.returncode, completed.stdout + completed.stderr, health_before, health_after), "health_before": health_before, "health_after": health_after, "error": (completed.stdout + completed.stderr)[-4000:], **manifest_metadata(source_row)}
                target = output_root / variant / ("preparation_results" if args.prepare_only else "case_results") / f"{scenario_id}.json"
                result["output_path"] = str(target)
                dump_json({"summary_row": result}, target)
            cached = result.pop("_result_cached", False)
            if not args.prepare_only and not cached and result.get("failure_type") in ENVIRONMENT_FAILURES:
                remove_invalid_environment_result(result, output_root / variant)
            else:
                rows.append(result)
            output_name = "preparation_rows.csv" if args.prepare_only else "execution_rows.csv"
            write_csv(output_root / variant / output_name, rows)
            if not args.prepare_only:
                dump_json(summarize_variant(rows, variant), output_root / variant / "summary_partial.json")
            print(json.dumps({"progress": f"{index + 1}/{len(rows_to_run)}", **result}, ensure_ascii=False), flush=True)
        if args.prepare_only:
            prepared = sum(row.get("status") == "prepared" for row in rows)
            dump_json({"variant": variant, "selected_total": len(rows_to_run), "prepared": prepared, "failed": len(rows) - prepared, "results": rows}, output_root / variant / "preparation_summary.json")
        else:
            write_variant_outputs(output_root, variant, rows)
        write_root_outputs(output_root)
    if not args.prepare_only:
        write_root_outputs(output_root)

if __name__ == "__main__":
    main()
