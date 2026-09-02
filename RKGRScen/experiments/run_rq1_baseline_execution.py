import argparse
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from RKGRScen.experiments import run_rq1_carla_full_execution as full

_DEBUG_ENV_PATH = Path(".dbg/arise-grounding-failures.env")

def _debug_report(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        url = "http://127.0.0.1:7777/event"
        session_id = "arise-grounding-failures"
        if _DEBUG_ENV_PATH.is_file():
            config = _DEBUG_ENV_PATH.read_text(encoding="utf-8").splitlines()
            values = {
                line.split("=", 1)[0]: line.split("=", 1)[1]
                for line in config
                if "=" in line
            }
            url = values.get("DEBUG_SERVER_URL", url)
            session_id = values.get("DEBUG_SESSION_ID", session_id)
        payload = json.dumps({
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {message}",
            "data": data,
        }).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=0.25).close()
    except Exception:
        pass

BASELINE_METHODS = ("template_mapping", "arise_derived")
DEFAULT_SHARED_MANIFEST = full.DEFAULT_SHARED_MANIFEST
SUPPORTED_MAP_NAMES = {f"Town0{index}" for index in range(1, 6)}
TEMPLATE_MAP_BY_VIOLATION = {
    "未按规定让行": "Town05",
    "未保持安全距离": "Town02",
    "未注意前方路况": "Town02",
    "违规变道": "Town04",
    "违规超车": "Town04",
    "逆行": "Town02",
    "超速": "Town02",
    "闯红灯": "Town05",
}

def _normalize_map_name(value: Any, path: Path) -> str:
    candidates = (str(value or ""), path.stem)
    for candidate in candidates:
        match = re.search(r"Town0[1-5]", candidate)
        if match:
            return match.group(0)
    raise ValueError(f"无法从地图图文件解析合法地图名: {path}")

def _load_map_nodes(base: Path) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    map_dir = base / "RKGRScen" / "data" / "maps"
    for path in sorted(map_dir.glob("Carla_Maps_Town*_graph.json")):
        payload = full.load_json(path)
        if not isinstance(payload, dict):
            continue
        graph = payload.get("graph") if isinstance(payload.get("graph"), dict) else {}
        map_name = _normalize_map_name(
            payload.get("map_name") or payload.get("name") or graph.get("map_name"),
            path,
        )
        if map_name not in SUPPORTED_MAP_NAMES:
            raise ValueError(f"Template mapping 不支持地图 {map_name}: {path}")
        raw_nodes = payload.get("nodes") or graph.get("nodes") or []
        for node in raw_nodes:
            if isinstance(node, dict):
                item = dict(node)
                item["map_name"] = map_name
                nodes.append(item)
    if not nodes:
        raise RuntimeError(f"Template mapping 未从 {map_dir} 加载到地图节点")
    return nodes

def _node_score(node: Dict[str, Any], violation_type: str, fine: bool) -> int:
    score = 0
    road_type = str(node.get("road_type") or node.get("type") or "").lower()
    is_junction = bool(node.get("is_junction")) or "intersection" in road_type or "junction" in road_type
    if is_junction or node.get("has_traffic_light"):
        if violation_type in {"未按规定让行", "闯红灯"}:
            score += 8
    if int(node.get("lane_count", 1) or 1) >= 2:
        if violation_type in {"违规变道", "违规超车", "超速"}:
            score += 5
    if not node.get("is_junction") and violation_type in {"未保持安全距离", "未注意前方路况", "超速"}:
        score += 4
    if fine:
        score += int(node.get("lane_count", 1) or 1)
        if node.get("has_traffic_light"):
            score += 1
    return score

def _candidate_nodes(nodes: List[Dict[str, Any]], violation_type: str, fine: bool) -> List[Dict[str, Any]]:
    target_map = TEMPLATE_MAP_BY_VIOLATION.get(violation_type, "Town02")
    pool = [node for node in nodes if node.get("map_name") == target_map]
    if not pool:
        raise RuntimeError(f"Template mapping 地图 {target_map} 没有候选节点")
    ranked = sorted(
        enumerate(pool),
        key=lambda item: (-_node_score(item[1], violation_type, fine), item[0]),
    )
    first = ranked[0][1]
    selected = [first]
    if violation_type in {"未按规定让行", "闯红灯", "违规变道", "违规超车", "逆行"}:
        if violation_type in {"违规变道", "违规超车"}:
            pair_pool = [
                node for node in pool
                if node is not first
                and node.get("road_id") == first.get("road_id")
                and int(node.get("lane_id", 0)) * int(first.get("lane_id", 0)) > 0
                and abs(int(node.get("lane_id", 0)) - int(first.get("lane_id", 0))) == 1
            ]
        else:
            pair_pool = [
                node for node in pool
                if node is not first
                and int(node.get("lane_id", 0)) != int(first.get("lane_id", 0))
                and (
                    node.get("road_id") != first.get("road_id")
                    or int(node.get("lane_id", 0)) * int(first.get("lane_id", 0)) < 0
                )
            ]
        if not pair_pool:
            raise RuntimeError(f"Template mapping 无法为 {violation_type} 找到第二个粗粒度候选车道")
        second = max(pair_pool, key=lambda node: _node_score(node, violation_type, fine))
        selected.append(second)
    return selected

def template_retrieval(row: Dict[str, Any], base: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    nodes = cache.setdefault("nodes", _load_map_nodes(base))
    violation_type = full.normalize_type(str(row.get("source_violation_type", "")))
    fine = row.get("granularity") == "fine"
    selected_nodes = _candidate_nodes(nodes, violation_type, fine)
    fallback = row.get("retrieval", {}).get("local_top_k", [{}])[0]
    local_top_k = []
    for selected in selected_nodes:
        node = full.normalize_node(selected, fallback)
        map_name = str(selected.get("map_name") or fallback.get("map_name") or "")
        if map_name.startswith("Carla/Maps/"):
            map_name = map_name.rsplit("/", 1)[-1]
        if map_name not in SUPPORTED_MAP_NAMES:
            raise ValueError(f"Template mapping 产生非法地图名: {map_name!r}")
        local_top_k.append({**node, "map_name": map_name, "score": 0.72, "match_type": "template_mapping"})
    return {
        "local_top_k": local_top_k,
        "scenario_spec": dict(row.get("scenario_spec") or row.get("retrieval", {}).get("scenario_spec") or {}),
        "baseline_diagnostics": {
            "method": "template_mapping",
            "selection": "fixed_map_first_coarse_match",
            "template_map": TEMPLATE_MAP_BY_VIOLATION.get(violation_type, "Town02"),
        },
    }

def _tokenize_text(text: str) -> List[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = ["".join(chinese[index:index + 2]) for index in range(max(0, len(chinese) - 1))]
    return latin + chinese + bigrams

def _text_vector(text: str) -> Counter:
    return Counter(_tokenize_text(text))

def _cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / max(left_norm * right_norm, 1e-9)

def _load_arise_fragment_library(base: Path) -> Dict[str, List[Dict[str, Any]]]:
    db_path = base / "ARISE-main" / "ARISE" / "db" / "database_v1_scenic3.json"
    payload = full.load_json(db_path)
    library: Dict[str, List[Dict[str, Any]]] = {}
    for section, section_payload in payload.items():
        descriptions = section_payload.get("description", []) if isinstance(section_payload, dict) else []
        snippets = section_payload.get("snippet", []) if isinstance(section_payload, dict) else []
        rows = []
        for index, description in enumerate(descriptions):
            snippet = snippets[index] if index < len(snippets) else ""
            text = f"{description}\n{snippet}"
            rows.append({
                "section": section,
                "index": index,
                "description": description,
                "snippet": snippet,
                "vector": _text_vector(text),
            })
        library[section] = rows
    return library

def _logical_to_nl(row: Dict[str, Any]) -> str:
    spec = dict(row.get("scenario_spec") or row.get("retrieval", {}).get("scenario_spec") or {})
    violation_type = full.normalize_type(str(row.get("source_violation_type", spec.get("violation_type", ""))))
    road_requirement = spec.get("road_requirement", {})
    conflict = spec.get("conflict", {})
    actors = spec.get("actors", [])
    actor_text = "; ".join(
        f"{actor.get('role', '')} {actor.get('action', '')} path {actor.get('path', '')} speed {actor.get('speed_kmh', '')}"
        for actor in actors
    )
    return " ".join([
        f"traffic violation {violation_type}",
        f"road {road_requirement.get('type', '')} lanes {road_requirement.get('min_lanes', '')}",
        f"conflict {conflict.get('type', '')} {conflict.get('location', '')} {conflict.get('trigger_condition', '')}",
        f"actors {actor_text}",
        str(spec.get("semantic_context", {}).get("reason", "")),
    ])

def _retrieve_arise_top1(library: Dict[str, List[Dict[str, Any]]], section: str, query: str) -> Dict[str, Any]:
    query_vector = _text_vector(query)
    candidates = library.get(section, [])
    if not candidates:
        raise RuntimeError(f"ARISE 片段库缺少 {section} section")
    ranked = sorted(
        candidates,
        key=lambda item: (-_cosine_similarity(query_vector, item["vector"]), item["index"]),
    )
    chosen = dict(ranked[0])
    chosen["score"] = round(_cosine_similarity(query_vector, chosen["vector"]), 6)
    chosen.pop("vector", None)
    return chosen

def _arise_map_from_geometry(geometry_snippet: str, violation_type: str) -> str:
    match = re.search(r"Town\s*=\s*['\"](Town0[1-5])['\"]", geometry_snippet)
    if match:
        return match.group(1)
    return TEMPLATE_MAP_BY_VIOLATION.get(violation_type, "Town02")

def _arise_geometry_intent(geometry: Dict[str, Any], spawn: Dict[str, Any], violation_type: str) -> Dict[str, Any]:
    text = f"{geometry.get('description', '')}\n{geometry.get('snippet', '')}\n{spawn.get('description', '')}\n{spawn.get('snippet', '')}".lower()
    needs_junction = any(token in text for token in ("intersection", "junction", "crossing")) or violation_type in {"未按规定让行"}
    needs_curve = "curve" in text or "curved" in text
    needs_straight = "straight" in text or violation_type in {"未保持安全距离", "未注意前方路况", "超速"}
    same_lane_front = any(token in text for token in ("same straight road", "same lane", "in front", "front of the ego"))
    adjacent_lane = any(token in text for token in ("adjacent lane", "target lane", "lane change", "parallel")) or violation_type in {"违规变道", "违规超车"}
    opposing = any(token in text for token in ("opposite", "oncoming", "opposing")) or violation_type == "逆行"
    return {
        "needs_junction": needs_junction,
        "needs_curve": needs_curve,
        "needs_straight": needs_straight,
        "same_lane_front": same_lane_front,
        "adjacent_lane": adjacent_lane,
        "opposing": opposing,
    }

def _node_length(node: Dict[str, Any]) -> float:
    start, end = node.get("start", {}), node.get("end", {})
    return math.hypot(float(end.get("x", 0.0)) - float(start.get("x", 0.0)), float(end.get("y", 0.0)) - float(start.get("y", 0.0)))

def _arise_node_score(node: Dict[str, Any], intent: Dict[str, Any], violation_type: str) -> float:
    road_type = str(node.get("road_type") or node.get("type") or "").lower()
    is_junction = bool(node.get("is_junction")) or "junction" in road_type or "intersection" in road_type
    lane_count = int(node.get("lane_count", 1) or 1)
    score = _node_length(node) * 0.02
    if intent["needs_junction"] and is_junction:
        score += 6.0
    if intent["needs_straight"] and not is_junction and "straight" in road_type:
        score += 6.0
    if intent["needs_curve"] and "curve" in road_type:
        score += 4.0
    if intent["adjacent_lane"] and lane_count >= 2:
        score += 5.0
    if violation_type in {"违规变道", "违规超车", "超速"} and lane_count >= 2:
        score += 2.0
    if violation_type == "未按规定让行" and node.get("has_traffic_light"):
        score += 1.5
    return score

def _arise_select_nodes(nodes: List[Dict[str, Any]], map_name: str, intent: Dict[str, Any], violation_type: str) -> List[Dict[str, Any]]:
    pool = [node for node in nodes if node.get("map_name") == map_name]
    if not pool:
        raise RuntimeError(f"ARISE-derived 地图 {map_name} 没有候选节点")
    if violation_type in {"违规变道", "违规超车"}:
        best_pair = None
        best_score = -1e9
        for left in pool:
            for right in pool:
                if left is right or left.get("road_id") != right.get("road_id"):
                    continue
                left_lane = int(left.get("lane_id", 0))
                right_lane = int(right.get("lane_id", 0))
                if left_lane == 0 or right_lane == 0 or left_lane * right_lane <= 0 or abs(left_lane - right_lane) != 1:
                    continue
                score = _arise_node_score(left, intent, violation_type) + _arise_node_score(right, intent, violation_type)
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            raise RuntimeError(f"ARISE-derived 无法为 {violation_type} 找到同向相邻车道对")
        return [best_pair[0], best_pair[1]]

    ranked = sorted(enumerate(pool), key=lambda item: (-_arise_node_score(item[1], intent, violation_type), item[0]))
    first = ranked[0][1]
    selected = [first]
    if violation_type in {"未按规定让行", "逆行"} or intent["opposing"]:
        pair_pool = []
        for node in pool:
            if node is first:
                continue
            same_road = node.get("road_id") == first.get("road_id")
            lane_product = int(node.get("lane_id", 0)) * int(first.get("lane_id", 0))
            if intent["opposing"] and (not same_road or lane_product < 0):
                pair_pool.append(node)
            elif violation_type == "未按规定让行" and int(node.get("lane_id", 0)) != int(first.get("lane_id", 0)):
                pair_pool.append(node)
        if not pair_pool:
            pair_pool = [node for node in pool if node is not first]
        if pair_pool:
            selected.append(max(pair_pool, key=lambda node: _arise_node_score(node, intent, violation_type)))
    return selected

def arise_retrieval(row: Dict[str, Any], base: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    library = cache.setdefault("arise_fragment_library", _load_arise_fragment_library(base))
    nodes = cache.setdefault("nodes", _load_map_nodes(base))
    query = _logical_to_nl(row)
    violation_type = full.normalize_type(str(row.get("source_violation_type", "")))
    retrieved = {
        section: _retrieve_arise_top1(library, section, query)
        for section in ("behavior", "geometry", "spawn", "misc", "weather")
    }
    map_name = _arise_map_from_geometry(str(retrieved["geometry"].get("snippet", "")), violation_type)
    intent = _arise_geometry_intent(retrieved["geometry"], retrieved["spawn"], violation_type)
    selected_nodes = _arise_select_nodes(nodes, map_name, intent, violation_type)
    fallback = row.get("retrieval", {}).get("local_top_k", [{}])[0]
    local_top_k = []
    for node_index, selected in enumerate(selected_nodes):
        normalized = full.normalize_node(selected, fallback)
        local_top_k.append({
            **normalized,
            "map_name": map_name,
            "score": float(retrieved["geometry"].get("score", 0.0)) + float(retrieved["spawn"].get("score", 0.0)),
            "match_type": "arise_flat_text_fragment",
            "arise_node_rank": node_index,
        })
    _debug_report(
        "B",
        "run_rq1_baseline_execution.py:arise_retrieval",
        "ARISE-derived flat text fragments selected",
        {
            "scenario_id": row.get("scenario_id"),
            "query_type": violation_type,
            "query": query[:500],
            "retrieved": {
                section: {
                    "index": item.get("index"),
                    "score": item.get("score"),
                    "description": str(item.get("description", ""))[:180],
                }
                for section, item in retrieved.items()
            },
            "map_name": map_name,
            "intent": intent,
            "local_top_k": [
                {"node_id": item.get("node_id"), "map_name": item.get("map_name"), "road_id": item.get("road_id"), "lane_id": item.get("lane_id")}
                for item in local_top_k
            ],
        },
    )
    return {
        "local_top_k": local_top_k,
        "scenario_spec": dict(row.get("scenario_spec") or row.get("retrieval", {}).get("scenario_spec") or {}),
        "baseline_diagnostics": {
            "method": "arise_derived",
            "selection": "arise_database_flat_text_retrieval",
            "embedding_model": "deterministic_token_cosine_equivalent",
            "arise_codebase": "ARISE-main",
            "arise_fragments": {
                section: {
                    "index": item.get("index"),
                    "score": item.get("score"),
                    "description": item.get("description"),
                    "snippet": item.get("snippet"),
                }
                for section, item in retrieved.items()
            },
            "geometry_intent": intent,
        },
    }

def baseline_row(row: Dict[str, Any], method: str, base: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(row)
    if method == "template_mapping":
        retrieval = template_retrieval(row, base, cache)
    elif method == "arise_derived":
        retrieval = arise_retrieval(row, base, cache)
    else:
        raise ValueError(f"未知 RQ1 baseline method: {method}")
    merged["retrieval"] = retrieval
    merged["scenario_spec"] = dict(retrieval.get("scenario_spec") or row.get("scenario_spec") or {})
    merged["baseline_method"] = method
    merged["llm_summary_used"] = False
    merged["llm_expansion_used"] = False
    _debug_report(
        "A",
        "run_rq1_baseline_execution.py:baseline_row",
        "ARISE-derived row adapted into shared runner",
        {
            "scenario_id": merged.get("scenario_id"),
            "method": method,
            "source_violation_type": merged.get("source_violation_type"),
            "retrieval_map": (retrieval.get("local_top_k") or [{}])[0].get("map_name"),
            "retrieval_node_count": len(retrieval.get("local_top_k", [])),
            "scenario_spec_actor_count": len(merged.get("scenario_spec", {}).get("actors", []) or []),
            "scenario_spec": merged.get("scenario_spec", {}),
        },
    )
    return merged

def load_rows(manifest_path: Path, base: Path, max_cases: Optional[int], offset: int, only_types: Optional[List[str]], cases_per_type: Optional[int]) -> List[Dict[str, Any]]:
    manifest = full.load_shared_manifest(manifest_path, base)
    retrieval_path = base / "RKGRScen" / "data" / "retrieval" / "p0_graphrag_retrieval_full" / "retrieval_results.json"
    rows = full.enrich_manifest_rows(
        full.manifest_rows(manifest, max_cases, offset, only_types, cases_per_type),
        full.retrieval_index(retrieval_path),
    )
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=BASELINE_METHODS, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--only-type", action="append", default=None)
    parser.add_argument("--cases-per-type", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--manifest", default=DEFAULT_SHARED_MANIFEST)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = base / manifest_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_to_run = load_rows(manifest_path, base, args.max_cases, args.offset, args.only_type, args.cases_per_type)
    selected_meta = {
        "method": args.method,
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "selected_total": len(rows_to_run),
        "max_cases": args.max_cases,
        "offset": args.offset,
        "only_type": args.only_type,
        "cases_per_type": args.cases_per_type,
        "prepare_only": args.prepare_only,
        "llm_summary_used": False,
        "llm_expansion_used": False,
    }
    full.dump_json(selected_meta, output_dir / "selected_input_meta.json")
    cache: Dict[str, Any] = {}

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(rows_to_run):
            raise SystemExit(f"case-index 越界: {args.case_index}, total={len(rows_to_run)}")
        try:
            row = baseline_row(rows_to_run[args.case_index], args.method, base, cache)
        except Exception as exc:
            raw = rows_to_run[args.case_index]
            result_path = output_dir / ("preparation_results" if args.prepare_only else "case_results") / f"{raw['scenario_id']}.json"
            result = {
                "scenario_id": str(raw["scenario_id"]),
                "source_path": raw.get("source_path", ""),
                "violation_type": full.normalize_type(str(raw.get("source_violation_type", ""))),
                "source_violation_type": raw.get("source_violation_type", ""),
                "status": "failed",
                "detected": False,
                "failure_type": "grounding_failed",
                "error": repr(exc),
                "output_path": str(result_path),
                **full.manifest_metadata(raw),
            }
            full.dump_json({"summary_row": result}, result_path)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return
        result = full.run_one(row, output_dir, base, args.timeout_s, args.prepare_only)
        result["method"] = args.method
        result["llm_summary_used"] = False
        result["llm_expansion_used"] = False
        if not args.prepare_only and not result.get("_result_cached", False):
            full.remove_invalid_environment_result(result, output_dir)
        if args.method == "arise_derived":
            _debug_report(
                "C",
                "run_rq1_baseline_execution.py:case-result",
                "ARISE-derived case completed",
                {
                    "scenario_id": result.get("scenario_id"),
                    "status": result.get("status"),
                    "failure_type": result.get("failure_type"),
                    "error": result.get("error"),
                    "health_before": result.get("health_before"),
                    "health_after": result.get("health_after"),
                    "map": result.get("map") or result.get("scenario_config", {}).get("map_name"),
                    "selected_node_count": len((result.get("retrieval_row", {}).get("retrieval", {}).get("local_top_k") or [])),
                    "actor_count": len((result.get("scenario_spec", {}).get("actors") or [])),
                },
            )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    rows: List[Dict[str, Any]] = []
    import subprocess, sys
    for idx, row in enumerate(rows_to_run, 1):
        scenario_id = str(row["scenario_id"])
        result_path = output_dir / ("preparation_results" if args.prepare_only else "case_results") / f"{scenario_id}.json"
        if result_path.exists():
            result = dict(full.load_json(result_path).get("summary_row", {}))
            result["_result_cached"] = True
        else:
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--method", args.method, "--case-index", str(idx - 1),
                "--offset", str(args.offset), "--timeout-s", str(args.timeout_s), "--output-dir", str(output_dir),
                "--manifest", str(manifest_path), *(["--prepare-only"] if args.prepare_only else []),
                *(["--max-cases", str(args.max_cases)] if args.max_cases is not None else []),
                *(["--cases-per-type", str(args.cases_per_type)] if args.cases_per_type is not None else []),
                *(sum((["--only-type", item] for item in (args.only_type or [])), [])),
            ], cwd=str(base), capture_output=True, text=True)
            if completed.returncode == 0:
                json_lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
                result = json.loads(json_lines[-1]) if json_lines else {"scenario_id": scenario_id, "status": "failed", "failure_type": "unknown_failure"}
            else:
                result = {"method": args.method, "scenario_id": scenario_id, "source_path": row.get("source_path", ""), "source_violation_type": row.get("source_violation_type", ""), "status": "failed", "detected": False, "failure_type": "child_process_failed", "error": (completed.stdout + completed.stderr)[-4000:], "output_path": str(result_path), **full.manifest_metadata(row)}
                full.dump_json({"summary_row": result}, result_path)
                full.remove_invalid_environment_result(result, output_dir)
        result["method"] = args.method
        result["llm_summary_used"] = False
        result["llm_expansion_used"] = False
        result.pop("_result_cached", None)
        rows.append(result)
        print(json.dumps({"progress": f"{idx}/{len(rows_to_run)}", **result}, ensure_ascii=False), flush=True)
        rows_name = "preparation_rows.csv" if args.prepare_only else "execution_rows.csv"
        full.write_csv(output_dir / rows_name, rows)
        full.dump_json(full.summarize_rows(rows), output_dir / "summary_partial.json")

    summary = full.summarize_rows(rows)
    summary["method"] = args.method
    summary["selected_input"] = selected_meta
    if args.prepare_only:
        full.dump_json(summary, output_dir / "preparation_summary.json")
        full.write_csv(output_dir / "preparation_rows.csv", rows)
    else:
        full.dump_json(summary, output_dir / "summary.json")
        full.write_csv(output_dir / "execution_rows.csv", rows)
        full.plot_execution_by_type(rows, output_dir / "charts" / "execution_status_by_type.png")
        full.plot_detected_by_type(rows, output_dir / "charts" / "detection_by_type.png")
        full.build_report(rows, summary, output_dir, len(rows_to_run))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
