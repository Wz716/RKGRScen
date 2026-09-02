import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from RKGRScen.experiments.batch_utils import carla_health_check, classify_failure, dump_json, summarize_rows

from RKGRScen.execution.violation_detector import detect_violation
from RKGRScen.models import CommunityRecord, RetrievalResult
from RKGRScen.query.constraint_solver import ConstraintSolver
from RKGRScen.query.constraint_validator import ConstraintValidator
from RKGRScen.query.scenario_match_evaluator import ScenarioMatchEvaluator

SUPPORTED_TYPES = {
    "未保持安全距离",
    "未按规定让行",
    "未注意前方路况",
    "违规变道",
    "违规超车",
    "逆行",
    "超速行驶",
    "超速",
    "闯红灯",
}

DEFAULT_SHARED_MANIFEST = "RKGRScen/data/evaluation/rq1_rq2_20260714_shared/manifest.json"

TYPE_NORMALIZATION = {
    "超速行驶": "超速",
}

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_shared_manifest(path: Path, base: Path) -> Dict[str, Any]:
    manifest = load_json(path)
    rows = manifest.get("rows", [])
    if not rows:
        raise ValueError(f"共享 manifest 没有 rows: {path}")
    labels_path = path.parent / "granularity_labels.csv"
    labels = {}
    if labels_path.exists():
        with labels_path.open(encoding="utf-8", newline="") as handle:
            labels = {item["scenario_id"]: item for item in csv.DictReader(handle)}
    for row in rows:
        label = labels.get(row["scenario_id"], {})
        row["granularity"] = label.get("granularity", row.get("granularity", ""))
    if int(manifest.get("seed", 20260714)) != 20260714:
        raise ValueError(f"共享 manifest seed 非固定值: {manifest.get('seed')}")
    return manifest

def manifest_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "granularity": row.get("granularity", ""),
        "source_sha256": row.get("source_sha256", ""),
        "original_source_path": row.get("original_source_path", ""),
        "original_source_sha256": row.get("original_source_sha256", ""),
        "synthetic": bool(row.get("synthetic", False)),
        "parent_scenario_id": row.get("parent_scenario_id", ""),
        "generation_rule": row.get("generation_rule", ""),
        "seed": row.get("seed", 20260714),
    }

def manifest_rows(manifest: Dict[str, Any], max_cases: Optional[int], offset: int, only_types: Optional[List[str]], cases_per_type: Optional[int]) -> List[Dict[str, Any]]:
    allowed = set(only_types) if only_types else SUPPORTED_TYPES
    rows = [dict(row) for row in manifest["rows"] if row.get("source_violation_type") in allowed]
    if offset:
        rows = rows[offset:]
    if cases_per_type is not None:
        counts: Counter = Counter()
        selected = []
        for row in rows:
            violation_type = normalize_type(str(row.get("source_violation_type", "")))
            if counts[violation_type] < cases_per_type:
                counts[violation_type] += 1
                selected.append(row)
        rows = selected
    if max_cases is not None:
        rows = rows[:max_cases]
    return rows

def retrieval_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("source_path")): row for row in load_json(path).get("results", []) if row.get("source_path")}

def enrich_manifest_rows(rows: List[Dict[str, Any]], retrieval_by_source: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for index, row in enumerate(rows, 1):
        merged = dict(row)
        retrieval_row = retrieval_by_source.get(str(row.get("source_path")))
        if retrieval_row is None and row.get("synthetic"):
            retrieval_row = retrieval_by_source.get(str(row.get("original_source_path")))
        if retrieval_row:
            merged["retrieval_source_path"] = retrieval_row.get("source_path")
            merged["retrieval"] = retrieval_row.get("retrieval", {})
            merged["scenario_spec"] = dict(retrieval_row.get("scenario_spec") or retrieval_row.get("retrieval", {}).get("scenario_spec") or {})
            merged.setdefault("status", retrieval_row.get("status"))
        if merged.get("scenario_spec"):
            merged["scenario_spec"]["violation_type"] = normalize_type(str(merged["scenario_spec"].get("violation_type", row.get("source_violation_type", ""))))
        merged["original_case_number"] = index
        enriched.append(merged)
    return enriched

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def normalize_type(violation_type: str) -> str:
    return TYPE_NORMALIZATION.get(violation_type, violation_type)

def remove_invalid_environment_result(summary_row: Dict[str, Any], output_dir: Path) -> None:
    failure_type = summary_row.get("failure_type")
    if failure_type not in {"simulator_unhealthy_before", "simulator_unhealthy_after"}:
        return
    original_path = Path(str(summary_row.get("output_path", "")))
    if not original_path.is_file():
        return

    invalid_dir = output_dir / "invalid_environment_case_results"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    destination_path = invalid_dir / original_path.name
    if destination_path.exists():
        destination_path = invalid_dir / f"{original_path.stem}_{int(time.time() * 1000)}{original_path.suffix}"
    original_path.replace(destination_path)

    record = {
        "scenario_id": summary_row.get("scenario_id"),
        "failure_type": failure_type,
        "original_path": str(original_path),
        "new_path": str(destination_path),
        "health": {
            "before": summary_row.get("health_before"),
            "after": summary_row.get("health_after"),
        },
        "time": datetime.now(timezone.utc).isoformat(),
    }
    record_path = output_dir / "removed_simulator_unhealthy_cases.json"
    payload = load_json(record_path) if record_path.exists() else {"removed_cases": []}
    records = payload.setdefault("removed_cases", [])
    key = (record["scenario_id"], record["failure_type"], record["original_path"])
    for index, existing in enumerate(records):
        existing_key = (existing.get("scenario_id"), existing.get("failure_type"), existing.get("original_path"))
        if existing_key == key:
            records[index] = record
            break
    else:
        records.append(record)
    dump_json(payload, record_path)

def normalize_node(node: Dict[str, Any], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fallback = fallback or {}
    road_type = node.get("road_type", fallback.get("road_type", "RoadSegment"))
    if isinstance(road_type, str) and road_type.lower() == "straight":
        road_type = "Straight"
    return {
        "node_id": node.get("node_id", node.get("id", fallback.get("node_id", fallback.get("id", "")))),
        "road_id": int(node.get("road_id", fallback.get("road_id", 0))),
        "lane_id": int(node.get("lane_id", fallback.get("lane_id", 0))),
        "section_id": int(node.get("section_id", fallback.get("section_id", 0)) or 0),
        "road_type": road_type,
        "lane_count": int(node.get("lane_count", fallback.get("lane_count", 1)) or 1),
        "curvature": float(node.get("curvature", fallback.get("curvature", 0.0)) or 0.0),
        "speed_limit": float(node.get("speed_limit", fallback.get("speed_limit", 40.0)) or 40.0),
        "is_junction": bool(node.get("is_junction", fallback.get("is_junction", road_type == "Intersection"))),
        "lane_change": node.get("lane_change", fallback.get("lane_change")),
        "has_traffic_light": bool(node.get("has_traffic_light", fallback.get("has_traffic_light", False))),
        "has_shoulder": bool(node.get("has_shoulder", fallback.get("has_shoulder", False))),
        "start": node.get("start", fallback.get("start", {})) or {},
        "end": node.get("end", fallback.get("end", {})) or {},
        "heading": float(node.get("heading", fallback.get("heading", 0.0)) or 0.0),
    }

def local_to_retrieval_result(row: Dict[str, Any]) -> RetrievalResult:
    local_top = row.get("retrieval", {}).get("local_top_k", [])
    if not local_top:
        raise RuntimeError("检索结果没有 local_top_k")
    top = local_top[0]
    matched_nodes: List[Dict[str, Any]] = []
    if top.get("match_type") == "lane_pair" and top.get("source_node") and top.get("target_node"):
        matched_nodes.append(normalize_node(top["source_node"], top))
        matched_nodes.append(normalize_node(top["target_node"], top))
    else:
        matched_nodes.append(normalize_node(top))
        for extra in local_top[1:3]:
            if extra.get("road_id") != top.get("road_id") or extra.get("lane_id") != top.get("lane_id"):
                matched_nodes.append(normalize_node(extra))
    community = CommunityRecord(
        community_id=str(top.get("community_id", "")),
        map_name=str(top.get("map_name", "")),
        node_ids=[node.get("node_id", "") for node in matched_nodes],
        structure={"source": "rq1_carla_full_execution", "match_type": top.get("match_type", "node")},
        summary=f"RQ1 CARLA execution local match for {row.get('source_violation_type')}",
        applicable_violations=[row.get("source_violation_type", "")],
        score=float(top.get("community_score", top.get("score", 1.0)) or 1.0),
    )
    return RetrievalResult(community=community, matched_nodes=matched_nodes, score=float(top.get("score", 1.0) or 1.0))

def selected_rows(
    payload: Dict[str, Any],
    max_cases: Optional[int],
    offset: int,
    only_types: Optional[List[str]],
    cases_per_type: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []
    allowed = set(only_types) if only_types else SUPPORTED_TYPES
    for row in payload.get("results", []):
        source_type = row.get("source_violation_type", "")
        if source_type not in allowed:
            continue
        if row.get("status") != "ok":
            continue
        normalized = normalize_type(source_type)
        spec = dict(row.get("scenario_spec") or row.get("retrieval", {}).get("scenario_spec") or {})
        if not spec:
            continue
        spec["violation_type"] = normalize_type(str(spec.get("violation_type", source_type)))
        if normalized not in {"未保持安全距离", "未按规定让行", "未注意前方路况", "违规变道", "违规超车", "逆行", "超速", "闯红灯"}:
            continue
        row = dict(row)
        row["scenario_spec"] = spec
        rows.append(row)

    for original_case_number, row in enumerate(rows, 1):
        row["original_case_number"] = original_case_number
    if offset:
        rows = rows[offset:]
    if cases_per_type is not None:
        per_type_counts: Counter = Counter()
        sampled_rows = []
        for row in rows:
            normalized = normalize_type(str(row.get("source_violation_type", "")))
            if per_type_counts[normalized] >= cases_per_type:
                continue
            per_type_counts[normalized] += 1
            sampled_rows.append(row)
        rows = sampled_rows
    if max_cases is not None:
        rows = rows[:max_cases]
    return rows

def run_one(row: Dict[str, Any], output_dir: Path, base: Path, timeout_s: float, prepare_only: bool = False) -> Dict[str, Any]:
    source_path = Path(str(row.get("source_path", "")))
    source = load_json(source_path) if source_path.exists() else {}
    spec = dict(row.get("scenario_spec") or row.get("retrieval", {}).get("scenario_spec") or {})
    violation_type = normalize_type(str(row.get("source_violation_type", spec.get("violation_type", ""))))
    spec["violation_type"] = violation_type
    scenario_id = str(row["scenario_id"])
    result_dir = output_dir / ("preparation_results" if prepare_only else "case_results")
    result_path = result_dir / f"{scenario_id}.json"
    metadata = manifest_metadata(row)
    if result_path.exists():
        prior = load_json(result_path)
        summary_row = dict(prior.get("summary_row", {}))
        summary_row.update(metadata)
        summary_row["scenario_id"] = scenario_id
        summary_row["source_path"] = str(source_path)
        summary_row["_result_cached"] = True
        return summary_row

    health_before = {"healthy": True, "reason": "prepare_only"} if prepare_only else carla_health_check(timeout_s=5.0)
    stage_timing_s: Dict[str, float] = {}
    try:
        start = time.perf_counter()
        retrieval = local_to_retrieval_result(row)
        stage_timing_s["retrieval_adaptation"] = round(time.perf_counter() - start, 4)
        start = time.perf_counter()
        config = ConstraintSolver().solve(spec, [retrieval])
        config.scenario_id = scenario_id
        stage_timing_s["constraint_solving"] = round(time.perf_counter() - start, 4)
        start = time.perf_counter()
        constraint_validation = ConstraintValidator().validate(config.to_dict())
        stage_timing_s["constraint_validation"] = round(time.perf_counter() - start, 4)
        if prepare_only:
            summary_row = {
                "scenario_id": scenario_id, "source_path": str(source_path), "violation_type": config.violation_type,
                "source_violation_type": row.get("source_violation_type", ""), "status": "prepared", "detected": False,
                "constraint_satisfied": constraint_validation.get("satisfied"),
                "constraint_satisfaction_rate": constraint_validation.get("satisfaction_rate"),
                **metadata, "stage_timing_s": stage_timing_s,
                "total_generation_time_s": round(sum(stage_timing_s.values()), 4), "output_path": str(result_path),
            }
            dump_json({"source_path": str(source_path), "source_organized_scenario": source, "retrieval_row": row,
                       "scenario_spec": spec, "scenario_config": config.to_dict(),
                       "constraint_validation": constraint_validation, "stage_timing_s": stage_timing_s,
                       "summary_row": summary_row}, result_path)
            return summary_row
        if not health_before.get("healthy", False):
            raise RuntimeError("CARLA simulator unhealthy before case")
        start = time.perf_counter()
        from RKGRScen.execution.carla_runner import CarlaScenarioRunner
        trace = CarlaScenarioRunner(timeout_s=timeout_s).run(config)
        stage_timing_s["execution"] = round(time.perf_counter() - start, 4)
        if not trace.ticks:
            raise RuntimeError("trace empty: CARLA runner 返回空执行轨迹")
        start = time.perf_counter()
        violation = detect_violation(config.violation_type, trace.ticks, config.expected_violation.get("params", {}))
        stage_timing_s["violation_detection"] = round(time.perf_counter() - start, 4)
        result = {"source_path": str(source_path), "source_organized_scenario": source, "retrieval_row": row,
                  "scenario_spec": spec, "scenario_config": config.to_dict(), "execution_trace": trace.to_dict(),
                  "violation_result": violation, "stage_timing_s": stage_timing_s,
                  "constraint_validation": constraint_validation, "health_before": health_before,
                  "health_after": carla_health_check(timeout_s=5.0)}
        start = time.perf_counter()
        result["scene_match_evaluation"] = ScenarioMatchEvaluator(base).evaluate_result(result, source)
        stage_timing_s["scene_match_evaluation"] = round(time.perf_counter() - start, 4)
        summary_row = {"scenario_id": scenario_id, "source_path": str(source_path), "violation_type": config.violation_type,
                       "source_violation_type": row.get("source_violation_type", ""), "status": "ok",
                       "detected": bool(violation.get("detected")), "tick_count": len(trace.ticks),
                       "match_score": result["scene_match_evaluation"].get("match_score"),
                       "grade": result["scene_match_evaluation"].get("grade"), "map": config.map_name,
                       "constraint_satisfied": constraint_validation.get("satisfied"),
                       "constraint_satisfaction_rate": constraint_validation.get("satisfaction_rate"),
                       "reason": violation.get("reason", ""), **metadata, "stage_timing_s": stage_timing_s,
                       "total_generation_time_s": round(sum(stage_timing_s.values()), 4), "output_path": str(result_path)}
        result["summary_row"] = summary_row
        dump_json(result, result_path)
        return summary_row
    except Exception as exc:
        health_after = health_before if prepare_only else carla_health_check(timeout_s=5.0)
        error_text = repr(exc)
        failure_type = classify_failure(1, error_text, health_before, health_after)
        summary_row = {"scenario_id": scenario_id, "source_path": str(source_path), "violation_type": violation_type,
                       "source_violation_type": row.get("source_violation_type", ""), "status": "failed", "detected": False,
                       "failure_type": failure_type, "error": error_text, "health_before": health_before,
                       "health_after": health_after, **metadata, "stage_timing_s": stage_timing_s, "output_path": str(result_path)}
        dump_json({"summary_row": summary_row}, result_path)
        return summary_row

def plot_execution_by_type(rows: List[Dict[str, Any]], path: Path) -> None:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row.get("source_violation_type") or row.get("violation_type")][row.get("status", "unknown")] += 1
    types = sorted(counts)
    ok = [counts[t].get("ok", 0) for t in types]
    failed = [counts[t].get("failed", 0) for t in types]
    x = range(len(types))
    plt.figure(figsize=(14, 6))
    plt.bar(x, ok, label="ok")
    plt.bar(x, failed, bottom=ok, label="failed")
    plt.xticks(list(x), types, rotation=35, ha="right")
    plt.ylabel("count")
    plt.title("RQ1 CARLA execution status by type")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def plot_detected_by_type(rows: List[Dict[str, Any]], path: Path) -> None:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("status") != "ok":
            continue
        counts[row.get("source_violation_type") or row.get("violation_type")]["detected" if row.get("detected") else "not_detected"] += 1
    types = sorted(counts)
    detected = [counts[t].get("detected", 0) for t in types]
    not_detected = [counts[t].get("not_detected", 0) for t in types]
    x = range(len(types))
    plt.figure(figsize=(14, 6))
    plt.bar(x, detected, label="detected")
    plt.bar(x, not_detected, bottom=detected, label="not_detected")
    plt.xticks(list(x), types, rotation=35, ha="right")
    plt.ylabel("count")
    plt.title("RQ1 CARLA violation reproduction by type")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def build_report(rows: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path, selected_total: int) -> None:
    by_type: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = row.get("source_violation_type") or row.get("violation_type")
        by_type[key]["total"] += 1
        by_type[key][row.get("status", "unknown")] += 1
        if row.get("detected"):
            by_type[key]["detected"] += 1
        if row.get("constraint_satisfied"):
            by_type[key]["constraint_satisfied"] += 1
    lines = [
        "# RQ1 CARLA 全量执行测试记录报告",
        "",
        "## 1. 实验口径",
        "",
        "本报告只统计真实调用 `CarlaScenarioRunner.run()` 的场景执行结果，不再把纯检索结果当作执行结果。",
        "",
        f"- 待执行输入数：{selected_total}",
        f"- 已记录结果数：{len(rows)}",
        f"- 成功执行：{summary.get('executed', 0)}",
        f"- 行为复现检测触发：{summary.get('detected', 0)}",
        f"- 执行失败：{summary.get('failed', 0)}",
        f"- 平均约束满足率：{summary.get('avg_constraint_satisfaction_rate', 0.0)}",
        f"- 平均总耗时：{summary.get('avg_total_generation_time_s', 0.0)}s",
        "",
        "## 2. 按违规类型统计",
        "",
        "| 违规类型 | 总数 | 成功执行 | 检测触发 | 执行失败 | 约束满足 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for type_name in sorted(by_type):
        c = by_type[type_name]
        lines.append(f"| {type_name} | {c['total']} | {c['ok']} | {c['detected']} | {c['failed']} | {c['constraint_satisfied']} |")
    lines.extend([
        "",
        "## 3. 失败类型",
        "",
        "| 失败类型 | 数量 |",
        "|---|---:|",
    ])
    for failure_type, count in sorted((summary.get("failure_types") or {}).items()):
        lines.append(f"| {failure_type} | {count} |")
    lines.extend([
        "",
        "## 4. 输出文件",
        "",
        "- `summary.json`：执行汇总",
        "- `execution_rows.csv`：逐场景执行结果",
        "- `case_results/`：逐场景完整 trace、配置、检测证据",
        "- `charts/execution_status_by_type.png`：执行状态图",
        "- `charts/detection_by_type.png`：复现检测图",
    ])
    (output_dir / "rq1_carla_execution_report.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--only-type", action="append", default=None)
    parser.add_argument("--cases-per-type", type=int, default=None)
    parser.add_argument("--output-dir", default="RKGRScen/data/evaluation/rq1_carla_full_execution")
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--manifest", default=DEFAULT_SHARED_MANIFEST)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.cases_per_type is not None and args.cases_per_type < 0:
        parser.error("--cases-per-type 必须是非负整数")

    base = Path(__file__).resolve().parents[2]
    retrieval_path = base / "RKGRScen" / "data" / "retrieval" / "p0_graphrag_retrieval_full" / "retrieval_results.json"
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = base / manifest_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_manifest = load_shared_manifest(manifest_path, base)
    rows_to_run = enrich_manifest_rows(
        manifest_rows(shared_manifest, args.max_cases, args.offset, args.only_type, args.cases_per_type),
        retrieval_index(retrieval_path),
    )
    selected_meta = {
        "manifest_path": str(manifest_path),
        "retrieval_path": str(retrieval_path),
        "output_dir": str(output_dir),
        "selected_total": len(rows_to_run),
        "shared_scenario_count": shared_manifest.get("scenario_count", len(shared_manifest.get("rows", []))),
        "max_cases": args.max_cases,
        "offset": args.offset,
        "only_type": args.only_type,
        "cases_per_type": args.cases_per_type,
        "prepare_only": args.prepare_only,
        "supported_types": sorted(SUPPORTED_TYPES),
    }
    dump_json(selected_meta, output_dir / "selected_input_meta.json")

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(rows_to_run):
            raise SystemExit(f"case-index 越界: {args.case_index}, total={len(rows_to_run)}")
        row = rows_to_run[args.case_index]
        result = run_one(row, output_dir, base, args.timeout_s, args.prepare_only)
        if not args.prepare_only and not result.get("_result_cached", False):
            remove_invalid_environment_result(result, output_dir)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows_to_run, 1):
        health_before = {"healthy": True, "reason": "prepare_only"} if args.prepare_only else carla_health_check(timeout_s=5.0)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case-index",
                str(idx - 1),
                "--offset",
                str(args.offset),
                "--timeout-s",
                str(args.timeout_s),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest_path),
                *(["--prepare-only"] if args.prepare_only else []),
                *(["--max-cases", str(args.max_cases)] if args.max_cases is not None else []),
                *(["--cases-per-type", str(args.cases_per_type)] if args.cases_per_type is not None else []),
                *(sum((["--only-type", item] for item in (args.only_type or [])), [])),
            ],
            cwd=str(base),
            capture_output=True,
            text=True,
        )
        health_after = health_before if args.prepare_only else carla_health_check(timeout_s=5.0)
        if completed.returncode == 0:
            json_lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
            original_case_number = int(row["original_case_number"])
            result = json.loads(json_lines[-1]) if json_lines else {"scenario_id": f"rq1_carla_{original_case_number:05d}", "status": "failed", "failure_type": "unknown_failure"}
            result.setdefault("health_before", health_before)
            result.setdefault("health_after", health_after)
            if result.get("status") == "failed" and not result.get("_result_cached", False):
                error_text = str(result.get("error", completed.stdout + completed.stderr))
                result["failure_type"] = classify_failure(1, error_text, health_before, health_after)
                result["health_before"] = health_before
                result["health_after"] = health_after
        else:
            output = completed.stdout + completed.stderr
            failure_type = classify_failure(completed.returncode, output, health_before, health_after)
            source_path = Path(row.get("source_path", ""))
            source_type = row.get("source_violation_type", "")
            normalized_type = normalize_type(source_type)
            scenario_id = str(row["scenario_id"])
            result_dir = output_dir / ("preparation_results" if args.prepare_only else "case_results")
            result = {
                "scenario_id": scenario_id,
                "source_path": row.get("source_path", ""),
                "violation_type": normalized_type,
                "source_violation_type": source_type,
                "status": "failed",
                "detected": False,
                "failure_type": failure_type,
                "returncode": completed.returncode,
                "health_before": health_before,
                "health_after": health_after,
                "error": output[-4000:],
                "output_path": str(result_dir / f"{scenario_id}.json"),
            }
            dump_json({"summary_row": result}, result_dir / f"{scenario_id}.json")
        if not args.prepare_only and not result.pop("_result_cached", False):
            remove_invalid_environment_result(result, output_dir)
        rows.append(result)
        print(json.dumps({"progress": f"{idx}/{len(rows_to_run)}", **result}, ensure_ascii=False), flush=True)
        rows_name = "preparation_rows.csv" if args.prepare_only else "execution_rows.csv"
        write_csv(output_dir / rows_name, rows)
        dump_json(summarize_rows(rows), output_dir / "summary_partial.json")

    summary = summarize_rows(rows)
    summary["selected_input"] = selected_meta
    if args.prepare_only:
        dump_json(summary, output_dir / "preparation_summary.json")
        write_csv(output_dir / "preparation_rows.csv", rows)
    else:
        dump_json(summary, output_dir / "summary.json")
        write_csv(output_dir / "execution_rows.csv", rows)
        plot_execution_by_type(rows, output_dir / "charts" / "execution_status_by_type.png")
        plot_detected_by_type(rows, output_dir / "charts" / "detection_by_type.png")
        build_report(rows, summary, output_dir, len(rows_to_run))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
