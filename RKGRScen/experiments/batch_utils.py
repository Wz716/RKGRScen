import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import carla
except ImportError:
    carla = None

def carla_health_check(host: str = "localhost", port: int = 2000, timeout_s: float = 5.0) -> Dict[str, Any]:
    if carla is None:
        return {"healthy": False, "reason": "carla package not available"}
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout_s)
        world = client.get_world()
        snapshot = world.get_snapshot()
        return {
            "healthy": True,
            "map": world.get_map().name,
            "frame": snapshot.frame if snapshot else None,
        }
    except Exception as exc:
        return {"healthy": False, "reason": repr(exc)}

def classify_failure(returncode: int, output: str, health_before: Optional[Dict[str, Any]] = None, health_after: Optional[Dict[str, Any]] = None) -> str:
    text = output.lower()
    explicit_failures = (
        ("waypoint_not_found", ("waypoint not found", "找不到 waypoint", "找不到可行驶 waypoint", "waypoint 参数无效", "waypoint 不是 driving", "waypoint 缺少车道类型")),
        ("invalid_lane_topology", ("invalid lane topology", "无效车道拓扑", "车道拓扑无效", "目标相邻 driving 车道不存在")),
        ("spawn_plan_unsatisfied", ("spawn plan unsatisfied", "spawn 计划无法满足", "生成计划无法满足", "无法满足地图占用和场景关系约束")),
        ("spawn_precheck_failed", ("spawn 合法性预检查失败", "spawn precheck failed", "spawn pre-check failed", "生成合法性预检查失败", "生成失败")),
    )
    for failure_type, markers in explicit_failures:
        if any(marker in text for marker in markers):
            return failure_type

    if health_before and not health_before.get("healthy", False):
        return "simulator_unhealthy_before"
    if health_after and not health_after.get("healthy", False):
        return "simulator_unhealthy_after"
    if "destroyed actor" in text or "actor has been destroyed" in text or "trying to operate on a destroyed actor" in text or "参与者已销毁" in text or returncode in {-6, -11}:
        return "actor_destroyed"
    if "trace empty" in text or "empty trace" in text or "trace has no ticks" in text or "轨迹为空" in text or "执行轨迹为空" in text:
        return "trace_empty"
    if "timeout" in text or "timed out" in text or "超时" in text:
        return "simulator_timeout"
    if any(marker in text for marker in ("carla_rpc_error", "rpc::rpc_error", "rpc error", "rpc_error", "rpc exception", "carla.client", "failed to connect to carla", "无法连接 carla", "carla rpc 错误")):
        return "carla_rpc_error"
    if "没有可执行 local candidate" in output or "没有 local_top_k" in output or "找不到检索结果" in output or "no local match" in text:
        return "no_local_match"
    return "unknown_failure"

def run_case_subprocess(script_path: Path, case_index: int, cwd: Path, python_executable: str = sys.executable) -> Dict[str, Any]:
    health_before = carla_health_check()
    completed = subprocess.run(
        [python_executable, str(script_path), "--case-index", str(case_index)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    health_after = carla_health_check()
    if completed.returncode == 0:
        lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
        row = json.loads(lines[-1]) if lines else {"scenario_id": f"case_{case_index + 1}", "status": "ok", "detected": False}
        row["health_before"] = health_before
        row["health_after"] = health_after
        return row
    return {
        "scenario_id": f"case_{case_index + 1}",
        "status": "failed",
        "returncode": completed.returncode,
        "failure_type": classify_failure(completed.returncode, output, health_before, health_after),
        "health_before": health_before,
        "health_after": health_after,
        "error": output,
    }

def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "total": len(rows),
        "executed": 0,
        "detected": 0,
        "failed": 0,
        "high_match": 0,
        "medium_match": 0,
        "low_match": 0,
        "failure_types": {},
        "constraint_satisfied": 0,
        "avg_constraint_satisfaction_rate": 0.0,
        "avg_total_generation_time_s": 0.0,
        "avg_stage_timing_s": {},
        "results": rows,
    }
    constraint_rates = []
    total_times = []
    stage_values: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("status") == "failed":
            summary["failed"] += 1
            failure_type = row.get("failure_type", "unknown_failure")
            summary["failure_types"][failure_type] = summary["failure_types"].get(failure_type, 0) + 1
            continue
        summary["executed"] += 1
        summary["detected"] += 1 if row.get("detected") else 0
        summary["constraint_satisfied"] += 1 if row.get("constraint_satisfied") else 0
        if row.get("constraint_satisfaction_rate") is not None:
            constraint_rates.append(float(row.get("constraint_satisfaction_rate")))
        if row.get("total_generation_time_s") is not None:
            total_times.append(float(row.get("total_generation_time_s")))
        for stage, value in (row.get("stage_timing_s") or {}).items():
            stage_values.setdefault(stage, []).append(float(value))
        grade = row.get("grade") or "low"
        if grade in {"high", "medium", "low"}:
            summary[f"{grade}_match"] += 1
    if constraint_rates:
        summary["avg_constraint_satisfaction_rate"] = round(sum(constraint_rates) / len(constraint_rates), 4)
    if total_times:
        summary["avg_total_generation_time_s"] = round(sum(total_times) / len(total_times), 4)
    summary["avg_stage_timing_s"] = {stage: round(sum(values) / len(values), 4) for stage, values in sorted(stage_values.items())}
    return summary

def dump_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
