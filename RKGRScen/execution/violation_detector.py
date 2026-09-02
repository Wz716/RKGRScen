from typing import Any, Dict, List, Optional, Union

from RKGRScen.models import ExecutionTrace, ViolationResult

class ViolationDetector:
    def detect(
        self,
        violation_type: str,
        trace: Union[ExecutionTrace, List[Dict[str, Any]]],
        expected_params: Optional[Dict[str, Any]] = None,
    ) -> ViolationResult:
        scenario_id = trace.scenario_id if isinstance(trace, ExecutionTrace) else ""
        ticks = trace.ticks if isinstance(trace, ExecutionTrace) else trace
        result = detect_violation(violation_type, ticks, expected_params or {})
        evidence = result.get("evidence")
        timestamp_s, location = _extract_evidence_position(evidence)
        return ViolationResult(
            scenario_id=scenario_id,
            violation_type=violation_type,
            detected=bool(result.get("detected", False)),
            reason=str(result.get("reason", "")),
            timestamp_s=timestamp_s,
            location=location,
            severity=result.get("severity"),
        )

def _extract_evidence_position(evidence: Any) -> Any:
    if not isinstance(evidence, dict):
        return None, None
    timestamp_s = evidence.get("timestamp_s")
    location = evidence.get("location")
    if location is None:
        location = evidence.get("ego", {}).get("location")
    nested = evidence.get("step")
    if isinstance(nested, dict):
        nested_timestamp, nested_location = _extract_evidence_position(nested)
        timestamp_s = timestamp_s if timestamp_s is not None else nested_timestamp
        location = location if location is not None else nested_location
    return timestamp_s, location

def detect_violation(violation_type: str, trace: List[Dict[str, Any]], expected_params: Dict[str, Any]) -> Dict[str, Any]:
    if violation_type == "未按规定让行":
        return detect_yield_violation(trace, expected_params)
    if violation_type == "闯红灯":
        return detect_red_light_violation(trace, expected_params)
    if violation_type == "违规变道":
        return detect_lane_change_violation(trace, expected_params)
    if violation_type == "违规超车":
        return detect_overtake_violation(trace, expected_params)
    if violation_type == "超速":
        return detect_speeding_violation(trace, expected_params)
    if violation_type == "逆行":
        return detect_wrong_way_violation(trace, expected_params)
    if violation_type == "未保持安全距离":
        return detect_following_distance_violation(trace, expected_params)
    if violation_type == "未注意前方路况":
        return detect_inattention_front_condition(trace, expected_params)
    return {"detected": False, "reason": "暂不支持的违规类型"}

def detect_yield_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据"}
    time_gap_range = params.get("time_gap_range_s", [0.5, 2.0])
    danger_distance = params.get("danger_distance_m", 8.0)
    for step in trace:
        ego = step.get("ego", {})
        npcs = step.get("npcs", [])
        if not npcs:
            continue
        ego_conflict_dist = ego.get("distance_to_conflict_m")
        ego_speed = ego.get("speed_mps", 0.0)
        if ego_conflict_dist is None or ego_speed <= 0.1:
            continue
        ego_ttc = ego_conflict_dist / max(ego_speed, 0.1)
        for npc in npcs:
            npc_conflict_dist = npc.get("distance_to_conflict_m")
            npc_speed = npc.get("speed_mps", 0.0)
            pair_distance = npc.get("distance_to_ego_m")
            if npc_conflict_dist is None or npc_speed <= 0.1:
                continue
            npc_ttc = npc_conflict_dist / max(npc_speed, 0.1)
            time_gap = abs(ego_ttc - npc_ttc)
            if time_gap_range[0] <= time_gap <= time_gap_range[1] and pair_distance is not None and pair_distance <= danger_distance:
                return {
                    "detected": True,
                    "reason": f"检测到未让行冲突，时间差 {round(time_gap, 2)}s，车间距离 {round(pair_distance, 2)}m",
                    "evidence": step,
                }
    return {"detected": False, "reason": "真实轨迹未形成让行冲突窗口"}

def detect_red_light_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据"}
    expected_phase = params.get("signal_phase", "red")
    stop_line_buffer = float(params.get("stop_line_buffer_m", 2.0))
    first_red_ts = None
    for step in trace:
        ts = step.get("timestamp_s")
        ego = step.get("ego", {})
        signal_state = ego.get("signal_state")
        if signal_state == expected_phase and first_red_ts is None:
            first_red_ts = ts
            break
    if first_red_ts is None:
        first_red_ts = params.get("first_red_timestamp_s")
    for step in trace:
        ts = step.get("timestamp_s")
        if first_red_ts is not None and ts is not None and ts < first_red_ts:
            continue
        ego = step.get("ego", {})
        signal_state = ego.get("signal_state")
        moving = ego.get("speed_mps", 0.0) > 0.5
        passed_stop_line = ego.get("passed_stop_line", False)
        dist = ego.get("distance_to_stop_line_m")
        if signal_state == expected_phase and moving and (passed_stop_line or (dist is not None and dist <= stop_line_buffer)):
            return {
                "detected": True,
                "reason": "检测到红灯相位下继续前进并越过停止线",
                "evidence": step,
            }
    return {"detected": False, "reason": "真实轨迹中未观测到红灯状态下越过停止线的证据"}

def detect_lane_change_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据", "closest_evidence": {"missing_requirements": ["轨迹数据"]}}
    danger_gap = float(params.get("danger_gap_m", 8.0))
    direction = params.get("lane_change_direction", "left")
    closest = None
    for step in trace:
        for violator in step.get("npcs", []):
            source_lane_id = violator.get("source_lane_id")
            target_lane_id = violator.get("target_lane_id")
            current_lane_id = violator.get("lane_id")
            phase = violator.get("lane_change_phase")
            pair_distance = violator.get("distance_to_ego_m")
            distinct_lanes = source_lane_id is not None and target_lane_id is not None and source_lane_id != target_lane_id
            direction_matches = violator.get("lane_change_direction") == direction
            target_lane_entered = target_lane_id is not None and current_lane_id == target_lane_id
            explicitly_completed = bool(violator.get("lane_change_completed")) or phase == "completed"
            target_entry = target_lane_entered or explicitly_completed
            dangerous = pair_distance is not None and float(pair_distance) <= danger_gap
            missing = []
            if not distinct_lanes:
                missing.append("源车道与目标车道必须不同")
            if not direction_matches:
                missing.append("变道方向不匹配")
            if not target_entry:
                missing.append("未进入目标车道且无 completed 证据")
            if not dangerous:
                missing.append("未进入危险距离")
            evidence_summary = {
                "npc_id": violator.get("id"),
                "timestamp_s": step.get("timestamp_s"),
                "source_lane_id": source_lane_id,
                "target_lane_id": target_lane_id,
                "current_lane_id": current_lane_id,
                "lane_change_direction": violator.get("lane_change_direction"),
                "expected_direction": direction,
                "lane_change_phase": phase,
                "lane_change_completed": explicitly_completed,
                "target_lane_entered": target_lane_entered,
                "distance_to_ego_m": pair_distance,
                "danger_gap_m": danger_gap,
                "control_steer": violator.get("control_steer", 0.0),
                "lateral_accel_proxy_mps2": violator.get("lateral_accel_proxy_mps2", 0.0),
                "max_steer_delta": violator.get("max_steer_delta", 0.0),
                "requirements": {
                    "distinct_source_target_lanes": distinct_lanes,
                    "direction_matches": direction_matches,
                    "target_lane_entry_or_completed": target_entry,
                    "within_danger_gap": dangerous,
                },
                "missing_requirements": missing,
            }
            score = sum(1 for met in evidence_summary["requirements"].values() if met)
            if closest is None or score > closest[0]:
                closest = (score, evidence_summary)
            if not missing:
                return {
                    "detected": True,
                    "reason": f"检测到违规变道：从车道 {source_lane_id} 进入 {target_lane_id}，危险距离 {round(float(pair_distance), 2)}m",
                    "evidence": step,
                    "evidence_summary": evidence_summary,
                }
    closest_evidence = closest[1] if closest else {"missing_requirements": ["轨迹中没有 NPC"]}
    return {
        "detected": False,
        "reason": "真实轨迹未满足危险变道的完整事件语义：" + "；".join(closest_evidence["missing_requirements"]),
        "closest_evidence": closest_evidence,
    }

def detect_overtake_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据", "closest_evidence": {"missing_requirements": ["轨迹数据"]}}
    danger_gap = float(params.get("danger_gap_m", 10.0))
    direction = params.get("lane_change_direction", "left")
    behind_threshold = float(params.get("behind_threshold_m", 3.0))
    ahead_threshold = float(params.get("ahead_threshold_m", 3.0))
    tracks: Dict[Any, Dict[str, Any]] = {}
    for step in trace:
        for npc in step.get("npcs", []):
            npc_id = npc.get("id")
            state = tracks.setdefault(npc_id, {
                "phase": "waiting_for_behind",
                "behind": None,
                "lane_change": None,
                "ahead": None,
                "min_distance_m": None,
                "last": None,
            })
            state["last"] = {"timestamp_s": step.get("timestamp_s"), "npc": npc}
            relative = npc.get("relative_longitudinal_m")
            pair_distance = npc.get("distance_to_ego_m")
            if state["phase"] != "waiting_for_behind" and pair_distance is not None:
                distance = float(pair_distance)
                state["min_distance_m"] = distance if state["min_distance_m"] is None else min(state["min_distance_m"], distance)
            if state["phase"] == "waiting_for_behind":
                if relative is not None and float(relative) <= -behind_threshold:
                    state["behind"] = {"timestamp_s": step.get("timestamp_s"), "relative_longitudinal_m": relative}
                    state["min_distance_m"] = float(pair_distance) if pair_distance is not None else None
                    state["phase"] = "waiting_for_lane_change"
                continue
            source_lane_id = npc.get("source_lane_id")
            target_lane_id = npc.get("target_lane_id")
            current_lane_id = npc.get("lane_id")
            adjacent_lanes = (
                source_lane_id is not None
                and target_lane_id is not None
                and source_lane_id != target_lane_id
                and abs(int(source_lane_id) - int(target_lane_id)) == 1
            )
            direction_matches = npc.get("lane_change_direction") == direction
            target_entry = target_lane_id is not None and current_lane_id == target_lane_id
            explicitly_completed = bool(npc.get("lane_change_completed")) or npc.get("lane_change_phase") == "completed"
            if state["phase"] == "waiting_for_lane_change" and adjacent_lanes and direction_matches and (target_entry or explicitly_completed):
                state["lane_change"] = {
                    "timestamp_s": step.get("timestamp_s"),
                    "source_lane_id": source_lane_id,
                    "target_lane_id": target_lane_id,
                    "current_lane_id": current_lane_id,
                    "target_lane_entered": target_entry,
                    "lane_change_completed": explicitly_completed,
                    "control_steer": npc.get("control_steer", 0.0),
                    "lateral_accel_proxy_mps2": npc.get("lateral_accel_proxy_mps2", 0.0),
                    "max_steer_delta": npc.get("max_steer_delta", 0.0),
                }
                state["phase"] = "waiting_for_ahead"
                continue
            if state["phase"] == "waiting_for_ahead" and relative is not None and float(relative) >= ahead_threshold:
                state["ahead"] = {"timestamp_s": step.get("timestamp_s"), "relative_longitudinal_m": relative}
                within_danger_gap = state["min_distance_m"] is not None and state["min_distance_m"] <= danger_gap
                evidence_summary = _overtake_evidence_summary(
                    npc_id, state, direction, behind_threshold, ahead_threshold, danger_gap, within_danger_gap
                )
                if within_danger_gap:
                    return {
                        "detected": True,
                        "reason": f"检测到违规超车完整序列，过程最小距离 {round(state['min_distance_m'], 2)}m",
                        "evidence": step,
                        "evidence_summary": evidence_summary,
                    }
                state["phase"] = "sequence_complete_but_safe_gap"
    closest = None
    for npc_id, state in tracks.items():
        within_danger_gap = state["min_distance_m"] is not None and state["min_distance_m"] <= danger_gap
        summary = _overtake_evidence_summary(
            npc_id, state, direction, behind_threshold, ahead_threshold, danger_gap, within_danger_gap
        )
        score = sum(1 for met in summary["requirements"].values() if met)
        if closest is None or score > closest[0]:
            closest = (score, summary)
    closest_evidence = closest[1] if closest else {"phase": "no_npc", "missing_requirements": ["轨迹中没有 NPC"]}
    return {
        "detected": False,
        "reason": "当前轨迹/场景配置未形成违规超车完整序列：" + "；".join(closest_evidence["missing_requirements"]),
        "closest_evidence": closest_evidence,
    }

def _overtake_evidence_summary(
    npc_id: Any,
    state: Dict[str, Any],
    direction: str,
    behind_threshold: float,
    ahead_threshold: float,
    danger_gap: float,
    within_danger_gap: bool,
) -> Dict[str, Any]:
    requirements = {
        "initially_behind": state.get("behind") is not None,
        "adjacent_lane_change_and_target_entry": state.get("lane_change") is not None,
        "subsequently_ahead": state.get("ahead") is not None,
        "within_danger_gap_during_sequence": within_danger_gap,
    }
    labels = {
        "initially_behind": f"未观测到 NPC 位于 ego 后方至少 {behind_threshold}m",
        "adjacent_lane_change_and_target_entry": f"未观测到方向为 {direction} 的相邻车道变道并进入目标车道",
        "subsequently_ahead": f"未观测到 NPC 随后位于 ego 前方至少 {ahead_threshold}m",
        "within_danger_gap_during_sequence": f"序列过程最小距离未达到 {danger_gap}m 危险阈值",
    }
    return {
        "npc_id": npc_id,
        "phase": state.get("phase"),
        "behind_evidence": state.get("behind"),
        "lane_change_evidence": state.get("lane_change"),
        "ahead_evidence": state.get("ahead"),
        "min_distance_m": state.get("min_distance_m"),
        "danger_gap_m": danger_gap,
        "requirements": requirements,
        "missing_requirements": [labels[key] for key, met in requirements.items() if not met],
    }

def detect_speeding_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据"}
    speed_limit_kmh = float(params.get("speed_limit_kmh", 40.0))
    target_speed_kmh = float(params.get("target_speed_kmh", speed_limit_kmh + 10.0))
    threshold_mps = speed_limit_kmh / 3.6
    target_mps = target_speed_kmh / 3.6
    subject = params.get("subject", "ego")
    violator_role = params.get("violator_role", "violator")
    consecutive = 0
    for step in trace:
        candidates = []
        if subject == "npc":
            candidates = [npc for npc in step.get("npcs", []) if npc.get("role") == violator_role]
            if not candidates:
                candidates = step.get("npcs", [])
        else:
            candidates = [step.get("ego", {})]
        for actor in candidates:
            speed_mps = float(actor.get("speed_mps", 0.0))
            if speed_mps >= threshold_mps and speed_mps >= target_mps - 1.0:
                consecutive += 1
                if consecutive >= 3:
                    subject_label = "NPC" if subject == "npc" else "ego"
                    return {
                        "detected": True,
                        "reason": f"检测到{subject_label}持续超速，速度 {round(speed_mps * 3.6, 2)}km/h，高于限速 {speed_limit_kmh}km/h",
                        "evidence": step,
                    }
            else:
                consecutive = 0
    return {"detected": False, "reason": "真实轨迹未达到持续超速阈值"}

def detect_inattention_front_condition(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据"}
    ttc_threshold = float(params.get("ttc_threshold_s", 3.0))
    danger_distance = float(params.get("danger_distance_m", 15.0))
    best = None
    for step in trace:
        ego = step.get("ego", {})
        ego_speed = float(ego.get("speed_mps", 0.0))
        for npc in step.get("npcs", []):
            if npc.get("role") not in {"obstacle", "hazard", "front_hazard", "lead", "npc", None}:
                continue
            gap = npc.get("distance_to_ego_m")
            if gap is None:
                continue
            gap = float(gap)
            npc_speed = float(npc.get("speed_mps", 0.0))
            ttc = float(npc.get("ttc_to_ego_s", gap / max(ego_speed - npc_speed, 0.1)))
            rss_margin = npc.get("rss_margin_m")
            if rss_margin is None:
                rss_safe_distance = _rss_longitudinal_safe_distance(
                    ego_speed,
                    npc_speed,
                    float(params.get("rss_response_time_s", 1.0)),
                    float(params.get("rss_ego_accel_max_mps2", 2.0)),
                    float(params.get("rss_ego_brake_min_mps2", 4.0)),
                    float(params.get("rss_front_brake_max_mps2", 8.0)),
                )
                rss_margin = gap - rss_safe_distance
            row = {"gap_m": round(gap, 3), "ttc_s": round(ttc, 3), "ego_speed_mps": round(ego_speed, 3), "obstacle_speed_mps": round(npc_speed, 3), "rss_margin_m": round(float(rss_margin), 3), "step": step}
            if best is None or row["ttc_s"] < best["ttc_s"]:
                best = row
            if ego_speed > 1.0 and gap <= danger_distance and (ttc <= ttc_threshold or float(rss_margin) < 0.0):
                return {
                    "detected": True,
                    "reason": f"检测到未注意前方路况风险，距离 {round(gap, 2)}m，TTC {round(ttc, 2)}s",
                    "evidence": row,
                }
    if best is not None:
        return {"detected": False, "reason": "真实轨迹未进入前方异常 TTC/RSS 风险窗口", "closest_evidence": best}
    return {"detected": False, "reason": "轨迹中没有可用的前方异常目标"}

def detect_following_distance_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据"}
    thw_threshold_s = float(params.get("thw_threshold_s", 1.0))
    min_gap_m = float(params.get("min_gap_m", 8.0))
    response_time_s = float(params.get("rss_response_time_s", 1.0))
    ego_accel_max = float(params.get("rss_ego_accel_max_mps2", 2.0))
    ego_brake_min = float(params.get("rss_ego_brake_min_mps2", 4.0))
    front_brake_max = float(params.get("rss_front_brake_max_mps2", 8.0))
    best = None
    for step in trace:
        gap = _first_present(step, ["gap_m", "distance_to_lead_m", "distance_to_ego_m"])
        thw = _first_present(step, ["thw_s", "time_headway_s"])
        ego_speed = _first_present(step, ["ego_speed_mps", "speed_mps"])
        lead_speed = _first_present(step, ["lead_speed_mps", "front_speed_mps", "npc_speed_mps"])
        if gap is None:
            ego = step.get("ego", {})
            npcs = step.get("npcs", [])
            gap = npcs[0].get("distance_to_ego_m") if npcs else None
            ego_speed = ego.get("speed_mps", ego_speed)
            lead_speed = npcs[0].get("speed_mps", lead_speed) if npcs else lead_speed
        if gap is None or ego_speed is None:
            continue
        gap = float(gap)
        ego_speed = float(ego_speed)
        lead_speed = float(lead_speed or 0.0)
        if thw is None:
            thw = gap / max(ego_speed, 0.1)
        thw = float(thw)
        rss_safe_distance = _rss_longitudinal_safe_distance(
            ego_speed,
            lead_speed,
            response_time_s,
            ego_accel_max,
            ego_brake_min,
            front_brake_max,
        )
        margin = gap - rss_safe_distance
        row = {
            "gap_m": round(gap, 3),
            "thw_s": round(thw, 3),
            "ego_speed_mps": round(ego_speed, 3),
            "lead_speed_mps": round(lead_speed, 3),
            "rss_safe_distance_m": round(rss_safe_distance, 3),
            "rss_margin_m": round(margin, 3),
            "step": step,
        }
        if best is None or row["rss_margin_m"] < best["rss_margin_m"]:
            best = row
        if ego_speed > 1.0 and thw <= thw_threshold_s and gap <= min_gap_m:
            return {
                "detected": True,
                "reason": f"检测到 THW 不足，距离 {round(gap, 2)}m，THW {round(thw, 2)}s",
                "evidence": row,
            }
        if ego_speed > 1.0 and gap < rss_safe_distance:
            return {
                "detected": True,
                "reason": f"检测到 RSS 纵向安全距离不足，实际距离 {round(gap, 2)}m，小于安全距离 {round(rss_safe_distance, 2)}m",
                "evidence": row,
            }
    if best is not None:
        return {"detected": False, "reason": "真实轨迹未触发 THW/RSS 安全距离不足", "closest_evidence": best}
    return {"detected": False, "reason": "轨迹缺少跟驰距离/速度字段"}

def _rss_longitudinal_safe_distance(ego_speed: float, front_speed: float, response_time_s: float, ego_accel_max: float, ego_brake_min: float, front_brake_max: float) -> float:
    ego_response_speed = ego_speed + ego_accel_max * response_time_s
    ego_distance = ego_speed * response_time_s + 0.5 * ego_accel_max * response_time_s ** 2 + ego_response_speed ** 2 / (2.0 * max(ego_brake_min, 0.1))
    front_distance = front_speed ** 2 / (2.0 * max(front_brake_max, 0.1))
    return max(0.0, ego_distance - front_distance)

def _first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None

def detect_wrong_way_violation(trace: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    if not trace:
        return {"detected": False, "reason": "无轨迹数据"}
    danger_distance = float(params.get("danger_distance_m", 10.0))
    heading_opposition_deg = float(params.get("heading_opposition_deg", 120.0))
    for step in trace:
        ego = step.get("ego", {})
        npcs = step.get("npcs", [])
        if not npcs:
            continue
        npc = npcs[0]
        pair_distance = npc.get("distance_to_ego_m")
        heading_gap = npc.get("heading_gap_to_ego_deg")
        ego_speed = float(ego.get("speed_mps", 0.0))
        npc_speed = float(npc.get("speed_mps", 0.0))
        if pair_distance is None or heading_gap is None:
            continue
        if pair_distance <= danger_distance and heading_gap >= heading_opposition_deg and ego_speed > 1.0 and npc_speed > 1.0:
            return {
                "detected": True,
                "reason": f"检测到逆行迎向接近，车间距离 {round(pair_distance, 2)}m，航向差 {round(heading_gap, 2)}°",
                "evidence": step,
            }
    return {"detected": False, "reason": "真实轨迹未形成迎向逆行接近"}
