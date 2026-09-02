from typing import Any, Dict, List

from RKGRScen.config import violation_map
from RKGRScen.llm_client import DeepSeekClient

class SceneExpander:

    ALLOWED_ROLES = ["violator", "priority", "ego", "npc"]

    def __init__(self, use_llm: bool = False) -> None:
        self.knowledge = violation_map()
        self.client = DeepSeekClient()
        self.use_llm = use_llm and self.client.enabled
        self.audit_metadata: Dict[str, Any] = {}
        self.output_schema = self._build_output_schema()

    def expand(
        self,
        dsl: Dict[str, Any],
        *,
        scenario_id: str = "",
        source_sha256: str = "",
    ) -> Dict[str, Any]:
        dsl = self._unwrap_organized_scenario(dsl)
        if self.use_llm:
            audit_context = {
                "scenario_id": scenario_id,
                "source_sha256": source_sha256,
            }
            return self._with_semantic_context(self._expand_with_llm(dsl, audit_context), dsl)
        violation_type = dsl["violation_type"]
        actors = dsl.get("actors", [])
        subject_mode = "dual" if len(actors) >= 2 else "single"
        if violation_type == "未按规定让行":
            spec = self._expand_yield_violation(dsl)
        elif violation_type == "闯红灯":
            spec = self._expand_red_light_violation(dsl)
        elif violation_type == "违规变道":
            spec = self._expand_lane_change_violation(dsl)
        elif violation_type == "违规超车":
            spec = self._expand_overtake_violation(dsl)
        elif violation_type == "逆行":
            spec = self._expand_wrong_way_violation(dsl)
        elif violation_type == "超速":
            spec = self._expand_speeding_violation(dsl)
        elif violation_type == "未保持安全距离":
            spec = self._expand_following_distance_violation(dsl)
        elif violation_type == "未注意前方路况":
            spec = self._expand_inattention_front_condition(dsl)
        else:
            spec = {
                "violation_type": violation_type,
                "subject_mode": subject_mode,
                "actors": self._normalize_actors(actors),
                "conflict": {
                    "type": f"{violation_type} 的语义展开",
                    "location": dsl.get("conflict_location", dsl.get("road_network", {}).get("type", "RoadSegment")),
                    "trigger_condition": dsl.get("trigger_condition", f"{violation_type} 的默认语义展开"),
                    "timing": {"time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.5, 2.0])},
                },
                "road_requirement": self._generic_requirement(dsl),
            }
        return self._with_semantic_context(spec, dsl)

    def _expand_with_llm(self, dsl: Dict[str, Any], audit_context: Dict[str, str]) -> Dict[str, Any]:
        violation_type = dsl["violation_type"]
        system_prompt = (
            "你是自动驾驶测试场景语义展开器。"
            "你必须严格输出 JSON，不要输出任何解释。"
            "输出顶层字段必须且只能包含 violation_type, subject_mode, actors, conflict, road_requirement。"
            "conflict 必须包含 type, location, trigger_condition, timing。"
            "timing 必须包含 time_gap_to_conflict_s，格式为长度为2的数组。"
            "road_requirement 必须包含 type, min_lanes, has_traffic_light, needs_opposing_lanes。"
            "role 只能使用 violator, priority, ego, npc。"
        )
        user_prompt = (
            f"违规类型知识: {self.knowledge.get(violation_type, {})}\n"
            f"输入DSL: {dsl}\n"
            "对于‘未按规定让行’，请尽量识别为对向直行优先、抢行左转或横向抢行等可检索语义。"
            "对于‘闯红灯’，请明确红灯相位、停止线、直行/左转动作与是否存在横向放行车。"
            "对于‘违规变道’，请明确源车道、目标车道、切入方向、目标车道优先车以及最小纵向间距。"
            "对于‘超速’，请明确限速、目标速度、起始速度与持续加速/保持超速的行为。"
            "如果 DSL 是双向交互路口，min_lanes 代表单侧最低有效车道要求，不要输出整条道路双向总车道数。"
            "请输出严格 JSON。"
        )
        result = self.client.generate_json(
            system_prompt,
            user_prompt,
            schema=self.output_schema,
            audit_metadata={
                "component": "scene_expander",
                "violation_type": violation_type,
                **audit_context,
            },
        )
        self.audit_metadata = dict(self.client.last_metadata)
        return self._normalize_expansion_result(result, dsl)

    def _build_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["violation_type", "subject_mode", "actors", "conflict", "road_requirement"],
            "properties": {
                "violation_type": {"type": "string", "enum": list(self.knowledge.keys())},
                "subject_mode": {"type": "string", "enum": ["single", "dual", "single_plus_obstacle"]},
                "actors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "role", "path", "action", "speed_kmh"],
                        "properties": {
                            "id": {"type": "string"},
                            "role": {"type": "string", "enum": self.ALLOWED_ROLES},
                            "path": {"type": "string"},
                            "action": {"type": "string"},
                            "speed_kmh": {"type": "number"},
                        },
                    },
                },
                "conflict": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "location", "trigger_condition", "timing"],
                    "properties": {
                        "type": {"type": "string"},
                        "location": {"type": "string"},
                        "trigger_condition": {"type": "string"},
                        "timing": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["time_gap_to_conflict_s"],
                            "properties": {
                                "time_gap_to_conflict_s": {
                                    "type": "array", "minItems": 2, "maxItems": 2,
                                    "items": {"type": "number"},
                                }
                            },
                        },
                    },
                },
                "road_requirement": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "min_lanes", "has_traffic_light", "needs_opposing_lanes"],
                    "properties": {
                        "type": {"type": "string"},
                        "min_lanes": {"type": "integer", "minimum": 1},
                        "has_traffic_light": {"type": "boolean"},
                        "needs_opposing_lanes": {"type": "boolean"},
                    },
                },
            },
        }

    def _normalize_expansion_result(self, result: Dict[str, Any], dsl: Dict[str, Any]) -> Dict[str, Any]:
        violation_type = result.get("violation_type", dsl["violation_type"])
        normalized_actors = self._normalize_actors(result.get("actors", dsl.get("actors", [])))
        conflict = result.get("conflict", {})
        timing = conflict.get("timing", {})
        time_gap = timing.get("time_gap_to_conflict_s", dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.5, 2.0]))
        if not isinstance(time_gap, list) or len(time_gap) != 2:
            time_gap = [0.5, 2.0]
        road_requirement = result.get("road_requirement", {})
        normalized = {
            "violation_type": violation_type,
            "subject_mode": result.get("subject_mode", "dual" if len(normalized_actors) >= 2 else "single"),
            "actors": normalized_actors,
            "conflict": {
                "type": conflict.get("type", self.knowledge[violation_type]["interaction_pattern"]),
                "location": conflict.get("location", conflict.get("zone", "intersection_center")),
                "trigger_condition": conflict.get("trigger_condition", "由场景执行器根据动作定义触发"),
                "timing": {"time_gap_to_conflict_s": time_gap},
            },
            "road_requirement": {
                "type": road_requirement.get("type", dsl.get("road_network", {}).get("type", "RoadSegment")),
                "min_lanes": int(road_requirement.get("min_lanes", dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 2) // 2 or 2))),
                "has_traffic_light": bool(road_requirement.get("has_traffic_light", dsl.get("road_network", {}).get("has_traffic_light", violation_type == "闯红灯"))),
                "needs_opposing_lanes": bool(road_requirement.get("needs_opposing_lanes", violation_type == "未按规定让行")),
            },
        }
        if violation_type == "超速":
            normalized["speed_requirement"] = {
                "speed_limit_kmh": int(dsl.get("road_network", {}).get("speed_limit_kmh", dsl.get("speed_limit_kmh", 40))),
                "target_speed_kmh": int(dsl.get("target_speed_kmh", max(int(dsl.get("road_network", {}).get("speed_limit_kmh", 40)) + 15, 50))),
            }
        return normalized

    def _expand_yield_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if len(actors) < 2:
            raise ValueError("未按规定让行场景至少需要两个参与者")
        actors[0]["role"] = actors[0].get("role", "violator")
        actors[1]["role"] = actors[1].get("role", "priority")
        return {
            "violation_type": "未按规定让行",
            "subject_mode": "dual",
            "actors": actors,
            "conflict": {
                "type": dsl.get("conflict_type", "左转车未让对向直行车"),
                "location": dsl.get("conflict_location", "路口中心区域"),
                "trigger_condition": dsl.get("trigger_condition", "让行方在优先车临近冲突点时抢行"),
                "timing": {
                    "time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.5, 2.0])
                },
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "Intersection"),
                "min_lanes": dsl.get("road_network", {}).get("min_lanes", 2),
                "has_traffic_light": dsl.get("road_network", {}).get("has_traffic_light", False),
                "needs_opposing_lanes": True,
            },
        }

    def _expand_red_light_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if actors:
            actors[0]["role"] = actors[0].get("role", "violator")
        if len(actors) > 1:
            actors[1]["role"] = actors[1].get("role", "priority")
        return {
            "violation_type": "闯红灯",
            "subject_mode": "dual" if len(actors) >= 2 else "single",
            "actors": actors,
            "conflict": {
                "type": dsl.get("conflict_type", "红灯相位通过停止线"),
                "location": dsl.get("conflict_location", "停止线与路口入口区域"),
                "trigger_condition": dsl.get("trigger_condition", "违规车在红灯相位保持前进并越过停止线"),
                "timing": {
                    "time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.0, 1.0])
                },
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "Intersection"),
                "min_lanes": dsl.get("road_network", {}).get("min_lanes", max(1, dsl.get("road_network", {}).get("lanes", 2) // 2)),
                "has_traffic_light": True,
                "needs_opposing_lanes": False,
            },
        }

    def _expand_lane_change_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if len(actors) < 2:
            raise ValueError("违规变道场景至少需要两个参与者")
        actors[0]["role"] = actors[0].get("role", "violator")
        actors[1]["role"] = actors[1].get("role", "priority")
        change_action = str(actors[0].get("action", "Change Lane Left"))
        lane_change_direction = "left" if "left" in change_action.lower() else "right"
        return {
            "violation_type": "违规变道",
            "subject_mode": "dual",
            "actors": actors,
            "conflict": {
                "type": dsl.get("conflict_type", "强行切入相邻车道"),
                "location": dsl.get("conflict_location", dsl.get("road_network", {}).get("type", "RoadSegment")),
                "trigger_condition": dsl.get("trigger_condition", "违规车在纵向间距不足时切入目标车道"),
                "timing": {
                    "time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.3, 1.2])
                },
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "RoadSegment"),
                "min_lanes": max(2, dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 2))),
                "has_traffic_light": False,
                "needs_opposing_lanes": False,
                "same_direction_multi_lane": True,
                "lane_change_direction": lane_change_direction,
            },
        }

    def _expand_overtake_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if len(actors) < 2:
            raise ValueError("违规超车场景至少需要两个参与者")
        actors[0]["role"] = actors[0].get("role", "violator")
        actors[1]["role"] = actors[1].get("role", "priority")
        change_action = str(actors[0].get("action", "Change Lane Left"))
        lane_change_direction = "left" if "left" in change_action.lower() else "right"
        return {
            "violation_type": "违规超车",
            "subject_mode": "dual",
            "actors": actors,
            "conflict": {
                "type": dsl.get("conflict_type", "借道超越前车并过早并回原车道"),
                "location": dsl.get("conflict_location", dsl.get("road_network", {}).get("type", "Straight")),
                "trigger_condition": dsl.get("trigger_condition", "后车在确认安全距离不足时强行超越前车"),
                "timing": {
                    "time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.2, 1.5])
                },
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "Straight"),
                "min_lanes": max(2, dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 2))),
                "has_traffic_light": False,
                "needs_opposing_lanes": True,
                "same_direction_multi_lane": True,
                "lane_change_direction": lane_change_direction,
            },
        }

    def _expand_speeding_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if not actors:
            actors = [{"id": "A", "role": "violator", "path": "straight", "action": "Move Forward", "speed_kmh": 60}]
        actors[0]["role"] = actors[0].get("role", "violator")
        speed_limit = int(dsl.get("road_network", {}).get("speed_limit_kmh", dsl.get("speed_limit_kmh", 40)))
        target_speed = int(dsl.get("target_speed_kmh", speed_limit + 15))
        return {
            "violation_type": "超速",
            "subject_mode": "single",
            "actors": actors[:1],
            "conflict": {
                "type": dsl.get("conflict_type", "超过道路限速阈值"),
                "location": dsl.get("conflict_location", dsl.get("road_network", {}).get("type", "RoadSegment")),
                "trigger_condition": dsl.get("trigger_condition", "违规车持续以高于限速的速度行驶"),
                "timing": {
                    "time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.0, 0.0])
                },
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "RoadSegment"),
                "min_lanes": max(1, dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 1))),
                "has_traffic_light": False,
                "needs_opposing_lanes": False,
            },
            "speed_requirement": {
                "speed_limit_kmh": speed_limit,
                "target_speed_kmh": target_speed,
            },
        }

    def _expand_wrong_way_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if len(actors) < 2:
            actors = actors + [{"id": "B", "role": "priority", "path": "N2S", "action": "Move Forward", "speed_kmh": 35}] * (2 - len(actors))
        actors[0]["role"] = actors[0].get("role", "violator")
        actors[1]["role"] = actors[1].get("role", "priority")
        return {
            "violation_type": "逆行",
            "subject_mode": "dual",
            "actors": actors[:2],
            "conflict": {
                "type": dsl.get("conflict_type", "在同一路段与正常车流相向行驶"),
                "location": dsl.get("conflict_location", dsl.get("road_network", {}).get("type", "Straight")),
                "trigger_condition": dsl.get("trigger_condition", "违规车沿相反方向进入正常行驶车道并与优先车迎向接近"),
                "timing": {
                    "time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.5, 2.0])
                },
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "Straight"),
                "min_lanes": max(2, dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 2))),
                "has_traffic_light": False,
                "needs_opposing_lanes": True,
            },
        }

    def _expand_following_distance_violation(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if len(actors) < 2:
            actors = [
                {"id": "ego", "role": "ego", "path": "follow", "action": "Move Forward", "speed_kmh": 35},
                {"id": "lead", "role": "lead", "path": "ahead", "action": "Lead Brake", "speed_kmh": 20},
            ]
        actors[0]["role"] = actors[0].get("role", "ego")
        actors[1]["role"] = actors[1].get("role", "lead")
        return {
            "violation_type": "未保持安全距离",
            "subject_mode": "dual",
            "actors": actors[:2],
            "conflict": {
                "type": dsl.get("conflict_type", "前车急刹诱发 ego 跟车距离不足"),
                "location": dsl.get("conflict_location", dsl.get("road_network", {}).get("type", "RoadSegment")),
                "trigger_condition": dsl.get("trigger_condition", "前车运行中急刹，ego 未保持足够纵向安全距离"),
                "timing": {"time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.0, 3.0])},
            },
            "road_requirement": {
                "type": dsl.get("road_network", {}).get("type", "RoadSegment"),
                "min_lanes": max(1, dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 1))),
                "has_traffic_light": False,
                "needs_opposing_lanes": False,
                "needs_long_straight": True,
            },
        }

    def _expand_inattention_front_condition(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        actors = self._normalize_actors(dsl.get("actors", []))
        if not actors:
            actors = [{"id": "ego", "role": "ego", "path": "straight", "action": "Move Forward", "speed_kmh": 35}]
        actors[0]["role"] = actors[0].get("role", "ego")
        obstacle = {"id": "obstacle", "role": "obstacle", "path": "ahead", "action": "Static Obstacle", "speed_kmh": 0}
        road_type = dsl.get("road_network", {}).get("type", "RoadSegment")
        needs_long_straight = road_type in {"RoadSegment", "Straight"}
        return {
            "violation_type": "未注意前方路况",
            "subject_mode": "single_obstacle",
            "actors": [actors[0], obstacle],
            "conflict": {
                "type": dsl.get("conflict_type", "前方静止异常车辆未被及时响应"),
                "location": dsl.get("conflict_location", road_type),
                "trigger_condition": dsl.get("trigger_condition", "ego 接近前方静止异常目标时未及时减速"),
                "timing": {"time_gap_to_conflict_s": dsl.get("timing", {}).get("time_gap_to_conflict_s", [0.0, 3.0])},
            },
            "road_requirement": {
                "type": road_type,
                "min_lanes": max(1, dsl.get("road_network", {}).get("min_lanes", dsl.get("road_network", {}).get("lanes", 1))),
                "has_traffic_light": bool(dsl.get("road_network", {}).get("has_traffic_light", False)),
                "needs_opposing_lanes": False,
                "needs_long_straight": needs_long_straight,
                "min_segment_length_m": 25.0 if needs_long_straight else 12.0,
            },
        }

    def _unwrap_organized_scenario(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "dsl" not in payload:
            return payload
        dsl = dict(payload.get("dsl") or {})
        dsl["violation_type"] = payload.get("violation_type", dsl.get("violation_type"))
        dsl["likelihood"] = payload.get("likelihood")
        dsl["semantic_reason"] = payload.get("reason", "")
        dsl["law"] = payload.get("law", "")
        return dsl

    def _with_semantic_context(self, spec: Dict[str, Any], dsl: Dict[str, Any]) -> Dict[str, Any]:
        if dsl.get("semantic_reason") or dsl.get("law") or dsl.get("likelihood") is not None:
            spec["semantic_context"] = {
                "reason": dsl.get("semantic_reason", ""),
                "law": dsl.get("law", ""),
                "likelihood": dsl.get("likelihood"),
            }
        return spec

    def _normalize_actors(self, actors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        default_roles = ["violator", "priority", "npc", "npc"]
        for index, actor in enumerate(actors):
            normalized.append(
                {
                    "id": actor.get("id", chr(ord("A") + index)),
                    "role": actor.get("role", default_roles[index] if index < len(default_roles) else "npc"),
                    "path": actor.get("path", actor.get("initial_position", "unknown")),
                    "action": actor.get("action", actor.get("actions", "Move Forward")),
                    "speed_kmh": actor.get("speed", actor.get("init_speed_kmh", actor.get("speed_limit", 30))),
                }
            )
        return normalized

    def _generic_requirement(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        road = dsl.get("road_network", {})
        lanes = road.get("min_lanes", road.get("lanes", 1))
        return {
            "type": road.get("type", "RoadSegment"),
            "min_lanes": max(1, int(lanes)),
            "has_traffic_light": road.get("has_traffic_light", False),
            "needs_opposing_lanes": road.get("needs_opposing_lanes", False),
        }
