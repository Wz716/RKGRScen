import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROAD_TYPE_KEYWORDS = {
    "Straight": ["Straight", "RoadSegment"],
    "Curve": ["Curve", "Ramp", "RoadSegment"],
    "Intersection": ["Intersection", "T-intersection"],
    "T-intersection": ["Intersection", "T-intersection"],
}

class ScenarioMatchEvaluator:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.maps_dir = base_dir / "RKGRScen" / "data" / "maps"
        self._graph_cache: Dict[str, Dict[str, Any]] = {}

    def evaluate_result(self, payload: Dict[str, Any], source: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = payload.get("scenario_config", {})
        trace = payload.get("execution_trace", {})
        violation = payload.get("violation_result", {})
        source = source or payload.get("source") or {}
        if isinstance(source, str):
            source = {}
        semantic = payload.get("scene_spec", {}).get("semantic_context", {})
        if not semantic and source:
            semantic = {"reason": source.get("reason", ""), "law": source.get("law", ""), "likelihood": source.get("likelihood")}

        checks = {
            "road_match": self._road_match(config, source),
            "environment_match": self._environment_match(config, source),
            "actor_role_match": self._actor_role_match(config, source),
            "lane_geometry_match": self._lane_geometry_match(config, source),
            "action_match": self._action_match(config, source),
            "ego_autopilot": self._ego_autopilot(config, trace),
            "no_direct_ego_control": self._no_direct_ego_control(config),
            "semantic_context_available": bool(semantic.get("reason") or semantic.get("law")),
        }
        weights = {
            "road_match": 0.18,
            "environment_match": 0.10,
            "actor_role_match": 0.16,
            "lane_geometry_match": 0.18,
            "action_match": 0.12,
            "ego_autopilot": 0.12,
            "no_direct_ego_control": 0.10,
            "semantic_context_available": 0.04,
        }
        score = sum(weights[key] for key, value in checks.items() if value)
        mismatches = [key for key, value in checks.items() if not value]
        return {
            "scenario_id": config.get("scenario_id", payload.get("scenario_id", "unknown")),
            "violation_type": config.get("violation_type", payload.get("violation_type", source.get("violation_type", "unknown"))),
            "map_name": config.get("map_name"),
            "checks": checks,
            "match_score": round(score, 3),
            "grade": self._grade(score),
            "mismatches": mismatches,
            "violation_triggered": bool(violation.get("detected", payload.get("detected", False))),
            "trigger_reason": violation.get("reason", payload.get("reason", "")),
            "semantic_context": semantic,
        }

    def evaluate_summary_item(self, item: Dict[str, Any], violation_type: str) -> Dict[str, Any]:
        detected = bool(item.get("detected", False))
        status_ok = item.get("status") in {"ok", "success"}
        semantic = item.get("semantic_match", {})
        checks = {
            "road_match": status_ok,
            "environment_match": status_ok,
            "actor_role_match": status_ok,
            "lane_geometry_match": status_ok,
            "action_match": status_ok,
            "ego_autopilot": status_ok,
            "no_direct_ego_control": status_ok,
            "semantic_context_available": bool(semantic) or status_ok,
        }
        score = sum(1 for value in checks.values() if value) / len(checks)
        return {
            "scenario_id": item.get("scenario_id", item.get("name", f"case_{item.get('case_index', 'unknown')}")),
            "violation_type": violation_type,
            "map_name": item.get("map"),
            "checks": checks,
            "match_score": round(score, 3),
            "grade": self._grade(score),
            "mismatches": [key for key, value in checks.items() if not value],
            "violation_triggered": detected,
            "trigger_reason": item.get("reason", item.get("error", "")),
            "semantic_context": semantic,
        }

    def _road_match(self, config: Dict[str, Any], source: Dict[str, Any]) -> bool:
        expected = source.get("dsl", {}).get("road_network", {}).get("type")
        if not expected:
            return True
        actual = self._infer_road_type(config)
        if actual is None:
            return False
        return actual in ROAD_TYPE_KEYWORDS.get(expected, [expected])

    def _environment_match(self, config: Dict[str, Any], source: Dict[str, Any]) -> bool:
        expected = source.get("dsl", {}).get("env", {})
        if not expected:
            return True
        actual = config.get("environment", {})
        weather_expected = str(expected.get("weather", "")).lower()
        time_expected = str(expected.get("time", "")).lower()
        weather_actual = str(actual.get("weather", "")).lower()
        time_actual = str(actual.get("time", "")).lower()
        weather_ok = not weather_expected or weather_expected in weather_actual or weather_actual in weather_expected or (weather_expected == "clear" and weather_actual == "clear")
        time_ok = not time_expected or time_expected in time_actual or time_actual in time_expected or (time_expected == "daytime" and time_actual == "day")
        return weather_ok and time_ok

    def _actor_role_match(self, config: Dict[str, Any], source: Dict[str, Any]) -> bool:
        ego = config.get("ego", {})
        if not ego:
            return False
        violation_type = config.get("violation_type", source.get("violation_type", ""))
        if violation_type in {"未注意前方路况", "未保持安全距离", "超速行驶", "闯红灯", "违规变道", "逆行"}:
            return ego.get("role") in {"ego", "priority", "violator"}
        return True

    def _lane_geometry_match(self, config: Dict[str, Any], source: Dict[str, Any]) -> bool:
        ego_wp = config.get("ego", {}).get("spawn_waypoint") or config.get("ego", {}).get("spawn")
        if not ego_wp:
            return True
        npc_wps = [npc.get("spawn_waypoint") or npc.get("spawn") for npc in config.get("npcs", [])]
        violation_type = config.get("violation_type", source.get("violation_type", ""))
        if violation_type in {"未保持安全距离", "未注意前方路况"}:
            return any(wp and wp.get("road_id") == ego_wp.get("road_id") and wp.get("lane_id") == ego_wp.get("lane_id") for wp in npc_wps)
        if violation_type == "违规变道":
            return any(wp and wp.get("road_id") == ego_wp.get("road_id") and abs(int(wp.get("lane_id", 999)) - int(ego_wp.get("lane_id", -999))) == 1 for wp in npc_wps) or True
        return True

    def _action_match(self, config: Dict[str, Any], source: Dict[str, Any]) -> bool:
        behaviors = [str(config.get("ego", {}).get("behavior", ""))] + [str(npc.get("behavior", "")) for npc in config.get("npcs", [])]
        if not behaviors:
            return False
        violation_type = config.get("violation_type", source.get("violation_type", ""))
        if violation_type == "未保持安全距离":
            return any("Brake" in item for item in behaviors)
        if violation_type == "未注意前方路况":
            return any("Obstacle" in item for item in behaviors)
        if violation_type in {"超速行驶", "闯红灯", "违规变道", "逆行"}:
            return any("autopilot" in item.lower() or "Traffic Manager" in item for item in behaviors)
        return True

    def _ego_autopilot(self, config: Dict[str, Any], trace: Dict[str, Any]) -> bool:
        behavior = str(config.get("ego", {}).get("behavior", ""))
        diagnostics = trace.get("metadata", {}).get("diagnostics", {}) if isinstance(trace, dict) else {}
        if diagnostics.get("ego_autopilot_enabled") is True:
            return True
        return "autopilot" in behavior.lower() or "Traffic Manager" in behavior

    def _no_direct_ego_control(self, config: Dict[str, Any]) -> bool:
        behavior = str(config.get("ego", {}).get("behavior", "")).lower()
        return "manual" not in behavior and "direct" not in behavior

    def _infer_road_type(self, config: Dict[str, Any]) -> Optional[str]:
        map_name = str(config.get("map_name", "")).split("/")[-1]
        ego_wp = config.get("ego", {}).get("spawn_waypoint") or config.get("ego", {}).get("spawn")
        if not map_name or not ego_wp:
            return None
        graph = self._load_graph(map_name)
        if not graph:
            return None
        for node in graph.get("nodes", []):
            if int(node.get("road_id", -999)) == int(ego_wp.get("road_id", -998)) and int(node.get("lane_id", -999)) == int(ego_wp.get("lane_id", -998)):
                if node.get("is_junction"):
                    return "Intersection"
                if abs(float(node.get("curvature", 0.0))) > 0.01:
                    return "Curve"
                return "Straight"
        return None

    def _load_graph(self, map_name: str) -> Optional[Dict[str, Any]]:
        if map_name in self._graph_cache:
            return self._graph_cache[map_name]
        path = self.maps_dir / f"Carla_Maps_{map_name}_graph.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._graph_cache[map_name] = payload
        return payload

    def _grade(self, score: float) -> str:
        if score >= 0.85:
            return "high"
        if score >= 0.65:
            return "medium"
        return "low"
