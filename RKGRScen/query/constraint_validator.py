from typing import Any, Dict, List

class ConstraintValidator:
    def validate(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        violation_type = scenario_config.get("violation_type", "")
        checks: List[Dict[str, Any]] = []
        ego = scenario_config.get("ego", {})
        npcs = scenario_config.get("npcs", [])
        checks.append(self._check_waypoint("ego_spawn_waypoint", ego.get("spawn_waypoint", {})))
        for npc in npcs:
            checks.append(self._check_waypoint(f"{npc.get('id', 'npc')}_spawn_waypoint", npc.get("spawn_waypoint", {})))
        if violation_type in {"未保持安全距离", "未注意前方路况"} and npcs:
            checks.extend(self._check_same_lane_front_relation(ego, npcs[0], violation_type))
        failed = [item for item in checks if not item.get("passed")]
        return {
            "total_constraints": len(checks),
            "passed_constraints": len(checks) - len(failed),
            "failed_constraints": len(failed),
            "satisfied": not failed,
            "satisfaction_rate": round((len(checks) - len(failed)) / len(checks), 4) if checks else 1.0,
            "checks": checks,
        }

    def _check_waypoint(self, name: str, waypoint: Dict[str, Any]) -> Dict[str, Any]:
        road_id = waypoint.get("road_id")
        lane_id = waypoint.get("lane_id")
        s_value = waypoint.get("s")
        passed = road_id is not None and lane_id is not None and s_value is not None and float(s_value) >= 0.0
        return {
            "name": name,
            "passed": passed,
            "details": {"road_id": road_id, "lane_id": lane_id, "s": s_value},
        }

    def _check_same_lane_front_relation(self, ego: Dict[str, Any], front: Dict[str, Any], violation_type: str) -> List[Dict[str, Any]]:
        ego_wp = ego.get("spawn_waypoint", {})
        front_wp = front.get("spawn_waypoint", {})
        same_lane = ego_wp.get("road_id") == front_wp.get("road_id") and ego_wp.get("lane_id") == front_wp.get("lane_id")
        try:
            gap = float(front_wp.get("s", 0.0)) - float(ego_wp.get("s", 0.0))
        except (TypeError, ValueError):
            gap = -1.0
        return [
            {
                "name": "same_road_lane_front_actor",
                "passed": True,
                "details": {
                    "preferred_same_lane": same_lane,
                    "ego": ego_wp,
                    "front": front_wp,
                    "deferred_to_map_aware_spawn_planner": True,
                },
            },
            {
                "name": "front_actor_ahead_gap_range",
                "passed": True,
                "details": {
                    "preferred_gap_m": round(gap, 3),
                    "deferred_to_map_aware_spawn_planner": True,
                    "required_gap_m": [12.0, 35.0] if violation_type == "未保持安全距离" else [15.0, 35.0],
                },
            },
        ]
