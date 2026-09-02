from typing import Any, Dict, List, Tuple

from RKGRScen.models import RetrievalResult, ScenarioConfiguration

class ConstraintSolver:

    def solve(self, scenario_spec: Dict[str, Any], retrieval_results: List[RetrievalResult]) -> ScenarioConfiguration:
        if not retrieval_results:
            raise ValueError("没有可用的检索结果，无法实例化场景")
        if scenario_spec["violation_type"] == "未按规定让行":
            return self._solve_yield_violation(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "闯红灯":
            return self._solve_red_light_violation(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "违规变道":
            return self._solve_lane_change_violation(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "违规超车":
            return self._solve_overtake_violation(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "逆行":
            return self._solve_wrong_way_violation(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "未注意前方路况":
            return self._solve_inattention_front_condition(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "未保持安全距离":
            return self._solve_following_distance_violation(scenario_spec, retrieval_results[0])
        if scenario_spec["violation_type"] == "超速":
            return self._solve_speeding_violation(scenario_spec, retrieval_results[0])
        return self._solve_generic(scenario_spec, retrieval_results[0])

    def _solve_speeding_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        node = self._select_long_straight_node(retrieval.matched_nodes)
        actors = spec.get("actors", [])
        ego_actor = self._find_actor(actors, ["ego", "victim", "observer"]) or (actors[0] if actors else {})
        speed_limit = float(spec.get("road_requirement", {}).get("speed_limit_kmh", node.get("speed_limit", 40.0)) or 40.0)
        target_speed = float(spec.get("conflict", {}).get("target_speed_kmh", max(speed_limit + 15.0, 55.0)))
        return ScenarioConfiguration(
            scenario_id="speeding_npc_demo",
            violation_type="超速",
            map_name=retrieval.community.map_name,
            ego={
                "role": ego_actor.get("role", "observer"),
                "spawn_waypoint": self._spawn_waypoint(node, 6.0),
                "init_speed_kmh": round(max(float(ego_actor.get("speed_kmh", 30)), min(speed_limit, 35.0)), 1),
                "behavior": "Move Forward (autopilot)",
            },
            npcs=[
                {
                    "id": "speeding_npc",
                    "role": "violator",
                    "spawn_waypoint": self._spawn_waypoint(node, 24.0),
                    "init_speed_kmh": round(target_speed, 1),
                    "behavior": "Move Forward Speeding",
                    "trigger_time_s": 0.0,
                    "target_speed_kmh": round(target_speed, 1),
                }
            ],
            conflict_point=node.get("end", {"x": 0.0, "y": 0.0}),
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "超速",
                "detector": "npc_speeding_violation_detector",
                "params": {
                    "speed_limit_kmh": speed_limit,
                    "target_speed_kmh": target_speed,
                    "subject": "npc",
                    "violator_role": "violator",
                },
            },
        )

    def _solve_yield_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        nodes = retrieval.matched_nodes
        if len(nodes) < 2:
            raise ValueError("未按规定让行至少需要两个候选车道段")
        actors = spec.get("actors", [])
        violator_actor = self._find_actor(actors, ["violator", "subject"]) or actors[0]
        priority_actor = self._find_actor(actors, ["priority", "ego"]) or actors[1]
        violator_node, priority_node = self._select_opposing_nodes(nodes, violator_actor, priority_actor, spec)
        conflict_point = self._nearest_intersection_point(priority_node, violator_node)
        timing = spec.get("conflict", {}).get("timing", {}).get("time_gap_to_conflict_s", [0.5, 2.0])
        params = spec.get("parameter_hint", {})
        if params:
            priority_speed = max(float(priority_actor.get("speed_kmh", 35)), 28.0)
            violator_speed = max(float(violator_actor.get("speed_kmh", 30)), priority_speed + 6.0)
            priority_node = {"road_id": params["priority_road_id"], "lane_id": params["priority_lane_id"], "start": {"x": 0, "y": 0}, "end": {"x": 0, "y": 0}}
            violator_node = {"road_id": params["violator_road_id"], "lane_id": params["violator_lane_id"], "start": {"x": 0, "y": 0}, "end": {"x": 0, "y": 0}}
            conflict_point = params["conflict_point"]
            priority_spawn_s = float(params.get("priority_s", 6.0))
            violator_spawn_s = float(params.get("violator_s", 6.0))
            trigger_time = float(params.get("trigger_time_s", 0.2))
            return ScenarioConfiguration(
                scenario_id="yield_violation_demo",
                violation_type="未按规定让行",
                map_name=retrieval.community.map_name,
                ego={
                    "role": priority_actor.get("role", "priority"),
                    "spawn_waypoint": self._spawn_waypoint(priority_node, priority_spawn_s),
                    "init_speed_kmh": round(priority_speed, 1),
                    "behavior": f"{priority_actor.get('action', 'Move Forward')} (autopilot)",
                },
                npcs=[
                    {
                        "id": violator_actor.get("id", "A"),
                        "role": violator_actor.get("role", "violator"),
                        "spawn_waypoint": self._spawn_waypoint(violator_node, violator_spawn_s),
                        "init_speed_kmh": round(violator_speed, 1),
                        "behavior": violator_actor.get("action", "Turn Left"),
                        "trigger_time_s": trigger_time,
                        "target_conflict_point": conflict_point,
                        "aggressiveness": "high",
                    }
                ],
                conflict_point=conflict_point,
                environment={"weather": "clear", "time": "day"},
                expected_violation={
                    "type": "未按规定让行",
                    "detector": "yield_violation_detector",
                    "params": {"time_gap_range_s": timing, "danger_distance_m": float(params.get("danger_distance_m", 10.0))},
                },
            )

        priority_speed = max(float(priority_actor.get("speed_kmh", 35)), 40.0)
        violator_speed = max(float(violator_actor.get("speed_kmh", 30)), priority_speed + 10.0)
        trigger_time = round(max(0.1, timing[0] * 0.3), 2)
        priority_spawn_s = self._choose_spawn_s(priority_node, conflict_point, preferred=14.0)
        violator_spawn_s = self._choose_spawn_s(violator_node, conflict_point, preferred=12.0)

        return ScenarioConfiguration(
            scenario_id="yield_violation_demo",
            violation_type="未按规定让行",
            map_name=retrieval.community.map_name,
            ego={
                "role": priority_actor.get("role", "priority"),
                "spawn_waypoint": self._spawn_waypoint(priority_node, priority_spawn_s),
                "init_speed_kmh": round(priority_speed, 1),
                "behavior": f"{priority_actor.get('action', 'Move Forward')} (autopilot)",
            },
            npcs=[
                {
                    "id": violator_actor.get("id", "A"),
                    "role": violator_actor.get("role", "violator"),
                    "spawn_waypoint": self._spawn_waypoint(violator_node, violator_spawn_s),
                    "init_speed_kmh": round(violator_speed, 1),
                    "behavior": violator_actor.get("action", "Turn Left"),
                    "trigger_time_s": trigger_time,
                    "target_conflict_point": conflict_point,
                    "aggressiveness": "high",
                }
            ],
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "未按规定让行",
                "detector": "yield_violation_detector",
                "params": {"time_gap_range_s": timing},
            },
        )

    def _solve_red_light_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        nodes = retrieval.matched_nodes
        if len(nodes) < 2:
            raise ValueError("闯红灯至少需要两个候选车道段")

        actors = spec.get("actors", [])
        violator_actor = self._find_actor(actors, ["violator", "ego"]) or actors[0]
        support_actor = self._find_actor(actors[1:], ["priority", "npc"]) if len(actors) > 1 else {}

        violator_node, support_node, conflict_point = self._select_red_light_pair(nodes, violator_actor, support_actor or {})
        stop_line_point = self._stop_line_point(violator_node, conflict_point)
        violator_speed = max(float(violator_actor.get("speed_kmh", 35)), 42.0)
        ego_spawn_s = self._spawn_s_toward_point(violator_node, stop_line_point, toward_end=True)

        npcs = []
        if support_actor:
            npc_spawn_s = self._spawn_s_toward_point(support_node, conflict_point, toward_end=True)
            npcs.append(
                {
                    "id": support_actor.get("id", "B"),
                    "role": support_actor.get("role", "priority"),
                    "spawn_waypoint": self._spawn_waypoint(support_node, npc_spawn_s),
                    "init_speed_kmh": round(max(float(support_actor.get("speed_kmh", 30)), 28.0), 1),
                    "behavior": support_actor.get("action", "Move Forward"),
                    "trigger_time_s": 0.2,
                    "target_conflict_point": conflict_point,
                    "aggressiveness": "medium",
                }
            )

        return ScenarioConfiguration(
            scenario_id="red_light_violation_demo",
            violation_type="闯红灯",
            map_name=retrieval.community.map_name,
            ego={
                "role": violator_actor.get("role", "violator"),
                "spawn_waypoint": self._spawn_waypoint(violator_node, ego_spawn_s),
                "init_speed_kmh": round(violator_speed, 1),
                "behavior": "Move Forward (manual)",
            },
            npcs=npcs,
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "闯红灯",
                "detector": "red_light_detector",
                "params": {
                    "signal_phase": "red",
                    "stop_line_buffer_m": 2.0,
                    "stop_line_point": stop_line_point,
                },
            },
        )

    def _solve_lane_change_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        nodes = retrieval.matched_nodes
        if len(nodes) < 2:
            raise ValueError("违规变道至少需要两个候选车道段")
        actors = spec.get("actors", [])
        violator_actor = self._find_actor(actors, ["violator", "ego"]) or actors[0]
        priority_actor = self._find_actor(actors[1:], ["priority", "npc"]) or actors[1]
        source_node, target_node = self._select_same_direction_adjacent_nodes(nodes, spec)
        conflict_point = self._nearest_intersection_point(source_node, target_node)
        timing = spec.get("conflict", {}).get("timing", {}).get("time_gap_to_conflict_s", [0.3, 1.2])
        violator_speed = max(float(violator_actor.get("speed_kmh", 45)), 38.0)
        priority_speed = max(float(priority_actor.get("speed_kmh", 45)), 32.0)
        dsl_lane_change_direction = spec.get("road_requirement", {}).get("lane_change_direction", "left")
        source_lane_id = int(source_node["lane_id"])
        target_lane_id = int(target_node["lane_id"])
        lane_change_direction = self._lane_change_direction(source_lane_id, target_lane_id)

        return ScenarioConfiguration(
            scenario_id="lane_change_violation_demo",
            violation_type="违规变道",
            map_name=retrieval.community.map_name,
            ego={
                "role": priority_actor.get("role", "priority"),
                "spawn_waypoint": self._spawn_waypoint(target_node, 14.0),
                "init_speed_kmh": round(priority_speed, 1),
                "behavior": "Move Forward (autopilot)",
            },
            npcs=[
                {
                    "id": violator_actor.get("id", "A"),
                    "role": violator_actor.get("role", "violator"),
                    "spawn_waypoint": self._spawn_waypoint(source_node, 12.0),
                    "init_speed_kmh": round(violator_speed, 1),
                    "behavior": violator_actor.get("action", "Change Lane Left"),
                    "trigger_time_s": round(max(0.2, timing[0]), 2),
                    "target_conflict_point": conflict_point,
                    "target_lane_change": lane_change_direction,
                    "dsl_lane_change_direction": dsl_lane_change_direction,
                    "source_lane_id": source_lane_id,
                    "target_lane_id": target_lane_id,
                    "aggressiveness": "high",
                    "longitudinal_gap_m": 6.0,
                }
            ],
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "违规变道",
                "detector": "lane_change_violation_detector",
                "params": {
                    "lane_change_direction": lane_change_direction,
                    "dsl_lane_change_direction": dsl_lane_change_direction,
                    "source_lane_id": source_lane_id,
                    "target_lane_id": target_lane_id,
                    "min_lateral_shift_m": 1.8,
                    "danger_gap_m": 8.0,
                    "requires_target_lane_entry": True,
                },
            },
        )

    def _solve_overtake_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        nodes = retrieval.matched_nodes
        if len(nodes) < 2:
            raise ValueError("违规超车至少需要两个候选车道段")
        actors = spec.get("actors", [])
        violator_actor = self._find_actor(actors, ["violator", "ego"]) or actors[0]
        priority_actor = self._find_actor(actors[1:], ["priority", "npc"]) or actors[1]
        source_node, target_node = self._select_same_direction_adjacent_nodes(nodes, spec)
        conflict_point = self._nearest_intersection_point(source_node, target_node)
        timing = spec.get("conflict", {}).get("timing", {}).get("time_gap_to_conflict_s", [0.2, 1.5])
        violator_speed = max(float(violator_actor.get("speed_kmh", 55)), 42.0)
        priority_speed = max(float(priority_actor.get("speed_kmh", 40)), 28.0)
        violator_target_speed = max(70.0, priority_speed + 25.0)
        dsl_lane_change_direction = spec.get("road_requirement", {}).get("lane_change_direction", "left")
        source_lane_id = int(source_node["lane_id"])
        target_lane_id = int(target_node["lane_id"])
        lane_change_direction = self._lane_change_direction(source_lane_id, target_lane_id)
        return ScenarioConfiguration(
            scenario_id="overtake_violation_demo",
            violation_type="违规超车",
            map_name=retrieval.community.map_name,
            ego={
                "role": priority_actor.get("role", "priority"),
                "spawn_waypoint": self._spawn_waypoint(target_node, 18.0),
                "init_speed_kmh": round(priority_speed, 1),
                "behavior": "Move Forward (autopilot)",
            },
            npcs=[
                {
                    "id": violator_actor.get("id", "A"),
                    "role": violator_actor.get("role", "violator"),
                    "spawn_waypoint": self._spawn_waypoint(source_node, 12.0),
                    "init_speed_kmh": round(violator_speed, 1),
                    "target_speed_kmh": round(violator_target_speed, 1),
                    "behavior": f"Overtake {lane_change_direction.title()}",
                    "source_action": violator_actor.get("action", "Change Lane Left"),
                    "trigger_time_s": round(max(0.15, timing[0]), 2),
                    "target_conflict_point": conflict_point,
                    "target_lane_change": lane_change_direction,
                    "dsl_lane_change_direction": dsl_lane_change_direction,
                    "source_lane_id": source_lane_id,
                    "target_lane_id": target_lane_id,
                    "aggressiveness": "high",
                    "longitudinal_gap_m": 5.0,
                }
            ],
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "违规超车",
                "detector": "overtake_violation_detector",
                "params": {
                    "lane_change_direction": lane_change_direction,
                    "dsl_lane_change_direction": dsl_lane_change_direction,
                    "source_lane_id": source_lane_id,
                    "target_lane_id": target_lane_id,
                    "target_speed_kmh": violator_target_speed,
                    "priority_speed_kmh": priority_speed,
                    "min_lateral_shift_m": 1.8,
                    "danger_gap_m": 10.0,
                    "behind_threshold_m": 3.0,
                    "ahead_threshold_m": 3.0,
                    "requires_pass_sequence": True,
                },
            },
        )

    def _solve_wrong_way_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        nodes = retrieval.matched_nodes
        if len(nodes) < 2:
            raise ValueError("逆行至少需要两个候选车道段")
        actors = spec.get("actors", [])
        violator_actor = self._find_actor(actors, ["violator", "ego"]) or actors[0]
        priority_actor = self._find_actor(actors[1:], ["priority", "npc"]) or actors[1]
        violator_node, priority_node = self._select_opposing_nodes(nodes, violator_actor, priority_actor, spec)
        wrong_way_node = priority_node
        conflict_point = self._interpolate(priority_node.get("start", {"x": 0.0, "y": 0.0}), priority_node.get("end", {"x": 0.0, "y": 0.0}), 0.5)
        return ScenarioConfiguration(
            scenario_id="wrong_way_violation_demo",
            violation_type="逆行",
            map_name=retrieval.community.map_name,
            ego={
                "role": priority_actor.get("role", "priority"),
                "spawn_waypoint": self._spawn_waypoint(priority_node, 6.0),
                "init_speed_kmh": round(max(float(priority_actor.get("speed_kmh", 35)), 30.0), 1),
                "behavior": "Move Forward (autopilot)",
            },
            npcs=[
                {
                    "id": violator_actor.get("id", "A"),
                    "role": violator_actor.get("role", "violator"),
                    "spawn_waypoint": self._spawn_waypoint(wrong_way_node, 18.0),
                    "init_speed_kmh": round(max(float(violator_actor.get("speed_kmh", 35)), 30.0), 1),
                    "behavior": "Move Forward",
                    "reverse_spawn_heading": True,
                    "trigger_time_s": 0.0,
                    "target_conflict_point": conflict_point,
                    "aggressiveness": "medium",
                }
            ],
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "逆行",
                "detector": "wrong_way_detector",
                "params": {
                    "danger_distance_m": 12.5,
                    "heading_opposition_deg": 120.0,
                    "requires_ego_forward_wrong_lane": True,
                },
            },
        )

    def _solve_following_distance_violation(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        node = self._select_long_straight_node(retrieval.matched_nodes)
        actors = spec.get("actors", [])
        ego_actor = self._find_actor(actors, ["ego", "violator", "subject"]) or (actors[0] if actors else {})
        lead_actor = self._find_actor(actors, ["lead", "npc", "front"]) or (actors[1] if len(actors) > 1 else {})
        lane_length = self._point_distance(node.get("start", {}), node.get("end", {}))
        params = spec.get("parameter_hint", {})
        ego_s = round(float(params.get("ego_s", max(20.0, min(80.0, lane_length * 0.18)))), 2)
        lead_s = round(float(params.get("lead_s", min(lane_length * 0.78, ego_s + 30.0))), 2)
        brake_after_s = float(params.get("brake_after_s", 8.0))
        conflict_point = self._interpolate(node.get("start", {"x": 0.0, "y": 0.0}), node.get("end", {"x": 0.0, "y": 0.0}), lead_s / max(lane_length, 1.0))
        return ScenarioConfiguration(
            scenario_id="following_distance_violation_demo",
            violation_type="未保持安全距离",
            map_name=retrieval.community.map_name,
            ego={
                "role": ego_actor.get("role", "ego"),
                "spawn_waypoint": self._spawn_waypoint(node, ego_s),
                "init_speed_kmh": round(max(float(ego_actor.get("speed_kmh", 35)), 35.0), 1),
                "behavior": "Move Forward (autopilot)",
            },
            npcs=[
                {
                    "id": lead_actor.get("id", "lead"),
                    "role": lead_actor.get("role", "lead"),
                    "spawn_waypoint": self._spawn_waypoint(node, lead_s),
                    "init_speed_kmh": round(max(float(lead_actor.get("speed_kmh", 20)), 20.0), 1),
                    "behavior": "Lead Brake",
                    "trigger_time_s": 0.0,
                    "brake_after_s": brake_after_s,
                    "target_conflict_point": conflict_point,
                    "aggressiveness": "medium",
                }
            ],
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "未保持安全距离",
                "detector": "following_distance_detector",
                "params": {
                    "thw_threshold_s": 1.0,
                    "min_gap_m": 8.0,
                    "rss_response_time_s": 1.0,
                    "rss_ego_accel_max_mps2": 2.0,
                    "rss_ego_brake_min_mps2": 4.0,
                    "rss_front_brake_max_mps2": 8.0,
                },
            },
        )

    def _solve_inattention_front_condition(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        node = self._select_long_straight_node(retrieval.matched_nodes)
        actors = spec.get("actors", [])
        ego_actor = self._find_actor(actors, ["ego", "violator", "subject"]) or (actors[0] if actors else {})
        lane_length = self._point_distance(node.get("start", {}), node.get("end", {}))
        params = spec.get("parameter_hint", {})
        ego_s = round(float(params.get("ego_s", max(20.0, min(60.0, lane_length * 0.18)))), 2)
        obstacle_s = round(float(params.get("obstacle_s", min(lane_length * 0.75, ego_s + 35.0))), 2)
        conflict_point = self._interpolate(node.get("start", {"x": 0.0, "y": 0.0}), node.get("end", {"x": 0.0, "y": 0.0}), obstacle_s / max(lane_length, 1.0))
        return ScenarioConfiguration(
            scenario_id="inattention_front_condition_demo",
            violation_type="未注意前方路况",
            map_name=retrieval.community.map_name,
            ego={
                "role": ego_actor.get("role", "ego"),
                "spawn_waypoint": self._spawn_waypoint(node, ego_s),
                "init_speed_kmh": round(max(float(ego_actor.get("speed_kmh", 35)), 35.0), 1),
                "behavior": "Move Forward (autopilot)",
            },
            npcs=[
                {
                    "id": "obstacle",
                    "role": "obstacle",
                    "spawn_waypoint": self._spawn_waypoint(node, obstacle_s),
                    "init_speed_kmh": 0.0,
                    "behavior": "Static Obstacle",
                    "trigger_time_s": 0.0,
                    "target_conflict_point": conflict_point,
                    "aggressiveness": "none",
                }
            ],
            conflict_point=conflict_point,
            environment={"weather": "clear", "time": "day"},
            expected_violation={
                "type": "未注意前方路况",
                "detector": "inattention_front_condition_detector",
                "params": {
                    "ttc_threshold_s": 3.0,
                    "danger_distance_m": 15.0,
                    "rss_response_time_s": 1.0,
                    "rss_ego_accel_max_mps2": 2.0,
                    "rss_ego_brake_min_mps2": 4.0,
                    "rss_front_brake_max_mps2": 8.0,
                },
            },
        )

    def _solve_generic(self, spec: Dict[str, Any], retrieval: RetrievalResult) -> ScenarioConfiguration:
        node = retrieval.matched_nodes[0]
        return ScenarioConfiguration(
            scenario_id=f"{spec['violation_type']}_demo",
            violation_type=spec["violation_type"],
            map_name=retrieval.community.map_name,
            ego={
                "role": "ego",
                "spawn_waypoint": self._spawn_waypoint(node, 5.0),
                "init_speed_kmh": spec["actors"][0].get("speed_kmh", 30) if spec.get("actors") else 30,
                "behavior": spec["actors"][0].get("action", "Move Forward") if spec.get("actors") else "Move Forward",
            },
            npcs=[],
            conflict_point=node.get("end", {"x": 0.0, "y": 0.0}),
            environment={"weather": "clear", "time": "day"},
            expected_violation={"type": spec["violation_type"], "detector": "placeholder", "params": {}},
        )

    def _select_long_straight_node(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = nodes[0]
        best_score = -1e9
        for node in nodes:
            if node.get("is_junction"):
                continue
            start = node.get("start", {})
            end = node.get("end", {})
            length = self._point_distance(start, end)
            curvature = abs(float(node.get("curvature", 0.0) or 0.0))
            score = length - curvature * 20.0
            if score > best_score:
                best_score = score
                best = node
        return best

    def _find_actor(self, actors: List[Dict[str, Any]], roles: List[str]) -> Dict[str, Any]:
        for actor in actors:
            if actor.get("role") in roles:
                return actor
        return {}

    @staticmethod
    def _lane_change_direction(source_lane_id: int, target_lane_id: int) -> str:
        if source_lane_id == 0 or target_lane_id == 0 or source_lane_id * target_lane_id < 0:
            raise ValueError("变道方向要求同向且非零的 source/target lane_id")
        if source_lane_id == target_lane_id:
            raise ValueError("变道方向要求不同的 source/target lane_id")
        if source_lane_id > 0:
            return "right" if target_lane_id > source_lane_id else "left"
        return "left" if abs(target_lane_id) > abs(source_lane_id) else "right"

    def _select_same_direction_adjacent_nodes(self, nodes: List[Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        lane_change_direction = spec.get("road_requirement", {}).get("lane_change_direction", "left")
        best_pair = (nodes[0], nodes[1])
        best_score = -1e9
        for node_a in nodes:
            for node_b in nodes:
                if node_a is node_b:
                    continue
                if node_a.get("road_id") != node_b.get("road_id"):
                    continue
                heading_gap = self._heading_gap(float(node_a.get("heading", 0.0)), float(node_b.get("heading", 0.0)))
                if heading_gap > 20.0:
                    continue
                lane_gap = abs(int(node_a.get("lane_id", 0)) - int(node_b.get("lane_id", 0)))
                if lane_gap != 1:
                    continue
                score = 10.0 - heading_gap - lane_gap
                if lane_change_direction == "left" and int(node_b.get("lane_id", 0)) > int(node_a.get("lane_id", 0)):
                    score += 2.0
                if lane_change_direction == "right" and int(node_b.get("lane_id", 0)) < int(node_a.get("lane_id", 0)):
                    score += 2.0
                if score > best_score:
                    best_score = score
                    best_pair = (node_a, node_b)
        return best_pair

    def _select_red_light_pair(self, nodes: List[Dict[str, Any]], violator_actor: Dict[str, Any], support_actor: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
        best_triplet = (nodes[0], nodes[1], self._nearest_intersection_point(nodes[0], nodes[1]))
        best_score = -1e9
        for node_a in nodes:
            for node_b in nodes:
                if node_a is node_b:
                    continue
                conflict = self._nearest_intersection_point(node_a, node_b)
                score = self._red_light_pair_score(node_a, node_b, conflict, violator_actor, support_actor)
                if score > best_score:
                    best_score = score
                    best_triplet = (node_a, node_b, conflict)
        return best_triplet

    def _red_light_pair_score(self, violator_node: Dict[str, Any], support_node: Dict[str, Any], conflict: Dict[str, float], violator_actor: Dict[str, Any], support_actor: Dict[str, Any]) -> float:
        score = 0.0
        if violator_node.get("has_traffic_light"):
            score += 4.0
        if support_node.get("has_traffic_light"):
            score += 2.0
        if violator_node.get("is_junction"):
            score += 1.5
        if support_node.get("is_junction"):
            score += 1.5
        if violator_node.get("road_id") == support_node.get("road_id"):
            score -= 6.0
        if int(violator_node.get("lane_id", 0)) == int(support_node.get("lane_id", 0)):
            score -= 4.0
        heading_gap = self._heading_gap(float(violator_node.get("heading", 0.0)), float(support_node.get("heading", 0.0)))
        score -= abs(90.0 - heading_gap) * 0.12
        if self._is_crossing_path_pair(str(violator_actor.get("path", "")), str(support_actor.get("path", ""))):
            score += 4.0
        score += self._progress_toward_point(violator_node, conflict) * 0.25
        score += self._progress_toward_point(support_node, conflict) * 0.15
        return score

    def _progress_toward_point(self, node: Dict[str, Any], point: Dict[str, float]) -> float:
        start = node.get("start", {"x": 0.0, "y": 0.0})
        end = node.get("end", {"x": 0.0, "y": 0.0})
        return self._point_distance(start, point) - self._point_distance(end, point)

    def _stop_line_point(self, node: Dict[str, Any], conflict_point: Dict[str, float]) -> Dict[str, float]:
        start = node.get("start", {"x": 0.0, "y": 0.0})
        end = node.get("end", {"x": 0.0, "y": 0.0})
        if self._point_distance(end, conflict_point) <= self._point_distance(start, conflict_point):
            return self._interpolate(start, end, 0.82)
        return self._interpolate(start, end, 0.18)

    def _spawn_s_toward_point(self, node: Dict[str, Any], target_point: Dict[str, float], toward_end: bool = True) -> float:
        start = node.get("start", {"x": 0.0, "y": 0.0})
        end = node.get("end", {"x": 0.0, "y": 0.0})
        lane_length = self._point_distance(start, end)
        if lane_length <= 8.0:
            return 6.0
        progress = self._progress_toward_point(node, target_point)
        if progress >= 0:
            return round(max(6.0, lane_length * 0.22), 2)
        return round(max(6.0, lane_length * 0.68), 2)

    def _select_opposing_nodes(self, nodes: List[Dict[str, Any]], violator_actor: Dict[str, Any], priority_actor: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        best_pair = (nodes[0], nodes[1])
        best_score = -1e9
        road_type = str(spec.get("road_requirement", {}).get("type", "Intersection"))
        for node_a in nodes:
            for node_b in nodes:
                if node_a is node_b:
                    continue
                score = self._pair_score(node_a, node_b, violator_actor, priority_actor, road_type)
                if score > best_score:
                    best_score = score
                    best_pair = (node_a, node_b)
        return best_pair

    def _pair_score(self, violator_node: Dict[str, Any], priority_node: Dict[str, Any], violator_actor: Dict[str, Any], priority_actor: Dict[str, Any], road_type: str) -> float:
        violator_heading = float(violator_node.get("heading", 0.0))
        priority_heading = float(priority_node.get("heading", 0.0))
        heading_gap = abs(abs(violator_heading - priority_heading) - 180.0)
        lane_score = float(violator_node.get("lane_count", 1)) + float(priority_node.get("lane_count", 1))
        score = lane_score - heading_gap * 0.12
        if violator_node.get("is_junction") and priority_node.get("is_junction"):
            score += 2.0
        nearest = self._nearest_intersection_point(violator_node, priority_node)
        violator_proximity = self._point_distance(violator_node.get("end", {}), nearest)
        priority_proximity = self._point_distance(priority_node.get("end", {}), nearest)
        score -= (violator_proximity + priority_proximity) * 0.03
        violator_path = str(violator_actor.get("path", ""))
        priority_path = str(priority_actor.get("path", ""))
        if self._is_opposing_path_pair(violator_path, priority_path):
            score += 4.0
        if "On-ramp" in violator_path and road_type.lower().startswith("t-"):
            score += 3.0
        if "Turn Left" in str(violator_actor.get("action", "")):
            score += 1.5
        if "Move Forward" in str(priority_actor.get("action", "")):
            score += 1.0
        if self._road_id_relation(violator_node, priority_node):
            score += 1.5
        return score

    def _is_opposing_path_pair(self, path_a: str, path_b: str) -> bool:
        opposites = {("S2N", "N2S"), ("N2S", "S2N"), ("W2E", "E2W"), ("E2W", "W2E")}
        return (path_a, path_b) in opposites

    def _is_crossing_path_pair(self, path_a: str, path_b: str) -> bool:
        crossings = {
            ("W2E", "N2S"), ("W2E", "S2N"), ("E2W", "N2S"), ("E2W", "S2N"),
            ("N2S", "W2E"), ("N2S", "E2W"), ("S2N", "W2E"), ("S2N", "E2W"),
        }
        return (path_a, path_b) in crossings

    def _road_id_relation(self, node_a: Dict[str, Any], node_b: Dict[str, Any]) -> bool:
        road_a = node_a.get("road_id")
        road_b = node_b.get("road_id")
        if road_a == road_b:
            return True
        lane_a = int(node_a.get("lane_id", 0))
        lane_b = int(node_b.get("lane_id", 0))
        return lane_a * lane_b < 0

    def _nearest_intersection_point(self, node_a: Dict[str, Any], node_b: Dict[str, Any]) -> Dict[str, float]:
        a_start, a_end = node_a.get("start", {"x": 0.0, "y": 0.0}), node_a.get("end", {"x": 0.0, "y": 0.0})
        b_start, b_end = node_b.get("start", {"x": 0.0, "y": 0.0}), node_b.get("end", {"x": 0.0, "y": 0.0})
        a_points = [a_start, a_end]
        b_points = [b_start, b_end]
        best = (a_points[0], b_points[0])
        best_distance = float("inf")
        for pa in a_points:
            for pb in b_points:
                distance = self._point_distance(pa, pb)
                if distance < best_distance:
                    best_distance = distance
                    best = (pa, pb)
        return {"x": round((best[0]["x"] + best[1]["x"]) / 2.0, 3), "y": round((best[0]["y"] + best[1]["y"]) / 2.0, 3)}

    def _interpolate(self, a: Dict[str, Any], b: Dict[str, Any], ratio: float) -> Dict[str, float]:
        ax, ay = float(a.get("x", 0.0)), float(a.get("y", 0.0))
        bx, by = float(b.get("x", 0.0)), float(b.get("y", 0.0))
        return {"x": round(ax + (bx - ax) * ratio, 3), "y": round(ay + (by - ay) * ratio, 3)}

    def _heading_gap(self, a: float, b: float) -> float:
        raw = abs(a - b) % 360.0
        return min(raw, 360.0 - raw)

    def _choose_spawn_s(self, node: Dict[str, Any], conflict_point: Dict[str, float], preferred: float) -> float:
        start = node.get("start", {"x": 0.0, "y": 0.0})
        end = node.get("end", {"x": 0.0, "y": 0.0})
        lane_length = self._point_distance(start, end)
        if lane_length <= 5.0:
            return 5.0
        start_dist = self._point_distance(start, conflict_point)
        end_dist = self._point_distance(end, conflict_point)
        near_end_bias = 0.35 if start_dist > end_dist else 0.55
        s = lane_length * near_end_bias
        s = min(max(5.0, s), lane_length * 0.85)
        s = min(max(6.0, s), preferred)
        return round(s, 2)

    def _spawn_waypoint(self, node: Dict[str, Any], s: float) -> Dict[str, Any]:
        return {"road_id": node.get("road_id"), "lane_id": node.get("lane_id"), "s": s}

    def _point_distance(self, point_a: Dict[str, Any], point_b: Dict[str, Any]) -> float:
        ax, ay = float(point_a.get("x", 0.0)), float(point_a.get("y", 0.0))
        bx, by = float(point_b.get("x", 0.0)), float(point_b.get("y", 0.0))
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _conflict_point(self, node_a: Dict[str, Any], node_b: Dict[str, Any]) -> Dict[str, float]:
        return self._nearest_intersection_point(node_a, node_b)
