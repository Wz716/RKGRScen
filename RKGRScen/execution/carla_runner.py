import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    carla = None

from RKGRScen.models import ExecutionTrace, ScenarioConfiguration

class CarlaScenarioRunner:
    def __init__(self, host: str = "localhost", port: int = 2000, timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def run(self, scenario: ScenarioConfiguration) -> ExecutionTrace:
        if carla is None:
            raise RuntimeError("未安装 carla Python API，无法运行真实场景")

        client = carla.Client(self.host, self.port)
        client.set_timeout(self.timeout_s)
        world = client.get_world()
        target_map = scenario.map_name
        load_map = target_map.split("/")[-1]
        if world.get_map().name not in {target_map, load_map}:
            world = client.load_world(load_map)
            time.sleep(2.0)

        road_map = world.get_map()
        spectator = world.get_spectator()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(False)

        blueprints = world.get_blueprint_library()
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0] if blueprints.filter("vehicle.tesla.model3") else blueprints.filter("vehicle.*")[0]
        npc_bp = blueprints.filter("vehicle.audi.a2")[0] if blueprints.filter("vehicle.audi.a2") else blueprints.filter("vehicle.*")[1]
        self._safe_set_blueprint_color(ego_bp, "255,0,0")
        self._safe_set_blueprint_color(npc_bp, "120,120,120")

        actors: List[Any] = []
        ticks: List[Dict[str, Any]] = []
        ego_start: Dict[str, Any] = {}
        ego_end: Dict[str, Any] = {}
        npc_tracks: List[Dict[str, Any]] = []
        diagnostics: Dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "map": scenario.map_name,
            "ego_spawned": False,
            "npc_spawned": [],
            "ego_autopilot_enabled": False,
            "ego_policy": "traffic_manager_autopilot",
            "traffic_light_forced_red": False,
            "red_switch_distance_m": 8.0,
            "red_activated": False,
            "first_red_timestamp_s": None,
            "latest_distance_to_stop_line": None,
            "traffic_light_match_mode": None,
            "candidate_traffic_lights": [],
            "spawn_plan": None,
            "spawn_precheck": [],
            "runtime_error": None,
            "destroyed_actor_seen": False,
        }
        try:
            diagnostics["spawn_plan"] = self._plan_spawn_waypoints(world, scenario)
            ego_wp = self._resolve_waypoint(world, scenario.ego["spawn_waypoint"])
            npc_spawn_items = []
            for npc in scenario.npcs:
                npc_wp = self._resolve_waypoint(world, npc["spawn_waypoint"])
                npc_spawn_items.append((npc, npc_wp))
            diagnostics["spawn_precheck"] = self._spawn_precheck(world, [("ego", ego_wp)] + [(str(npc.get("id", "npc")), wp) for npc, wp in npc_spawn_items])
            failed_precheck = [item for item in diagnostics["spawn_precheck"] if not item.get("ok", False)]
            if failed_precheck:
                raise RuntimeError(f"spawn 合法性预检查失败: {failed_precheck}")

            ego_actor = self._spawn_with_retry(world, ego_bp, ego_wp)
            if ego_actor is None or not self._is_actor_alive(ego_actor):
                raise RuntimeError("ego 生成失败")
            actors.append(ego_actor)
            diagnostics["ego_spawned"] = True
            diagnostics["ego_spawn_transform"] = self._transform_to_dict(self._safe_get_transform(ego_actor))

            self._safe_enable_autopilot(ego_actor, traffic_manager.get_port())
            diagnostics["ego_autopilot_enabled"] = True
            self._safe_ignore_lights(traffic_manager, ego_actor, scenario.violation_type)

            self._safe_update_spectator(spectator, ego_actor)

            npc_actors = []
            for npc, npc_wp in npc_spawn_items:
                npc_actor = self._spawn_with_retry(world, npc_bp, npc_wp, reverse_heading=bool(npc.get("reverse_spawn_heading", False)))
                diagnostics["npc_spawned"].append({"id": npc.get("id"), "spawned": npc_actor is not None and self._is_actor_alive(npc_actor), "spawn_waypoint": npc.get("spawn_waypoint")})
                if npc_actor is None or not self._is_actor_alive(npc_actor):
                    raise RuntimeError(f"NPC {npc.get('id')} 生成失败")
                actors.append(npc_actor)
                self._safe_disable_autopilot(npc_actor)
                npc_state = self._build_npc_state(npc_cfg=npc, spawn_wp=npc_wp)
                npc_actors.append((npc, npc_actor, npc_wp, npc_state))

            duration_s = 20.0
            step_s = 0.2
            start_time = time.time()
            violation_params = scenario.expected_violation.get("params", {})
            dynamic_stop_line = dict(violation_params.get("stop_line_point", scenario.conflict_point))
            red_activated = False
            first_red_timestamp_s = None
            lane_change_baselines = {
                npc_cfg.get("id"): self._copy_transform(self._safe_get_transform(npc_actor))
                for npc_cfg, npc_actor, _, _ in npc_actors
            }
            while time.time() - start_time <= duration_s:
                timestamp_s = round(time.time() - start_time, 3)
                current_distance = None
                required_actors = [ego_actor] + [item[1] for item in npc_actors]
                required_alive = self._all_required_actors_alive(required_actors)
                if not required_alive:
                    diagnostics["destroyed_actor_seen"] = True
                    break
                try:
                    if scenario.violation_type == "闯红灯":
                        dynamic_stop_line = self._estimate_stop_line_from_waypoint_chain(road_map, ego_actor, fallback=dynamic_stop_line)
                        diagnostics["dynamic_stop_line"] = dynamic_stop_line
                        current_distance = self._distance_actor_to_point(ego_actor, dynamic_stop_line)
                        diagnostics["latest_distance_to_stop_line"] = current_distance
                        if current_distance <= float(diagnostics["red_switch_distance_m"]):
                            forced, match_mode, candidates = self._force_front_traffic_light_red(world, road_map, ego_actor, dynamic_stop_line)
                            diagnostics["traffic_light_forced_red"] = diagnostics["traffic_light_forced_red"] or forced
                            diagnostics["candidate_traffic_lights"] = candidates
                            if match_mode:
                                diagnostics["traffic_light_match_mode"] = match_mode
                            if forced and first_red_timestamp_s is None:
                                first_red_timestamp_s = timestamp_s
                            red_activated = red_activated or forced
                        diagnostics["red_activated"] = red_activated
                        diagnostics["first_red_timestamp_s"] = first_red_timestamp_s
                    for npc_cfg, npc_actor, _, npc_state in npc_actors:
                        self._drive_npc(npc_cfg, npc_actor, timestamp_s, road_map, npc_state)

                    snapshot = world.wait_for_tick(seconds=1.0)
                    if snapshot is None:
                        continue
                    self._safe_update_spectator(spectator, ego_actor)
                    sample = self._sample_tick(
                        road_map,
                        timestamp_s,
                        ego_actor,
                        npc_actors,
                        scenario.conflict_point,
                        scenario.violation_type,
                        dynamic_stop_line,
                        violation_params,
                        red_activated,
                        current_distance,
                        first_red_timestamp_s,
                        lane_change_baselines,
                        scenario.ego.get("role", "ego"),
                    )
                except RuntimeError as exc:
                    diagnostics["runtime_error"] = repr(exc)
                    diagnostics["destroyed_actor_seen"] = True
                    break
                if sample is None:
                    break
                ticks.append(sample)
                if scenario.violation_type == "逆行" and self._wrong_way_risk_observed(sample, violation_params):
                    break
                if scenario.violation_type == "未按规定让行" and self._yield_risk_observed(sample, violation_params):
                    break
                if scenario.violation_type in {"未保持安全距离", "未注意前方路况"} and self._front_risk_observed(sample, violation_params):
                    break
                time.sleep(step_s)

            if ticks:
                ego_start = ticks[0]["ego"]
                ego_end = ticks[-1]["ego"]
                for npc in scenario.npcs:
                    npc_id = npc.get("id")
                    track = []
                    for tick in ticks:
                        for row in tick.get("npcs", []):
                            if row.get("id") == npc_id:
                                track.append(row)
                    npc_tracks.append({"id": npc_id, "track": track})

            time.sleep(2.0)
        finally:
            actor_ids = [int(actor.id) for actor in actors]
            if actor_ids:
                client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in actor_ids], True)

        return ExecutionTrace(
            scenario_id=scenario.scenario_id,
            ticks=ticks,
            metadata={
                "runner": "carla_real",
                "map": scenario.map_name,
                "tick_count": len(ticks),
                "ego_start": ego_start,
                "ego_end": ego_end,
                "npc_tracks": npc_tracks,
                "diagnostics": diagnostics,
            },
        )

    def _safe_set_blueprint_color(self, blueprint: Any, color: str) -> None:
        try:
            if blueprint.has_attribute("color"):
                blueprint.set_attribute("color", color)
        except RuntimeError:
            pass

    def _safe_enable_autopilot(self, actor: Any, port: int) -> None:
        if self._is_actor_alive(actor):
            try:
                actor.set_autopilot(True, port)
            except RuntimeError:
                pass

    def _safe_disable_autopilot(self, actor: Any) -> None:
        if self._is_actor_alive(actor):
            try:
                actor.set_autopilot(False)
            except RuntimeError:
                pass

    def _safe_apply_control(self, actor: Any, throttle: float = 0.65, steer: float = 0.0, brake: float = 0.0) -> None:
        if self._is_actor_alive(actor):
            try:
                actor.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False, reverse=False))
            except RuntimeError:
                pass

    def _safe_ignore_lights(self, traffic_manager: Any, actor: Any, violation_type: str) -> None:
        if self._is_actor_alive(actor):
            try:
                traffic_manager.ignore_lights_percentage(actor, 100.0 if violation_type == "闯红灯" else 0.0)
            except RuntimeError:
                pass

    def _force_front_traffic_light_red(self, world: Any, road_map: Any, actor: Any, stop_line_point: Dict[str, Any]) -> Any:
        if not self._is_actor_alive(actor):
            return False, None, []
        try:
            actor_transform = actor.get_transform()
            actor_location = actor_transform.location
            actor_yaw = actor_transform.rotation.yaw
        except RuntimeError:
            return False, None, []

        best_light = None
        best_score = float("inf")
        best_mode = None
        candidates = []
        for tl in world.get_actors().filter("traffic.traffic_light*"):
            try:
                tl_loc = tl.get_transform().location
                dx = tl_loc.x - actor_location.x
                dy = tl_loc.y - actor_location.y
                distance = math.hypot(dx, dy)
                heading_to_light = math.degrees(math.atan2(dy, dx))
                heading_diff = abs(self._normalize_angle_deg(heading_to_light - actor_yaw))
                stop_line_gap = math.hypot(tl_loc.x - float(stop_line_point.get("x", 0.0)), tl_loc.y - float(stop_line_point.get("y", 0.0)))
                score = distance + stop_line_gap * 2.0 + heading_diff * 0.2
                candidate = {
                    "id": getattr(tl, "id", None),
                    "x": round(tl_loc.x, 3),
                    "y": round(tl_loc.y, 3),
                    "distance": round(distance, 3),
                    "heading_diff": round(heading_diff, 3),
                    "stop_line_gap": round(stop_line_gap, 3),
                    "score": round(score, 3),
                    "eligible": distance <= 40.0 and heading_diff <= 80.0,
                }
                candidates.append(candidate)
                if not candidate["eligible"]:
                    continue
                if score < best_score:
                    best_score = score
                    best_light = tl
                    best_mode = "global_match"
            except RuntimeError:
                continue

        candidates.sort(key=lambda item: item["score"])
        candidates = candidates[:12]

        if best_light is not None:
            try:
                best_light.set_state(carla.TrafficLightState.Red)
                best_light.set_red_time(10.0)
                best_light.freeze(True)
                return True, best_mode, candidates
            except RuntimeError:
                pass

        try:
            tl = actor.get_traffic_light()
            if tl is None:
                return False, None, candidates
            tl.set_state(carla.TrafficLightState.Red)
            tl.set_red_time(10.0)
            tl.freeze(True)
            return True, "actor_bound", candidates
        except RuntimeError:
            return False, None, candidates

    def _estimate_stop_line_from_waypoint_chain(self, road_map: Any, actor: Any, fallback: Dict[str, Any]) -> Dict[str, float]:
        if not self._is_actor_alive(actor):
            return fallback
        try:
            transform = actor.get_transform()
            waypoint = road_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except RuntimeError:
            return fallback
        if waypoint is None:
            return fallback
        last_non_junction = waypoint
        current = waypoint
        for _ in range(20):
            if not current.is_junction:
                last_non_junction = current
            nxt = current.next(2.0)
            if not nxt:
                break
            current = nxt[0]
            if current.is_junction:
                break
        loc = last_non_junction.transform.location
        return {"x": round(loc.x, 3), "y": round(loc.y, 3)}

    def _drive_npc(self, npc_cfg: Dict[str, Any], npc_actor: Any, timestamp_s: float, road_map: Any, npc_state: Dict[str, Any]) -> None:
        if not self._is_actor_alive(npc_actor):
            return
        if timestamp_s < float(npc_cfg.get("trigger_time_s", 0.0)):
            npc_state["control_phase"] = "waiting"
            npc_state["control_steer"] = 0.0
            self._safe_apply_control(npc_actor, throttle=0.0, steer=0.0, brake=1.0)
            return
        try:
            transform = npc_actor.get_transform()
            velocity = npc_actor.get_velocity()
            current_wp = road_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except RuntimeError:
            return
        speed_kmh = float(npc_cfg.get("init_speed_kmh", 35.0))
        behavior = str(npc_cfg.get("behavior", "Move Forward"))
        current_speed_kmh = self._vector_speed(velocity) * 3.6
        if "Static Obstacle" in behavior:
            npc_state["control_phase"] = "stopped"
            npc_state["control_steer"] = 0.0
            self._safe_apply_control(npc_actor, throttle=0.0, steer=0.0, brake=1.0)
            return
        if "Lead Brake" in behavior and timestamp_s >= float(npc_cfg.get("brake_after_s", 8.0)):
            npc_state["control_phase"] = "braking"
            npc_state["control_steer"] = 0.0
            self._safe_apply_control(npc_actor, throttle=0.0, steer=0.0, brake=1.0)
            return

        is_overtake = "overtake" in behavior.lower() or "超车" in behavior
        is_lane_change = "Change Lane" in behavior or "Overtake" in behavior
        if is_lane_change:
            target_wp, phase = self._lane_change_target(current_wp, transform, npc_state)
        else:
            target_wp = self._lane_follow_target(current_wp, bool(npc_cfg.get("reverse_spawn_heading", False)))
            phase = "lane_follow"
        if npc_state.get("lane_change_completed_latched", False):
            previous_steer = float(npc_state.get("previous_steer", 0.0))
            decayed_steer = max(previous_steer - 0.08, min(previous_steer + 0.08, 0.0))
            npc_state["previous_steer"] = decayed_steer
            correction_steer = self._path_steer(transform, target_wp, 0.15)
            steer = max(-0.15, min(0.15, decayed_steer + correction_steer * 0.25))
        else:
            steer = self._smooth_path_steer(transform, target_wp, npc_state)
        if is_overtake:
            if npc_state.get("lane_change_completed_latched", False):
                npc_state["overtake_acceleration_started"] = True
                target_speed = max(float(npc_cfg.get("target_speed_kmh", speed_kmh)), speed_kmh)
                throttle, brake = self._speeding_control(current_speed_kmh, target_speed)
            else:
                npc_state["overtake_acceleration_started"] = False
                target_speed = speed_kmh
                throttle, brake = self._speed_control(current_speed_kmh, target_speed)
        elif "Speeding" in behavior:
            target_speed = max(float(npc_cfg.get("target_speed_kmh", speed_kmh)), speed_kmh)
            throttle, brake = self._speeding_control(current_speed_kmh, target_speed)
        else:
            target_speed = speed_kmh
            throttle, brake = self._speed_control(current_speed_kmh, target_speed)
        npc_state["control_phase"] = phase
        npc_state["control_steer"] = steer
        self._safe_apply_control(npc_actor, throttle=throttle, steer=steer, brake=brake)

    def _build_npc_state(self, npc_cfg: Dict[str, Any], spawn_wp: Any) -> Dict[str, Any]:
        state = {
            "mode": "default",
            "previous_steer": 0.0,
            "control_steer": 0.0,
            "control_phase": "waiting",
            "source_lane_id": npc_cfg.get("source_lane_id", getattr(spawn_wp, "lane_id", None)),
            "target_lane_id": npc_cfg.get("target_lane_id"),
            "target_lane_road_id": None,
            "target_lane_stable_ticks": 0,
            "lane_change_completed_latched": False,
            "overtake_acceleration_started": False,
            "sample_previous_yaw_deg": None,
            "sample_previous_timestamp_s": None,
            "sample_previous_steer": None,
            "max_steer_delta": 0.0,
        }
        behavior = str(npc_cfg.get("behavior", ""))
        if "Change Lane" not in behavior and "Overtake" not in behavior:
            return state
        state["mode"] = "lane_change"
        configured_target_lane_id = npc_cfg.get("target_lane_id")
        target_wp, effective_direction = self._target_lane_waypoint(
            spawn_wp,
            configured_target_lane_id,
            str(npc_cfg.get("target_lane_change", "left")),
        )
        state["lane_change_direction"] = effective_direction
        state["effective_lane_change_direction"] = effective_direction
        state["target_lane_id"] = configured_target_lane_id if configured_target_lane_id is not None else getattr(target_wp, "lane_id", None)
        state["target_lane_road_id"] = getattr(target_wp, "road_id", None)
        state["control_phase"] = "approach"
        return state

    def _lane_follow_target(self, waypoint: Any, reverse: bool = False) -> Any:
        if waypoint is None:
            return None
        try:
            candidates = waypoint.previous(8.0) if reverse else waypoint.next(8.0)
        except RuntimeError:
            return waypoint
        return candidates[0] if candidates else waypoint

    def _lane_change_target(self, current_wp: Any, transform: Any, npc_state: Dict[str, Any]) -> Tuple[Any, str]:
        if current_wp is None:
            return None, "lane_change_unavailable"
        target_lane_id = npc_state.get("target_lane_id")
        if target_lane_id is None:
            return self._lane_follow_target(current_wp), "lane_change_unavailable"
        if npc_state.get("lane_change_completed_latched", False):
            target_wp = self._saved_target_lane_waypoint(current_wp, npc_state)
            return self._lane_follow_target(target_wp or current_wp), "completed"
        if current_wp.lane_id == target_lane_id:
            lane_center = current_wp.transform.location
            center_gap = self._distance_2d(transform.location.x, transform.location.y, lane_center.x, lane_center.y)
            if center_gap < 0.8:
                stable_ticks = int(npc_state.get("target_lane_stable_ticks", 0)) + 1
                npc_state["target_lane_stable_ticks"] = stable_ticks
                if stable_ticks >= 5:
                    npc_state["lane_change_completed_latched"] = True
                    return self._lane_follow_target(current_wp), "completed"
            else:
                npc_state["target_lane_stable_ticks"] = 0
            return self._lane_follow_target(current_wp), "settling"
        npc_state["target_lane_stable_ticks"] = 0
        target_wp, effective_direction = self._target_lane_waypoint(
            current_wp,
            target_lane_id,
            str(npc_state.get("lane_change_direction", "left")),
        )
        if effective_direction is not None:
            npc_state["lane_change_direction"] = effective_direction
            npc_state["effective_lane_change_direction"] = effective_direction
        if target_wp is None or target_wp.lane_id != target_lane_id:
            return self._lane_follow_target(current_wp), "lane_change_unavailable"
        if npc_state.get("target_lane_road_id") is None:
            npc_state["target_lane_road_id"] = getattr(target_wp, "road_id", None)
        return self._lane_follow_target(target_wp), "changing"

    def _saved_target_lane_waypoint(self, current_wp: Any, npc_state: Dict[str, Any]) -> Any:
        target_lane_id = npc_state.get("target_lane_id")
        target_road_id = npc_state.get("target_lane_road_id")
        candidates = [current_wp]
        visited = set()
        for _ in range(3):
            next_candidates = []
            for waypoint in candidates:
                key = (getattr(waypoint, "road_id", None), getattr(waypoint, "lane_id", None))
                if key in visited:
                    continue
                visited.add(key)
                road_matches = target_road_id is None or getattr(waypoint, "road_id", None) == target_road_id
                if road_matches and getattr(waypoint, "lane_id", None) == target_lane_id:
                    return waypoint
                for direction in ("left", "right"):
                    adjacent = self._adjacent_lane_waypoint(waypoint, direction)
                    if adjacent is not None:
                        next_candidates.append(adjacent)
            candidates = next_candidates
        return None

    def _path_steer(self, transform: Any, target_wp: Any, limit: float = 0.45) -> float:
        if target_wp is None:
            return 0.0
        target_loc = target_wp.transform.location
        desired_yaw = math.degrees(math.atan2(target_loc.y - transform.location.y, target_loc.x - transform.location.x))
        return max(-limit, min(limit, self._normalize_angle_deg(desired_yaw - transform.rotation.yaw) / 45.0))

    def _smooth_path_steer(self, transform: Any, target_wp: Any, npc_state: Dict[str, Any]) -> float:
        desired_steer = self._path_steer(transform, target_wp)
        previous_steer = float(npc_state.get("previous_steer", 0.0))
        steer = max(previous_steer - 0.08, min(previous_steer + 0.08, desired_steer))
        npc_state["previous_steer"] = steer
        return steer

    def _speed_control(self, current_speed_kmh: float, target_speed_kmh: float) -> Tuple[float, float]:
        speed_error = target_speed_kmh - current_speed_kmh
        if speed_error < -2.0:
            return 0.0, min(0.35, -speed_error / 20.0)
        if speed_error <= 1.0:
            return 0.18, 0.0
        return min(0.85, 0.25 + speed_error / 35.0), 0.0

    def _speeding_control(self, current_speed_kmh: float, target_speed_kmh: float) -> Tuple[float, float]:
        if target_speed_kmh <= 0.0:
            return 0.0, 0.0
        speed_ratio = current_speed_kmh / target_speed_kmh
        if speed_ratio < 0.95:
            return 1.0, 0.0
        if speed_ratio <= 1.05:
            return max(0.0, min(1.0, (1.05 - speed_ratio) / 0.10)), 0.0
        return 0.0, min(0.2, (speed_ratio - 1.05) * 2.0)

    def _adjacent_lane_waypoint(self, waypoint: Any, lane_change_direction: str) -> Any:
        try:
            if lane_change_direction == "left":
                candidate = waypoint.get_left_lane()
            else:
                candidate = waypoint.get_right_lane()
        except RuntimeError:
            return None
        if candidate is None:
            return None
        try:
            if candidate.lane_type != carla.LaneType.Driving:
                return None
        except RuntimeError:
            return None
        return candidate

    def _target_lane_waypoint(
        self,
        waypoint: Any,
        target_lane_id: Any,
        fallback_direction: str,
    ) -> Tuple[Any, Optional[str]]:
        if target_lane_id is None:
            direction = fallback_direction.lower()
            return self._adjacent_lane_waypoint(waypoint, direction), direction
        try:
            expected_lane_id = int(target_lane_id)
        except (TypeError, ValueError):
            return None, None
        for direction in ("left", "right"):
            candidate = self._adjacent_lane_waypoint(waypoint, direction)
            if candidate is not None and int(candidate.lane_id) == expected_lane_id:
                return candidate, direction
        return None, None

    def _plan_spawn_waypoints(self, world: Any, scenario: ScenarioConfiguration) -> Dict[str, Any]:
        actor_specs = [("ego", scenario.ego)] + [
            (str(npc.get("id", "npc")), npc) for npc in scenario.npcs
        ]
        original_specs = {
            name: copy.deepcopy(actor.get("spawn_waypoint", {}))
            for name, actor in actor_specs
        }
        candidate_sets: List[Tuple[str, Dict[str, Any], List[Any]]] = []
        for name, actor in actor_specs:
            waypoint_spec = actor.get("spawn_waypoint", {})
            candidates = self._spawn_waypoint_candidates(world.get_map(), waypoint_spec)
            if not candidates:
                raise RuntimeError(
                    f"waypoint not found: 找不到可行驶 waypoint: actor={name}, spec={waypoint_spec}"
                )
            candidate_sets.append((name, actor, candidates))

        constraints = self._spawn_plan_constraints(scenario.violation_type)
        if scenario.violation_type in {"违规变道", "违规超车"}:
            topology_exists = any(
                self._lane_change_pair_valid(ego_wp, npc_wp, scenario.npcs[0])
                for ego_wp in candidate_sets[0][2]
                for npc_wp in candidate_sets[1][2]
            ) if scenario.npcs else False
            if not topology_exists:
                raise RuntimeError(
                    "invalid lane topology: 变道 NPC 的目标相邻 Driving 车道不存在或与 ego 车道不匹配"
                )

        occupied = self._existing_vehicle_locations(world)
        attempts = [0]
        selected: List[Tuple[str, Dict[str, Any], Any]] = []
        solution: List[Tuple[str, Dict[str, Any], Any]] = []

        def search(index: int) -> bool:
            if index >= len(candidate_sets):
                attempts[0] += 1
                if self._spawn_combination_satisfies(scenario, selected):
                    solution.extend(selected)
                    return True
                return False
            name, actor, candidates = candidate_sets[index]
            for waypoint in candidates:
                location = waypoint.transform.location
                if any(self._location_distance(location, item) < 3.0 for item in occupied):
                    attempts[0] += 1
                    continue
                if any(self._waypoint_distance(waypoint, item[2]) < 5.0 for item in selected):
                    attempts[0] += 1
                    continue
                selected.append((name, actor, waypoint))
                if search(index + 1):
                    return True
                selected.pop()
            return False

        if not search(0):
            raise RuntimeError(
                f"spawn plan unsatisfied: 无法满足地图占用和场景关系约束; attempts={attempts[0]}"
            )

        final_specs: Dict[str, Dict[str, Any]] = {}
        for name, actor, waypoint in solution:
            current = actor["spawn_waypoint"]
            current["road_id"] = int(waypoint.road_id)
            current["lane_id"] = int(waypoint.lane_id)
            current["s"] = round(float(waypoint.s), 3)
            final_specs[name] = copy.deepcopy(current)
        return {
            "original_spec": original_specs,
            "final_spec": final_specs,
            "candidate_attempts": attempts[0],
            "constraints": constraints,
        }

    def _spawn_waypoint_candidates(self, road_map: Any, waypoint_spec: Dict[str, Any]) -> List[Any]:
        try:
            road_id = int(waypoint_spec["road_id"])
            lane_id = int(waypoint_spec["lane_id"])
            preferred_s = float(waypoint_spec["s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"waypoint not found: waypoint 参数无效: {waypoint_spec}") from exc
        s_values = [preferred_s]
        for offset in (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 35.0):
            s_values.extend((preferred_s - offset, preferred_s + offset))
        s_values.extend((6.0, 10.0, 14.0, 18.0, 24.0, 32.0, 40.0))
        unique_s = []
        seen = set()
        for s_value in s_values:
            normalized = round(max(0.0, float(s_value)), 3)
            if normalized not in seen:
                seen.add(normalized)
                unique_s.append(normalized)

        candidates = []
        candidate_keys = set()
        for s_value in unique_s:
            try:
                waypoint = road_map.get_waypoint_xodr(road_id, lane_id, s_value)
                if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
                    continue
            except RuntimeError:
                continue
            key = (int(waypoint.road_id), int(waypoint.lane_id), round(float(waypoint.s), 3))
            if key not in candidate_keys:
                candidate_keys.add(key)
                candidates.append(waypoint)
        return candidates

    def _spawn_combination_satisfies(
        self,
        scenario: ScenarioConfiguration,
        selected: List[Tuple[str, Dict[str, Any], Any]],
    ) -> bool:
        if len(selected) < 2:
            return True
        ego_wp = selected[0][2]
        npc_wp = selected[1][2]
        violation_type = scenario.violation_type
        if violation_type == "未保持安全距离":
            return self._front_waypoint_relation(ego_wp, npc_wp, 12.0, 35.0)
        if violation_type == "未注意前方路况":
            return self._front_waypoint_relation(ego_wp, npc_wp, 15.0, 35.0)
        if violation_type in {"超速", "超速行驶"}:
            return self._front_waypoint_relation(ego_wp, npc_wp, 15.0, 35.0)
        if violation_type == "违规变道":
            relative_longitudinal = self._relative_longitudinal_gap(ego_wp, npc_wp)
            return (
                self._lane_change_pair_valid(ego_wp, npc_wp, scenario.npcs[0])
                and 3.0 <= abs(relative_longitudinal) <= 12.0
            )
        if violation_type == "违规超车":
            relative_longitudinal = self._relative_longitudinal_gap(ego_wp, npc_wp)
            return (
                self._lane_change_pair_valid(ego_wp, npc_wp, scenario.npcs[0])
                and -15.0 <= relative_longitudinal <= -5.0
            )
        return True

    def _relative_longitudinal_gap(self, ego_wp: Any, other_wp: Any) -> float:
        ego_transform = ego_wp.transform
        other_location = other_wp.transform.location
        dx = other_location.x - ego_transform.location.x
        dy = other_location.y - ego_transform.location.y
        yaw = math.radians(ego_transform.rotation.yaw)
        return dx * math.cos(yaw) + dy * math.sin(yaw)

    def _front_waypoint_relation(self, ego_wp: Any, front_wp: Any, minimum: float, maximum: float) -> bool:
        if ego_wp.road_id != front_wp.road_id or ego_wp.lane_id != front_wp.lane_id:
            return False
        forward_gap = self._relative_longitudinal_gap(ego_wp, front_wp)
        distance = self._waypoint_distance(ego_wp, front_wp)
        return forward_gap > 0.0 and minimum <= distance <= maximum

    def _lane_change_pair_valid(self, ego_wp: Any, npc_wp: Any, npc: Dict[str, Any]) -> bool:
        if ego_wp.road_id != npc_wp.road_id or ego_wp.lane_id == npc_wp.lane_id:
            return False
        target_wp, effective_direction = self._target_lane_waypoint(
            npc_wp,
            npc.get("target_lane_id"),
            str(npc.get("target_lane_change", "left")),
        )
        if effective_direction is not None:
            npc["effective_lane_change_direction"] = effective_direction
        return (
            target_wp is not None
            and target_wp.road_id == npc_wp.road_id
            and target_wp.road_id == ego_wp.road_id
            and target_wp.lane_id == ego_wp.lane_id
        )

    def _spawn_plan_constraints(self, violation_type: str) -> List[str]:
        constraints = [
            "existing vehicles >= 3m",
            "planned actors >= 5m",
            "Driving waypoints only",
        ]
        descriptions = {
            "未保持安全距离": "front NPC on ego lane, forward gap 12-35m",
            "未注意前方路况": "front obstacle on ego lane, forward gap 15-35m",
            "超速": "speeding NPC ahead on ego lane, forward gap 15-35m",
            "超速行驶": "speeding NPC ahead on ego lane, forward gap 15-35m",
            "违规变道": "same road, distinct adjacent Driving lanes, longitudinal gap 3-12m",
            "违规超车": "same road, distinct adjacent Driving lanes, NPC behind ego 5-15m",
            "未按规定让行": "planned actors do not overlap",
            "逆行": "planned actors do not overlap",
        }
        if violation_type in descriptions:
            constraints.append(descriptions[violation_type])
        return constraints

    def _existing_vehicle_locations(self, world: Any) -> List[Any]:
        occupied = []
        try:
            for actor in world.get_actors().filter("vehicle.*"):
                transform = self._safe_get_transform(actor)
                if transform is not None:
                    occupied.append(transform.location)
        except RuntimeError:
            pass
        return occupied

    def _location_distance(self, first: Any, second: Any) -> float:
        return self._distance_2d(first.x, first.y, second.x, second.y)

    def _waypoint_distance(self, first: Any, second: Any) -> float:
        return self._location_distance(first.transform.location, second.transform.location)

    def _spawn_precheck(self, world: Any, named_waypoints: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        occupied = []
        try:
            for actor in world.get_actors().filter("vehicle.*"):
                transform = self._safe_get_transform(actor)
                if transform is not None:
                    occupied.append(transform.location)
        except RuntimeError:
            occupied = []
        for name, waypoint in named_waypoints:
            loc = waypoint.transform.location
            nearest_existing = min((self._distance_2d(loc.x, loc.y, item.x, item.y) for item in occupied), default=float("inf"))
            nearest_planned = min(
                (
                    self._distance_2d(loc.x, loc.y, other.transform.location.x, other.transform.location.y)
                    for other_name, other in named_waypoints
                    if other_name != name
                ),
                default=float("inf"),
            )
            ok = nearest_existing >= 3.0 and nearest_planned >= 5.0
            results.append({
                "id": name,
                "ok": ok,
                "x": round(loc.x, 3),
                "y": round(loc.y, 3),
                "nearest_existing_vehicle_m": round(nearest_existing, 3) if nearest_existing != float("inf") else None,
                "nearest_planned_spawn_m": round(nearest_planned, 3) if nearest_planned != float("inf") else None,
            })
        return results

    def _resolve_waypoint(self, world: Any, waypoint_spec: Dict[str, Any]) -> Any:
        try:
            waypoint = world.get_map().get_waypoint_xodr(
                int(waypoint_spec["road_id"]),
                int(waypoint_spec["lane_id"]),
                float(waypoint_spec["s"]),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"waypoint not found: waypoint 参数无效或地图解析失败: {waypoint_spec}") from exc
        if waypoint is None:
            raise RuntimeError(f"waypoint not found: 找不到 waypoint: {waypoint_spec}")
        try:
            if waypoint.lane_type != carla.LaneType.Driving:
                raise RuntimeError(f"waypoint not found: waypoint 不是 Driving 类型: {waypoint_spec}")
        except AttributeError as exc:
            raise RuntimeError(f"waypoint not found: waypoint 缺少车道类型: {waypoint_spec}") from exc
        return waypoint

    def _spawn_with_retry(self, world: Any, blueprint: Any, waypoint: Any, reverse_heading: bool = False) -> Any:
        base_transform = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
        if reverse_heading:
            base_transform.rotation.yaw += 180.0
        base_transform.location.z += 0.6
        actor = world.try_spawn_actor(blueprint, base_transform)
        if actor is not None:
            return actor
        for offset in [0.5, 1.5, 3.0, 5.0]:
            candidate = waypoint.next(offset)
            candidate_wp = candidate[0] if candidate else waypoint
            transform = carla.Transform(candidate_wp.transform.location, candidate_wp.transform.rotation)
            if reverse_heading:
                transform.rotation.yaw += 180.0
            transform.location.z += 0.6
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                return actor
        return None

    def _safe_update_spectator(self, spectator: Any, ego_actor: Any) -> None:
        try:
            if not self._is_actor_alive(ego_actor):
                return
            transform = ego_actor.get_transform()
            spectator.set_transform(
                carla.Transform(
                    carla.Location(x=transform.location.x, y=transform.location.y, z=55.0),
                    carla.Rotation(pitch=-90.0, yaw=transform.rotation.yaw, roll=0.0),
                )
            )
        except RuntimeError:
            pass

    def _sample_tick(
        self,
        road_map: Any,
        timestamp_s: float,
        ego_actor: Any,
        npc_actors: List[Any],
        conflict_point: Dict[str, Any],
        violation_type: str,
        stop_line_point: Dict[str, Any],
        violation_params: Dict[str, Any],
        red_activated: bool,
        latest_distance_to_stop_line: Optional[float],
        first_red_timestamp_s: Optional[float],
        lane_change_baselines: Dict[str, Any],
        ego_role: str = "ego",
    ) -> Dict[str, Any]:
        if not self._is_actor_alive(ego_actor):
            return None
        try:
            ego_transform = ego_actor.get_transform()
            ego_location = ego_transform.location
            ego_velocity = ego_actor.get_velocity()
        except RuntimeError:
            return None
        ego_signal_state = self._get_signal_state(ego_actor)
        ego_passed_stop_line = self._passed_stop_line(ego_location, stop_line_point, violation_params)
        npc_ticks = []
        for npc_cfg, npc_actor, _, npc_state in npc_actors:
            if not self._is_actor_alive(npc_actor):
                continue
            try:
                npc_transform = npc_actor.get_transform()
                npc_location = npc_transform.location
                npc_velocity = npc_actor.get_velocity()
                npc_waypoint = road_map.get_waypoint(npc_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            except RuntimeError:
                continue
            npc_speed_mps = self._vector_speed(npc_velocity)
            distance_to_ego = self._distance_2d(npc_location.x, npc_location.y, ego_location.x, ego_location.y)
            closing_speed = max(self._vector_speed(ego_velocity) - npc_speed_mps, 0.0)
            ttc_to_ego = distance_to_ego / max(closing_speed, 0.1)
            rss_safe_distance = self._rss_longitudinal_safe_distance(self._vector_speed(ego_velocity), npc_speed_mps, violation_params)
            baseline_transform = lane_change_baselines.get(npc_cfg.get("id"))
            lateral_offset = self._lateral_offset_m(baseline_transform, npc_transform)
            heading_gap_to_ego = abs(self._normalize_angle_deg(npc_transform.rotation.yaw - ego_transform.rotation.yaw))
            relative_x = npc_location.x - ego_location.x
            relative_y = npc_location.y - ego_location.y
            ego_yaw_rad = math.radians(ego_transform.rotation.yaw)
            relative_longitudinal = relative_x * math.cos(ego_yaw_rad) + relative_y * math.sin(ego_yaw_rad)
            lane_change_phase = npc_state.get("control_phase")
            source_lane_id = npc_state.get("source_lane_id")
            target_lane_id = npc_state.get("target_lane_id")
            current_lane_id = getattr(npc_waypoint, "lane_id", None)
            lane_change_completed = (
                target_lane_id is not None
                and current_lane_id == target_lane_id
                and lane_change_phase in {"completed", "settling"}
            )
            control_steer = float(npc_state.get("control_steer", 0.0))
            previous_yaw = npc_state.get("sample_previous_yaw_deg")
            previous_timestamp = npc_state.get("sample_previous_timestamp_s")
            lateral_accel_proxy = 0.0
            if previous_yaw is not None and previous_timestamp is not None:
                delta_time = timestamp_s - float(previous_timestamp)
                if delta_time > 0.0:
                    yaw_rate_rad_s = math.radians(
                        self._normalize_angle_deg(npc_transform.rotation.yaw - float(previous_yaw))
                    ) / delta_time
                    lateral_accel_proxy = npc_speed_mps ** 2 * yaw_rate_rad_s
            previous_sample_steer = npc_state.get("sample_previous_steer")
            steer_delta = 0.0 if previous_sample_steer is None else abs(control_steer - float(previous_sample_steer))
            max_steer_delta = max(float(npc_state.get("max_steer_delta", 0.0)), steer_delta)
            npc_state["sample_previous_yaw_deg"] = float(npc_transform.rotation.yaw)
            npc_state["sample_previous_timestamp_s"] = timestamp_s
            npc_state["sample_previous_steer"] = control_steer
            npc_state["max_steer_delta"] = max_steer_delta
            npc_ticks.append({
                "id": npc_cfg.get("id"),
                "role": npc_cfg.get("role"),
                "speed_mps": round(npc_speed_mps, 3),
                "distance_to_conflict_m": round(self._distance_2d(npc_location.x, npc_location.y, conflict_point["x"], conflict_point["y"]), 3),
                "distance_to_ego_m": round(distance_to_ego, 3),
                "ttc_to_ego_s": round(ttc_to_ego, 3),
                "rss_safe_distance_m": round(rss_safe_distance, 3),
                "rss_margin_m": round(distance_to_ego - rss_safe_distance, 3),
                "location": {"x": round(npc_location.x, 3), "y": round(npc_location.y, 3)},
                "yaw": round(npc_transform.rotation.yaw, 3),
                "heading_gap_to_ego_deg": round(heading_gap_to_ego, 3),
                "lateral_offset_m": round(lateral_offset, 3),
                "lane_change_active": "Change Lane" in str(npc_cfg.get("behavior", "")) or "Overtake" in str(npc_cfg.get("behavior", "")),
                "lane_change_direction": npc_state.get("effective_lane_change_direction", npc_cfg.get("target_lane_change")),
                "effective_lane_change_direction": npc_state.get("effective_lane_change_direction"),
                "road_id": getattr(npc_waypoint, "road_id", None),
                "lane_id": current_lane_id,
                "source_lane_id": source_lane_id,
                "target_lane_id": target_lane_id,
                "lane_change_phase": lane_change_phase,
                "lane_change_completed": lane_change_completed,
                "relative_longitudinal_m": round(relative_longitudinal, 3),
                "control_steer": round(control_steer, 4),
                "lateral_accel_proxy_mps2": round(lateral_accel_proxy, 4),
                "max_steer_delta": round(max_steer_delta, 4),
            })
        return {
            "timestamp_s": timestamp_s,
            "ego": {
                "role": ego_role,
                "speed_mps": round(self._vector_speed(ego_velocity), 3),
                "distance_to_conflict_m": round(self._distance_2d(ego_location.x, ego_location.y, conflict_point["x"], conflict_point["y"]), 3),
                "distance_to_stop_line_m": round(self._distance_2d(ego_location.x, ego_location.y, stop_line_point["x"], stop_line_point["y"]), 3),
                "passed_stop_line": ego_passed_stop_line,
                "signal_state": ego_signal_state,
                "location": {"x": round(ego_location.x, 3), "y": round(ego_location.y, 3)},
                "yaw": round(ego_transform.rotation.yaw, 3),
                "stop_line_point": stop_line_point,
                "red_activated": red_activated,
                "latest_distance_to_stop_line": None if latest_distance_to_stop_line is None else round(float(latest_distance_to_stop_line), 3),
                "first_red_timestamp_s": first_red_timestamp_s,
            },
            "npcs": npc_ticks,
            "conflict_point": conflict_point,
            "traffic_state": {
                "signal": ego_signal_state,
                "expected_violation": violation_type,
                "red_activated": red_activated,
                "first_red_timestamp_s": first_red_timestamp_s,
            },
        }

    def _wrong_way_risk_observed(self, sample: Dict[str, Any], params: Dict[str, Any]) -> bool:
        npcs = sample.get("npcs", [])
        if not npcs:
            return False
        danger_distance = float(params.get("danger_distance_m", 10.0))
        heading_opposition_deg = float(params.get("heading_opposition_deg", 120.0))
        npc = npcs[0]
        pair_distance = npc.get("distance_to_ego_m")
        heading_gap = npc.get("heading_gap_to_ego_deg")
        ego_speed = float(sample.get("ego", {}).get("speed_mps", 0.0))
        npc_speed = float(npc.get("speed_mps", 0.0))
        if pair_distance is None or heading_gap is None:
            return False
        return pair_distance <= danger_distance and heading_gap >= heading_opposition_deg and ego_speed > 1.0 and npc_speed > 1.0

    def _yield_risk_observed(self, sample: Dict[str, Any], params: Dict[str, Any]) -> bool:
        ego = sample.get("ego", {})
        ego_dist = ego.get("distance_to_conflict_m")
        ego_speed = float(ego.get("speed_mps", 0.0))
        if ego_dist is None or ego_speed <= 0.1:
            return False
        ego_ttc = float(ego_dist) / max(ego_speed, 0.1)
        low, high = params.get("time_gap_range_s", [0.5, 2.0])
        danger_distance = float(params.get("danger_distance_m", 8.0))
        for npc in sample.get("npcs", []):
            npc_dist = npc.get("distance_to_conflict_m")
            npc_speed = float(npc.get("speed_mps", 0.0))
            pair_distance = npc.get("distance_to_ego_m")
            if npc_dist is None or npc_speed <= 0.1 or pair_distance is None:
                continue
            npc_ttc = float(npc_dist) / max(npc_speed, 0.1)
            if low <= abs(ego_ttc - npc_ttc) <= high and float(pair_distance) <= danger_distance:
                return True
        return False

    def _front_risk_observed(self, sample: Dict[str, Any], params: Dict[str, Any]) -> bool:
        ego_speed = float(sample.get("ego", {}).get("speed_mps", 0.0))
        if ego_speed <= 1.0:
            return False
        danger_distance = float(params.get("danger_distance_m", params.get("min_gap_m", 15.0)))
        ttc_threshold = float(params.get("ttc_threshold_s", 3.0))
        for npc in sample.get("npcs", []):
            gap = npc.get("distance_to_ego_m")
            if gap is None:
                continue
            ttc = float(npc.get("ttc_to_ego_s", 999.0))
            rss_margin = float(npc.get("rss_margin_m", 999.0))
            if float(gap) <= danger_distance and (ttc <= ttc_threshold or rss_margin < 0.0):
                return True
        return False

    def _rss_longitudinal_safe_distance(self, ego_speed: float, front_speed: float, params: Dict[str, Any]) -> float:
        response_time = float(params.get("rss_response_time_s", 1.0))
        ego_accel_max = float(params.get("rss_ego_accel_max_mps2", 2.0))
        ego_brake_min = float(params.get("rss_ego_brake_min_mps2", 4.0))
        front_brake_max = float(params.get("rss_front_brake_max_mps2", 8.0))
        ego_response_speed = ego_speed + ego_accel_max * response_time
        ego_distance = ego_speed * response_time + 0.5 * ego_accel_max * response_time ** 2 + ego_response_speed ** 2 / (2.0 * max(ego_brake_min, 0.1))
        front_distance = front_speed ** 2 / (2.0 * max(front_brake_max, 0.1))
        return max(0.0, ego_distance - front_distance)

    def _get_signal_state(self, actor: Any) -> str:
        try:
            state = actor.get_traffic_light_state()
            if state is not None:
                return str(state).split(".")[-1].lower()
        except RuntimeError:
            pass
        try:
            tl = actor.get_traffic_light()
            if tl is not None:
                return str(tl.get_state()).split(".")[-1].lower()
        except RuntimeError:
            pass
        try:
            if actor.is_at_traffic_light():
                return "unknown"
        except RuntimeError:
            pass
        return "green"

    def _passed_stop_line(self, ego_location: Any, stop_line_point: Dict[str, Any], violation_params: Dict[str, Any]) -> bool:
        buffer_m = float(violation_params.get("stop_line_buffer_m", 2.0))
        dist = self._distance_2d(ego_location.x, ego_location.y, float(stop_line_point.get("x", 0.0)), float(stop_line_point.get("y", 0.0)))
        return dist <= buffer_m

    def _distance_actor_to_point(self, actor: Any, point: Dict[str, Any]) -> float:
        transform = self._safe_get_transform(actor)
        if transform is None:
            return float("inf")
        return self._distance_2d(transform.location.x, transform.location.y, float(point.get("x", 0.0)), float(point.get("y", 0.0)))

    def _safe_get_transform(self, actor: Any) -> Optional[Any]:
        if not self._is_actor_alive(actor):
            return None
        try:
            return actor.get_transform()
        except RuntimeError:
            return None

    def _is_actor_alive(self, actor: Any) -> bool:
        try:
            return actor is not None and actor.is_alive
        except RuntimeError:
            return False

    def _all_required_actors_alive(self, actors: List[Any]) -> bool:
        return all(self._is_actor_alive(actor) for actor in actors)

    def _vector_speed(self, velocity: Any) -> float:
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _distance_2d(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    def _normalize_angle_deg(self, angle: float) -> float:
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def _transform_to_dict(self, transform: Optional[Any]) -> Dict[str, Any]:
        if transform is None:
            return {}
        return {
            "x": round(transform.location.x, 3),
            "y": round(transform.location.y, 3),
            "z": round(transform.location.z, 3),
            "yaw": round(transform.rotation.yaw, 3),
            "pitch": round(transform.rotation.pitch, 3),
            "roll": round(transform.rotation.roll, 3),
        }

    def _copy_transform(self, transform: Optional[Any]) -> Optional[Any]:
        if transform is None:
            return None
        return carla.Transform(
            carla.Location(x=float(transform.location.x), y=float(transform.location.y), z=float(transform.location.z)),
            carla.Rotation(pitch=float(transform.rotation.pitch), yaw=float(transform.rotation.yaw), roll=float(transform.rotation.roll)),
        )

    def _lateral_offset_m(self, baseline: Optional[Any], current: Any) -> float:
        if baseline is None or current is None:
            return 0.0
        yaw = math.radians(baseline.rotation.yaw)
        dx = current.location.x - baseline.location.x
        dy = current.location.y - baseline.location.y
        lateral_x = -math.sin(yaw)
        lateral_y = math.cos(yaw)
        return dx * lateral_x + dy * lateral_y
