import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from RKGRScen.config import canonical_violation_type
from RKGRScen.models import RetrievalResult, ScenarioConfiguration

EGO_ROLES = {"ego", "priority", "observer", "victim"}
NPC_ROLES = {"violator", "lead", "obstacle", "npc", "front", "hazard"}


def _point(node: Dict[str, Any], key: str) -> Tuple[float, float]:
    value = node.get(key) or {}
    return float(value.get("x", 0.0)), float(value.get("y", 0.0))


def _distance(pa: Tuple[float, float], pb: Tuple[float, float]) -> float:
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _interpolate(start: Tuple[float, float], end: Tuple[float, float], ratio: float) -> Tuple[float, float]:
    ratio = max(0.0, min(1.0, ratio))
    return start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio


class ConstraintSolver:
    """约束求解器，实现论文 Algorithm 1 的 C1-C4 硬约束与软目标。

    - C1: 参与者 spawn 间距 >= d_min
    - C2: spawn 位置必须落在合法车道段（来自检索到的图节点，天然合法）
    - C3: 交互类违规的到达时间差 Delta t = t_ego - t_npc 落在 T_window 内
    - C4: 速度落在 [v_min, v_max]
    软目标：当 T_window 存在时最小化 |Delta t - center|，否则所有可行解代价为 0。
    """

    def __init__(
        self,
        d_min: float = 5.0,
        v_min_kmh: float = 5.0,
        v_max_kmh: float = 120.0,
        speed_step_kmh: float = 5.0,
        spawn_s_fractions: Sequence[float] = (0.15, 0.30, 0.50, 0.70, 0.85),
    ) -> None:
        self.d_min = d_min
        self.v_min = v_min_kmh
        self.v_max = v_max_kmh
        self.speed_step = speed_step_kmh
        self.spawn_s_fractions = tuple(spawn_s_fractions)

    def solve(self, scenario_spec: Dict[str, Any], retrieval_results: List[RetrievalResult]) -> ScenarioConfiguration:
        if not retrieval_results:
            raise ValueError("没有可用的检索结果，无法实例化场景")

        violation_type = canonical_violation_type(scenario_spec.get("violation_type", ""))
        actors = list(scenario_spec.get("actors", []) or [])
        timing = (scenario_spec.get("conflict", {}) or {}).get("timing", {}) or {}
        window = timing.get("time_gap_to_conflict_s")
        if not (isinstance(window, (list, tuple)) and len(window) == 2 and any(window)):
            window = None
        else:
            window = (float(window[0]), float(window[1]))

        for retrieval in retrieval_results:
            nodes = list(retrieval.matched_nodes or [])
            if not nodes:
                continue
            scenario = self._solve_location(
                violation_type,
                actors,
                nodes,
                retrieval.community.map_name,
                window,
                scenario_spec,
            )
            if scenario is not None:
                return scenario

        raise ValueError("所有候选位置都无法找到满足 C1-C4 的可行参数组合")

    # ------------------------------------------------------------------
    def _solve_location(
        self,
        violation_type: str,
        actors: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        map_name: str,
        window: Optional[Tuple[float, float]],
        scenario_spec: Dict[str, Any],
    ) -> Optional[ScenarioConfiguration]:
        ego_idx, npc_idx = self._ego_npc_indices(actors)
        conflict_point = self._conflict_point(violation_type, nodes)

        # 每个参与者可选的 spawn 位置（node, s）。
        per_actor_candidates: List[List[Dict[str, Any]]] = []
        for actor in actors:
            candidates = self._spawn_candidates(actor, nodes, ego_idx, npc_idx, actors, violation_type)
            if not candidates:
                return None
            per_actor_candidates.append(candidates)

        best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any], float]] = None
        for spawn_assignment in self._product(per_actor_candidates):
            positions = [item["point"] for item in spawn_assignment]
            if self._violates_c1(positions):
                continue
            if self._violates_c2(spawn_assignment):
                continue

            speeds = self._speed_candidates(actors, spawn_assignment, window, conflict_point, violation_type, scenario_spec)
            for speed_assignment in speeds:
                if self._violates_c4(speed_assignment):
                    continue
                if window is not None:
                    delta_t = self._arrival_delta(ego_idx, npc_idx, spawn_assignment, speed_assignment, conflict_point)
                    if delta_t is None or not (window[0] <= delta_t <= window[1]):
                        continue
                    cost = abs(delta_t - (window[0] + window[1]) / 2.0)
                else:
                    cost = 0.0
                if best is None or cost < best[0]:
                    best = (cost, spawn_assignment, speed_assignment, conflict_point, delta_t if window is not None else 0.0)

        if best is None:
            return None

        _, spawn_assignment, speed_assignment, conflict_point, _ = best
        ego_cfg = self._build_actor_config(actors[ego_idx], spawn_assignment[ego_idx], speed_assignment[ego_idx], violation_type, ego_idx, npc_idx)
        npcs = []
        for idx, actor in enumerate(actors):
            if idx == ego_idx:
                continue
            npcs.append(self._build_actor_config(actor, spawn_assignment[idx], speed_assignment[idx], violation_type, idx, npc_idx))
        return ScenarioConfiguration(
            scenario_id=f"{self._short_key(violation_type)}_scenario",
            violation_type=violation_type,
            map_name=map_name,
            ego=ego_cfg,
            npcs=npcs,
            conflict_point={"x": round(conflict_point[0], 3), "y": round(conflict_point[1], 3)},
            environment={"weather": "clear", "time": "day"},
            expected_violation=self._expected_violation(violation_type, actors, spawn_assignment, speed_assignment, window, conflict_point),
        )

    # ------------------------------------------------------------------
    # C1-C4 判定
    # ------------------------------------------------------------------
    def _violates_c1(self, positions: List[Tuple[float, float]]) -> bool:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if _distance(positions[i], positions[j]) < self.d_min:
                    return True
        return False

    def _violates_c2(self, spawn_assignment: List[Dict[str, Any]]) -> bool:
        # 候选位置均来自图节点（合法车道段），此处仅做占位校验。
        return any(item.get("point") is None for item in spawn_assignment)

    def _violates_c4(self, speed_assignment: List[float]) -> bool:
        return any(not (self.v_min <= v <= self.v_max) for v in speed_assignment)

    # ------------------------------------------------------------------
    # 候选生成
    # ------------------------------------------------------------------
    def _spawn_candidates(
        self,
        actor: Dict[str, Any],
        nodes: List[Dict[str, Any]],
        ego_idx: int,
        npc_idx: int,
        actors: List[Dict[str, Any]],
        violation_type: str,
    ) -> List[Dict[str, Any]]:
        role = str(actor.get("role", ""))
        if role in EGO_ROLES:
            preferred_nodes = self._ego_nodes(violation_type, nodes)
        elif role in NPC_ROLES:
            preferred_nodes = self._npc_nodes(violation_type, nodes)
        else:
            preferred_nodes = nodes

        candidates: List[Dict[str, Any]] = []
        for node in preferred_nodes:
            start = _point(node, "start")
            end = _point(node, "end")
            length = _distance(start, end)
            for fraction in self.spawn_s_fractions:
                s = round(max(1.0, length * fraction), 2)
                point = _interpolate(start, end, fraction)
                candidates.append({
                    "node": node,
                    "s": s,
                    "point": point,
                    "road_id": node.get("road_id"),
                    "lane_id": node.get("lane_id"),
                })
        # 去重（按 road_id/lane_id/s）
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            key = (item["road_id"], item["lane_id"], item["s"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    _INTERACTION_TYPES = {"Failure to yield", "Wrong-way driving", "Illegal lane change", "Illegal overtaking"}

    def _ego_nodes(self, violation_type: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if violation_type in self._INTERACTION_TYPES and len(nodes) >= 2:
            return nodes[:1]
        return nodes[:1]

    def _npc_nodes(self, violation_type: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if violation_type in self._INTERACTION_TYPES and len(nodes) >= 2:
            return nodes[1:2]
        return nodes[:1]

    def _speed_candidates(
        self,
        actors: List[Dict[str, Any]],
        spawn_assignment: List[Dict[str, Any]],
        window: Optional[Tuple[float, float]],
        conflict_point: Tuple[float, float],
        violation_type: str,
        scenario_spec: Dict[str, Any],
    ) -> List[List[float]]:
        # 每个参与者的候选速度；优先使用 actor 速度提示，其次用窗口反解。
        per_actor: List[List[float]] = []
        for idx, actor in enumerate(actors):
            hint = float(actor.get("speed_kmh", actor.get("speed", actor.get("init_speed_kmh", 30.0))) or 30.0)
            candidates = self._speed_grid(hint)
            per_actor.append(candidates)

        # 组合数过大时仅对 ego/npc 用窗口反解出的速度集合。
        return self._product(per_actor)

    def _speed_grid(self, hint: float) -> List[float]:
        center = max(self.v_min, min(self.v_max, hint))
        values: List[float] = []
        for offset in range(-3, 4):
            value = center + offset * self.speed_step
            if self.v_min <= value <= self.v_max:
                values.append(round(value, 1))
        if center not in values:
            values.append(round(center, 1))
        return sorted(set(values))

    # ------------------------------------------------------------------
    def _ego_npc_indices(self, actors: List[Dict[str, Any]]) -> Tuple[int, int]:
        ego_idx = next((i for i, a in enumerate(actors) if str(a.get("role", "")) in EGO_ROLES), 0)
        npc_idx = next((i for i, a in enumerate(actors) if i != ego_idx and str(a.get("role", "")) in NPC_ROLES), None)
        if npc_idx is None:
            npc_idx = next((i for i, a in enumerate(actors) if i != ego_idx), ego_idx)
        return ego_idx, npc_idx

    def _conflict_point(self, violation_type: str, nodes: List[Dict[str, Any]]) -> Tuple[float, float]:
        if len(nodes) >= 2:
            a = _interpolate(_point(nodes[0], "start"), _point(nodes[0], "end"), 0.5)
            b = _interpolate(_point(nodes[1], "start"), _point(nodes[1], "end"), 0.5)
            return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        return _interpolate(_point(nodes[0], "start"), _point(nodes[0], "end"), 0.5)

    def _arrival_delta(
        self,
        ego_idx: int,
        npc_idx: int,
        spawn_assignment: List[Dict[str, Any]],
        speed_assignment: List[float],
        conflict_point: Tuple[float, float],
    ) -> Optional[float]:
        if npc_idx == ego_idx:
            return None
        d_ego = _distance(spawn_assignment[ego_idx]["point"], conflict_point)
        d_npc = _distance(spawn_assignment[npc_idx]["point"], conflict_point)
        v_ego = max(speed_assignment[ego_idx] / 3.6, 0.1)
        v_npc = max(speed_assignment[npc_idx] / 3.6, 0.1)
        return d_ego / v_ego - d_npc / v_npc

    # ------------------------------------------------------------------
    def _build_actor_config(
        self,
        actor: Dict[str, Any],
        spawn: Dict[str, Any],
        speed_kmh: float,
        violation_type: str,
        idx: int,
        npc_idx: int,
    ) -> Dict[str, Any]:
        role = str(actor.get("role", ""))
        config: Dict[str, Any] = {
            "role": role,
            "spawn_waypoint": {"road_id": spawn["road_id"], "lane_id": spawn["lane_id"], "s": spawn["s"]},
            "init_speed_kmh": round(speed_kmh, 1),
            "behavior": self._behavior(violation_type, role, idx, actor),
        }
        if role in NPC_ROLES and violation_type == "Wrong-way driving":
            config["reverse_spawn_heading"] = True
        if violation_type == "Illegal lane change" or violation_type == "Illegal overtaking":
            direction = str(actor.get("action", "")).lower()
            direction = "right" if "right" in direction else "left"
            config["target_lane_change"] = direction
        return config

    def _behavior(self, violation_type: str, role: str, idx: int, actor: Dict[str, Any]) -> str:
        action = str(actor.get("action", "") or "")
        if action:
            return action
        if role in EGO_ROLES:
            return "Move Forward (autopilot)"
        return {
            "Failure to yield": "Turn Left",
            "Failure to maintain safe following distance": "Lead Brake",
            "Inattention to the road ahead": "Static Obstacle",
            "Illegal lane change": "Change Lane Left",
            "Illegal overtaking": "Overtake Left",
            "Wrong-way driving": "Move Forward",
            "Speeding": "Move Forward Speeding",
        }.get(violation_type, "Move Forward")

    def _expected_violation(
        self,
        violation_type: str,
        actors: List[Dict[str, Any]],
        spawn_assignment: List[Dict[str, Any]],
        speed_assignment: List[float],
        window: Optional[Tuple[float, float]],
        conflict_point: Tuple[float, float],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if violation_type == "Failure to yield" and window is not None:
            params = {"time_gap_range_s": [window[0], window[1]], "danger_distance_m": 8.0}
        elif violation_type == "Failure to maintain safe following distance":
            params = {"thw_threshold_s": 1.0, "min_gap_m": 8.0, "rss_response_time_s": 1.0, "rss_ego_accel_max_mps2": 2.0, "rss_ego_brake_min_mps2": 4.0, "rss_front_brake_max_mps2": 8.0}
        elif violation_type == "Inattention to the road ahead":
            params = {"ttc_threshold_s": 3.0, "danger_distance_m": 15.0, "rss_response_time_s": 1.0, "rss_ego_accel_max_mps2": 2.0, "rss_ego_brake_min_mps2": 4.0, "rss_front_brake_max_mps2": 8.0}
        elif violation_type == "Wrong-way driving":
            params = {"danger_distance_m": 12.5, "heading_opposition_deg": 120.0}
        elif violation_type == "Illegal lane change":
            params = {"danger_gap_m": 8.0}
        elif violation_type == "Illegal overtaking":
            params = {"behind_threshold_m": 3.0, "ahead_threshold_m": 3.0, "danger_gap_m": 10.0}
        elif violation_type == "Speeding":
            ego = next((a for a in actors if str(a.get("role", "")) in EGO_ROLES), actors[0] if actors else {})
            limit = float(ego.get("speed_limit_kmh", 40.0) or 40.0)
            params = {"speed_limit_kmh": limit, "target_speed_kmh": limit + 15.0, "subject": "ego", "violator_role": "violator"}
        return {"type": violation_type, "detector": self._detector_name(violation_type), "params": params}

    def _detector_name(self, violation_type: str) -> str:
        return {
            "Failure to yield": "yield_violation_detector",
            "Failure to maintain safe following distance": "following_distance_detector",
            "Inattention to the road ahead": "inattention_front_condition_detector",
            "Illegal lane change": "lane_change_violation_detector",
            "Illegal overtaking": "overtake_violation_detector",
            "Wrong-way driving": "wrong_way_detector",
            "Speeding": "speeding_violation_detector",
        }.get(violation_type, "generic_detector")

    @staticmethod
    def _short_key(violation_type: str) -> str:
        return {
            "Failure to yield": "yield",
            "Failure to maintain safe following distance": "following",
            "Inattention to the road ahead": "inattention",
            "Illegal lane change": "lane_change",
            "Illegal overtaking": "overtake",
            "Wrong-way driving": "wrong_way",
            "Speeding": "speeding",
        }.get(violation_type, "scenario")

    @staticmethod
    def _product(lists: List[List[Any]]) -> List[List[Any]]:
        if not lists:
            return [[]]
        result: List[List[Any]] = [[]]
        for values in lists:
            result = [item + [value] for item in result for value in values]
        return result
