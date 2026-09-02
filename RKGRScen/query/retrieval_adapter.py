import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    carla = None

from RKGRScen.models import CommunityRecord, RetrievalResult

class RetrievalScenarioAdapter:
    def __init__(self, base_dir: Path, carla_host: str = "localhost", carla_port: int = 2000) -> None:
        self.base_dir = Path(base_dir)
        self.carla_host = carla_host
        self.carla_port = carla_port
        self._retrieval_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._valid_s_cache: Dict[Tuple[str, int, int], List[float]] = {}

    def load_retrieval_lookup(self) -> Dict[str, Dict[str, Any]]:
        if self._retrieval_cache is not None:
            return self._retrieval_cache
        path = self.base_dir / "RKGRScen" / "data" / "retrieval" / "p0_graphrag_retrieval" / "retrieval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._retrieval_cache = {row["source_path"]: row for row in payload.get("results", [])}
        return self._retrieval_cache

    def row_for_source(self, source_path: Path) -> Dict[str, Any]:
        row = self.load_retrieval_lookup().get(str(source_path))
        if not row:
            raise RuntimeError(f"找不到检索结果: {source_path}")
        if not row.get("retrieval", {}).get("local_top_k"):
            raise RuntimeError(f"检索结果没有 local_top_k: {source_path}")
        return row

    def top_local(self, row: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        local = row.get("retrieval", {}).get("local_top_k", [])
        if len(local) <= index:
            raise RuntimeError(f"local_top_k 不足 index={index}: {row.get('source_path')}")
        return local[index]

    def executable_top_local(self, row: Dict[str, Any], min_gap_m: float = 20.0) -> Dict[str, Any]:
        last_error = None
        for top in row.get("retrieval", {}).get("local_top_k", []):
            try:
                values = self.valid_s_values(str(top["map_name"]), int(top["road_id"]), int(top["lane_id"]), min_count=2)
                if values[-1] - values[0] >= min_gap_m:
                    return top
                last_error = RuntimeError(f"可用 s 范围不足: {values[0]}-{values[-1]}")
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(f"没有可执行 local candidate: {row.get('source_path')} error={last_error}")

    def build_retrieval_result(self, row: Dict[str, Any], index: int = 0) -> RetrievalResult:
        top = self.executable_top_local(row) if index == 0 else self.top_local(row, index)
        community = CommunityRecord(
            community_id=top["community_id"],
            map_name=top["map_name"],
            node_ids=[top["node_id"]],
            structure={"source": "retrieval_driven"},
            summary=f"retrieval driven local match for {row.get('source_violation_type')}",
            applicable_violations=[row.get("source_violation_type", "")],
            score=float(top.get("community_score", top.get("score", 1.0))),
        )
        matched_node = {
            "node_id": top["node_id"],
            "road_id": top["road_id"],
            "lane_id": top["lane_id"],
            "section_id": top.get("section_id", 0),
            "road_type": top.get("road_type"),
            "lane_count": top.get("lane_count"),
            "curvature": top.get("curvature"),
            "speed_limit": top.get("speed_limit"),
            "is_junction": top.get("road_type") == "Intersection",
            "lane_change": top.get("lane_change"),
            "has_traffic_light": top.get("has_traffic_light"),
            "has_shoulder": top.get("has_shoulder"),
            "start": top.get("start", {}),
            "end": top.get("end", {}),
            "heading": top.get("heading", 0.0),
        }
        return RetrievalResult(community=community, matched_nodes=[matched_node], score=float(top.get("score", 1.0)))

    def valid_s_values(self, map_name: str, road_id: int, lane_id: int, min_count: int = 2) -> List[float]:
        key = (map_name, int(road_id), int(lane_id))
        if key in self._valid_s_cache:
            return self._valid_s_cache[key]
        if carla is None:
            values = [5.0, 25.0, 45.0]
        else:
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(20.0)
            world = client.load_world(map_name.split("/")[-1])
            road_map = world.get_map()
            values = []
            for index in range(1, 260):
                s = float(index * 2)
                waypoint = road_map.get_waypoint_xodr(int(road_id), int(lane_id), s)
                if waypoint is not None:
                    values.append(round(s, 2))
        if len(values) < min_count:
            raise RuntimeError(f"检索 road/lane 无足够可用 s: map={map_name} road={road_id} lane={lane_id} valid={values[:5]}")
        self._valid_s_cache[key] = values
        return values

    def choose_s_pair(self, values: List[float], gap_m: float) -> Tuple[float, float]:
        start_index = max(0, min(len(values) - 2, len(values) // 4))
        ego_s = values[start_index]
        front_s = None
        for value in values[start_index + 1:]:
            if value - ego_s >= gap_m:
                front_s = value
                break
        if front_s is None:
            ego_s = values[0]
            front_s = values[-1]
        return ego_s, front_s

    def same_lane_parameter_hint(self, row: Dict[str, Any], gap_m: float = 28.0) -> Dict[str, Any]:
        top = self.executable_top_local(row, min_gap_m=max(8.0, gap_m * 0.75))
        map_name = str(top["map_name"])
        road_id = int(top["road_id"])
        lane_id = int(top["lane_id"])
        values = self.valid_s_values(map_name, road_id, lane_id)
        ego_s, front_s = self.choose_s_pair(values, gap_m)
        return {
            "map_name": map_name,
            "road_id": road_id,
            "lane_id": lane_id,
            "ego_s": ego_s,
            "front_s": front_s,
            "lead_s": front_s,
            "obstacle_s": front_s,
            "priority_road_id": road_id,
            "priority_lane_id": lane_id,
            "violator_road_id": road_id,
            "violator_lane_id": lane_id,
            "priority_s": ego_s,
            "violator_s": front_s,
            "conflict_point": {"x": top.get("start", {}).get("x", 0.0), "y": top.get("start", {}).get("y", 0.0)},
        }
