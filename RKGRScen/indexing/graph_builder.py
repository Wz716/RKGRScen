import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import networkx as nx

try:
    import carla
except ImportError:
    carla = None

class RoadGraphBuilder:

    def __init__(self, waypoint_step: float = 5.0) -> None:
        self.waypoint_step = waypoint_step

    def build_from_records(self, map_name: str, lane_records: Iterable[Dict[str, Any]]) -> nx.DiGraph:
        graph = nx.DiGraph(map_name=map_name, waypoint_step=self.waypoint_step)
        for record in lane_records:
            node_id = record["node_id"]
            graph.add_node(
                node_id,
                road_id=record.get("road_id"),
                lane_id=record.get("lane_id"),
                section_id=record.get("section_id", 0),
                road_type=record.get("road_type", "straight"),
                lane_count=record.get("lane_count", 1),
                curvature=self._estimate_curvature(record),
                speed_limit=record.get("speed_limit", 40),
                is_junction=record.get("is_junction", False),
                lane_change=record.get("lane_change", "none"),
                has_traffic_light=record.get("has_traffic_light", False),
                has_shoulder=record.get("has_shoulder", False),
                start=record.get("start", {"x": 0.0, "y": 0.0}),
                end=record.get("end", {"x": 0.0, "y": 0.0}),
                heading=record.get("heading", 0.0),
                tags=record.get("tags", []),
            )

        for record in lane_records:
            source = record["node_id"]
            for edge in record.get("next", []):
                graph.add_edge(source, edge["node_id"], connection_type=edge.get("type", "forward"), weight=edge.get("weight", 1.0))
        return graph

    def build_from_carla_map(
        self,
        carla_map: "carla.Map",
        map_name: Optional[str] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> nx.DiGraph:
        if carla is None:
            raise RuntimeError("未安装 carla Python API，无法从实时地图构建图谱")

        graph = nx.DiGraph(map_name=map_name or carla_map.name, waypoint_step=self.waypoint_step)
        lane_groups: Dict[str, List[Any]] = {}

        start = time.time()
        waypoints = carla_map.generate_waypoints(self.waypoint_step)
        self._log(logger, f"build_from_carla_map: 生成 waypoint {len(waypoints)} 个, 耗时 {time.time() - start:.2f}s")

        start = time.time()
        for index, waypoint in enumerate(waypoints, start=1):
            key = self._lane_segment_key(waypoint)
            lane_groups.setdefault(key, []).append(waypoint)
            if logger and index % 200 == 0:
                self._log(logger, f"build_from_carla_map: 已完成 lane grouping {index}/{len(waypoints)}")
        self._log(logger, f"build_from_carla_map: lane grouping 完成, 分组数 {len(lane_groups)}, 耗时 {time.time() - start:.2f}s")

        start = time.time()
        group_items = list(lane_groups.items())
        for index, (node_id, group_waypoints) in enumerate(group_items, start=1):
            ordered = sorted(group_waypoints, key=lambda item: item.s)
            graph.add_node(node_id, **self._carla_lane_attributes(ordered))
            if logger and index % 50 == 0:
                self._log(logger, f"build_from_carla_map: 已添加 node {index}/{len(group_items)}")
        self._log(logger, f"build_from_carla_map: node 构建完成, 耗时 {time.time() - start:.2f}s")

        start = time.time()
        for index, (node_id, group_waypoints) in enumerate(group_items, start=1):
            ordered = sorted(group_waypoints, key=lambda item: item.s)
            target_pairs = set()
            for waypoint in ordered[-2:]:
                next_waypoints = waypoint.next(self.waypoint_step)
                for next_waypoint in next_waypoints[:4]:
                    next_id = self._lane_segment_key(next_waypoint)
                    if next_id == node_id or next_id not in lane_groups:
                        continue
                    edge_type = "junction" if waypoint.is_junction or next_waypoint.is_junction else "forward"
                    target_pairs.add((next_id, edge_type))
            for target_id, edge_type in target_pairs:
                graph.add_edge(node_id, target_id, connection_type=edge_type, weight=1.0)
            if logger and index % 50 == 0:
                self._log(logger, f"build_from_carla_map: 已添加 edge source {index}/{len(group_items)}")
        self._log(logger, f"build_from_carla_map: edge 构建完成, 耗时 {time.time() - start:.2f}s")
        return graph

    def save_graph(self, graph: nx.DiGraph, output_path: Path) -> None:
        payload = nx.node_link_data(graph)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_graph(self, path: Path) -> nx.DiGraph:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return nx.node_link_graph(payload)

    def summarize_graph(self, graph: nx.DiGraph) -> Dict[str, Any]:
        junction_nodes = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("is_junction"))
        traffic_light_nodes = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("has_traffic_light"))
        shoulder_nodes = sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("has_shoulder"))
        lane_counts = [int(attrs.get("lane_count", 1)) for _, attrs in graph.nodes(data=True)]
        return {
            "map_name": graph.graph.get("map_name"),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "junction_node_count": junction_nodes,
            "traffic_light_node_count": traffic_light_nodes,
            "shoulder_node_count": shoulder_nodes,
            "max_lane_count": max(lane_counts) if lane_counts else 0,
        }

    def _lane_segment_key(self, waypoint: "carla.Waypoint") -> str:
        return f"road_{waypoint.road_id}_section_{waypoint.section_id}_lane_{waypoint.lane_id}"

    def _carla_lane_attributes(self, waypoints: List["carla.Waypoint"]) -> Dict[str, Any]:
        first = waypoints[0]
        last = waypoints[-1]
        lane_count = self._lane_count(first)
        speed_limit = 40
        lane_change = str(first.lane_change)
        has_shoulder = self._has_shoulder(first)
        has_traffic_light = bool(first.is_junction)
        return {
            "road_id": first.road_id,
            "lane_id": first.lane_id,
            "section_id": first.section_id,
            "road_type": "junction" if first.is_junction else self._road_type(waypoints),
            "lane_count": lane_count,
            "curvature": self._curvature_from_waypoints(waypoints),
            "speed_limit": speed_limit,
            "is_junction": bool(first.is_junction),
            "lane_change": lane_change,
            "has_traffic_light": has_traffic_light,
            "has_shoulder": has_shoulder,
            "start": {"x": round(first.transform.location.x, 3), "y": round(first.transform.location.y, 3)},
            "end": {"x": round(last.transform.location.x, 3), "y": round(last.transform.location.y, 3)},
            "heading": round(first.transform.rotation.yaw, 3),
            "tags": [str(first.lane_type)],
        }

    def _lane_count(self, waypoint: "carla.Waypoint") -> int:
        count = 1
        count += self._count_directional_lanes(waypoint, direction="left")
        count += self._count_directional_lanes(waypoint, direction="right")
        return count

    def _count_directional_lanes(self, waypoint: "carla.Waypoint", direction: str) -> int:
        count = 0
        visited = set()
        getter_name = "get_left_lane" if direction == "left" else "get_right_lane"
        getter = getattr(waypoint, getter_name)
        probe = getter()
        while probe is not None:
            probe_key = (probe.road_id, probe.section_id, probe.lane_id)
            if probe_key in visited:
                break
            visited.add(probe_key)
            if probe.road_id != waypoint.road_id or probe.section_id != waypoint.section_id or probe.lane_type != waypoint.lane_type:
                break
            count += 1
            if count >= 8:
                break
            probe = getattr(probe, getter_name)()
        return count

    def _has_shoulder(self, waypoint: "carla.Waypoint") -> bool:
        for neighbor in (waypoint.get_left_lane(), waypoint.get_right_lane()):
            if neighbor and str(neighbor.lane_type).endswith("Shoulder"):
                return True
        return False

    def _road_type(self, waypoints: List["carla.Waypoint"]) -> str:
        curvature = self._curvature_from_waypoints(waypoints)
        return "curve" if curvature > 0.05 else "straight"

    def _curvature_from_waypoints(self, waypoints: List["carla.Waypoint"]) -> float:
        if len(waypoints) < 2:
            return 0.0
        start = waypoints[0].transform.location
        end = waypoints[-1].transform.location
        heading = abs(waypoints[-1].transform.rotation.yaw - waypoints[0].transform.rotation.yaw)
        distance = math.hypot(end.x - start.x, end.y - start.y)
        if distance == 0:
            return 0.0
        return round(heading / max(distance, 1.0), 4)

    def _estimate_curvature(self, record: Dict[str, Any]) -> float:
        start = record.get("start", {"x": 0.0, "y": 0.0})
        end = record.get("end", {"x": 0.0, "y": 0.0})
        heading = abs(record.get("heading", 0.0))
        distance = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
        if distance == 0:
            return 0.0
        return round(heading / max(distance, 1.0), 4)

    def _log(self, logger: Optional[Callable[[str], None]], message: str) -> None:
        if logger is not None:
            logger(message)
