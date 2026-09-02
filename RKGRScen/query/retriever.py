import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from RKGRScen.models import CommunityRecord, RetrievalResult

class GraphRetriever:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        self.maps_dir = self.base_dir / "data" / "maps"
        self.community_dir = self.base_dir / "data" / "community_index"
        self._graph_cache: Dict[str, Dict[str, Any]] = {}
        self._node_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def retrieve(self, scenario_spec: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        communities = self.load_communities()
        global_candidates = self.global_search(scenario_spec, communities, top_k=top_k)
        local_results = self.local_search_json(scenario_spec, global_candidates, top_k=top_k)
        return {
            "scenario_spec": scenario_spec,
            "global_top_k": [self._community_export(item) for item in global_candidates],
            "local_top_k": local_results,
        }

    def retrieve_without_community(self, scenario_spec: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
        requirement = scenario_spec.get("road_requirement", {})
        results: List[Dict[str, Any]] = []
        for town_id in range(1, 6):
            town = f"Town{town_id:02d}"
            nodes_by_id = self._graph_nodes_by_id(town)
            lane_index = self._lane_index_by_road(town)
            community = CommunityRecord(
                community_id=f"{town}_full_graph",
                map_name=town,
                node_ids=list(nodes_by_id),
                structure={"source": "full_graph"},
                summary=f"All graph nodes in {town}",
                applicable_violations=[],
                score=0.0,
            )
            if requirement.get("same_direction_multi_lane"):
                results.extend(self._full_graph_lane_pair_results(community, nodes_by_id, requirement, lane_index))
                continue
            for node_id, attrs in nodes_by_id.items():
                matched, detail, node_score = self._score_node(attrs, requirement, lane_index)
                if matched:
                    results.append(self._export_node_result(community, node_id, attrs, detail, node_score))
        results.sort(key=lambda item: item["score"], reverse=True)
        if results:
            top_town = self._town_name(results[0].get("map_name", ""))
            results = [item for item in results if self._town_name(item.get("map_name", "")) == top_town]
        return {
            "scenario_spec": scenario_spec,
            "global_top_k": [],
            "local_top_k": results[:top_k],
            "mode": "full_graph",
        }

    def _full_graph_lane_pair_results(
        self,
        community: CommunityRecord,
        nodes_by_id: Dict[str, Dict[str, Any]],
        requirement: Dict[str, Any],
        lane_index: Dict[int, List[int]],
    ) -> List[Dict[str, Any]]:
        direction = str(requirement.get("lane_change_direction", "left"))
        nodes_by_road: Dict[int, List[Tuple[Dict[str, Any], Dict[str, List[str]], float]]] = {}
        scored: List[Tuple[Dict[str, Any], Dict[str, List[str]], float]] = []
        for attrs in nodes_by_id.values():
            matched, detail, node_score = self._score_node(attrs, requirement, lane_index)
            if matched:
                item = (attrs, detail, node_score)
                nodes_by_road.setdefault(int(attrs.get("road_id", 0)), []).append(item)
                scored.append(item)
        results: List[Dict[str, Any]] = []
        for source, source_detail, source_score in scored:
            for target, _, target_score in nodes_by_road.get(int(source.get("road_id", 0)), []):
                if not self._same_direction_adjacent(source, target, direction):
                    continue
                pair_score = source_score + target_score + 3.0
                results.append({
                    "match_type": "lane_pair",
                    "community_id": community.community_id,
                    "map_name": community.map_name,
                    "node_id": source.get("id"),
                    "road_id": source.get("road_id"),
                    "lane_id": source.get("lane_id"),
                    "section_id": source.get("section_id"),
                    "road_type": self._road_type(source),
                    "lane_count": source.get("lane_count"),
                    "curvature": source.get("curvature"),
                    "speed_limit": source.get("speed_limit"),
                    "has_traffic_light": source.get("has_traffic_light"),
                    "has_shoulder": source.get("has_shoulder"),
                    "lane_change": source.get("lane_change"),
                    "start": source.get("start"),
                    "end": source.get("end"),
                    "heading": source.get("heading"),
                    "source_node": {"node_id": source.get("id"), **source},
                    "target_node": {"node_id": target.get("id"), **target},
                    "lane_pair": {
                        "source_node_id": source.get("id"),
                        "source_road_id": source.get("road_id"),
                        "source_lane_id": source.get("lane_id"),
                        "target_node_id": target.get("id"),
                        "target_road_id": target.get("road_id"),
                        "target_lane_id": target.get("lane_id"),
                        "direction": direction,
                    },
                    "community_score": 0.0,
                    "node_match_score": round(pair_score, 4),
                    "score": round(pair_score, 4),
                    "matched_constraints": source_detail["matched"] + ["same_direction_adjacent_lane_pair"],
                    "failed_constraints": [],
                    "limited_index": False,
                })
        return results

    def load_communities(self) -> List[CommunityRecord]:
        records: List[CommunityRecord] = []
        for town_id in range(1, 6):
            town = f"Town{town_id:02d}"
            full = self.community_dir / f"{town}_community_tagged_index_full.json"
            tagged = self.community_dir / f"{town}_community_tagged_index_llm.json"
            fallback_tagged = self.community_dir / f"{town}_community_tagged_index.json"
            summary = self.community_dir / f"{town}_community_detection_summary.json"
            if full.exists():
                records.extend(self._load_community_file(full, limited_index=False))
            elif tagged.exists():
                records.extend(self._load_community_file(tagged, limited_index=False))
            elif fallback_tagged.exists():
                records.extend(self._load_community_file(fallback_tagged, limited_index=False))
            elif summary.exists():
                records.extend(self._load_community_file(summary, limited_index=True))
        return records

    def global_search(self, scenario_spec: Dict[str, Any], communities: Iterable[CommunityRecord], top_k: int = 3) -> List[CommunityRecord]:
        violation_type = self._canonical_violation(scenario_spec.get("violation_type", ""))
        candidates: List[CommunityRecord] = []
        for community in communities:
            applicable = [self._canonical_violation(item) for item in community.applicable_violations]
            hard_filter_pass = not applicable or violation_type in applicable or self._structural_fallback_match(scenario_spec.get("road_requirement", {}), community.structure)
            if not hard_filter_pass:
                continue
            score = self._community_score(scenario_spec, community)
            if score <= 0:
                continue
            community.score = score
            candidates.append(community)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def local_search(self, graph: nx.DiGraph, scenario_spec: Dict[str, Any], candidates: Iterable[CommunityRecord]) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for community in candidates:
            matched_nodes: List[Dict[str, Any]] = []
            for node_id in community.node_ids:
                attrs = graph.nodes[node_id]
                if self._match_requirement(attrs, scenario_spec["road_requirement"]):
                    matched_nodes.append({"node_id": node_id, **attrs})
            if matched_nodes:
                results.append(RetrievalResult(community=community, matched_nodes=matched_nodes, score=community.score))
        results.sort(key=lambda item: (item.score, len(item.matched_nodes)), reverse=True)
        return results

    def local_search_json(self, scenario_spec: Dict[str, Any], candidates: Iterable[CommunityRecord], top_k: int = 3) -> List[Dict[str, Any]]:
        requirement = scenario_spec.get("road_requirement", {})
        results: List[Dict[str, Any]] = []
        for community in candidates:
            map_name = self._town_name(community.map_name)
            nodes = self._graph_nodes_by_id(map_name)
            same_road_lane_index = self._lane_index_by_road(map_name)
            if requirement.get("same_direction_multi_lane"):
                results.extend(self._local_lane_pair_results(community, nodes, requirement, same_road_lane_index))
                continue
            for node_id in community.node_ids:
                attrs = nodes.get(node_id)
                if not attrs:
                    continue
                matched, detail, node_score = self._score_node(attrs, requirement, same_road_lane_index)
                if not matched:
                    continue
                results.append(self._export_node_result(community, node_id, attrs, detail, node_score))
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def _export_node_result(self, community: CommunityRecord, node_id: str, attrs: Dict[str, Any], detail: Dict[str, List[str]], node_score: float) -> Dict[str, Any]:
        return {
            "match_type": "single_node",
            "community_id": community.community_id,
            "map_name": community.map_name,
            "node_id": node_id,
            "road_id": attrs.get("road_id"),
            "lane_id": attrs.get("lane_id"),
            "section_id": attrs.get("section_id"),
            "road_type": self._road_type(attrs),
            "lane_count": attrs.get("lane_count"),
            "curvature": attrs.get("curvature"),
            "speed_limit": attrs.get("speed_limit"),
            "has_traffic_light": attrs.get("has_traffic_light"),
            "has_shoulder": attrs.get("has_shoulder"),
            "lane_change": attrs.get("lane_change"),
            "start": attrs.get("start"),
            "end": attrs.get("end"),
            "heading": attrs.get("heading"),
            "community_score": round(community.score, 4),
            "node_match_score": round(node_score, 4),
            "score": round(community.score + node_score, 4),
            "matched_constraints": detail["matched"],
            "failed_constraints": detail["failed"],
            "limited_index": bool(getattr(community, "limited_index", False)),
        }

    def _local_lane_pair_results(self, community: CommunityRecord, nodes_by_id: Dict[str, Dict[str, Any]], requirement: Dict[str, Any], lane_index: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        direction = str(requirement.get("lane_change_direction", "left"))
        community_nodes = [nodes_by_id[node_id] for node_id in community.node_ids if node_id in nodes_by_id]
        scored_nodes = []
        for attrs in community_nodes:
            matched, detail, node_score = self._score_node(attrs, requirement, lane_index)
            if matched:
                scored_nodes.append((attrs, detail, node_score))
        results: List[Dict[str, Any]] = []
        for source, source_detail, source_score in scored_nodes:
            for target, target_detail, target_score in scored_nodes:
                if source is target:
                    continue
                if not self._same_direction_adjacent(source, target, direction):
                    continue
                pair_score = source_score + target_score + 3.0
                results.append({
                    "match_type": "lane_pair",
                    "community_id": community.community_id,
                    "map_name": community.map_name,
                    "node_id": source.get("id"),
                    "road_id": source.get("road_id"),
                    "lane_id": source.get("lane_id"),
                    "section_id": source.get("section_id"),
                    "road_type": self._road_type(source),
                    "lane_count": source.get("lane_count"),
                    "curvature": source.get("curvature"),
                    "speed_limit": source.get("speed_limit"),
                    "has_traffic_light": source.get("has_traffic_light"),
                    "has_shoulder": source.get("has_shoulder"),
                    "lane_change": source.get("lane_change"),
                    "start": source.get("start"),
                    "end": source.get("end"),
                    "heading": source.get("heading"),
                    "source_node": {"node_id": source.get("id"), **source},
                    "target_node": {"node_id": target.get("id"), **target},
                    "lane_pair": {
                        "source_node_id": source.get("id"),
                        "source_road_id": source.get("road_id"),
                        "source_lane_id": source.get("lane_id"),
                        "target_node_id": target.get("id"),
                        "target_road_id": target.get("road_id"),
                        "target_lane_id": target.get("lane_id"),
                        "direction": direction,
                    },
                    "community_score": round(community.score, 4),
                    "node_match_score": round(pair_score, 4),
                    "score": round(community.score + pair_score, 4),
                    "matched_constraints": source_detail["matched"] + ["same_direction_adjacent_lane_pair"],
                    "failed_constraints": [],
                    "limited_index": bool(getattr(community, "limited_index", False)),
                })
        return results

    def _heading_gap(self, heading_a: float, heading_b: float) -> float:
        gap = (heading_a - heading_b + 180.0) % 360.0 - 180.0
        return abs(gap)

    def _same_direction_adjacent(self, source: Dict[str, Any], target: Dict[str, Any], direction: str) -> bool:
        if int(source.get("road_id", -1)) != int(target.get("road_id", -2)):
            return False
        source_lane = int(source.get("lane_id", 0))
        target_lane = int(target.get("lane_id", 0))
        if source_lane * target_lane <= 0 or abs(source_lane - target_lane) != 1:
            return False
        if abs(self._heading_gap(float(source.get("heading", 0.0)), float(target.get("heading", 0.0)))) > 25.0:
            return False
        lane_change = str(source.get("lane_change", ""))
        if direction == "left" and lane_change not in {"Left", "Both"}:
            return False
        if direction == "right" and lane_change not in {"Right", "Both"}:
            return False
        return True

    def _load_community_file(self, path: Path, limited_index: bool) -> List[CommunityRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records: List[CommunityRecord] = []
        for raw in payload.get("communities", []):
            map_name = raw.get("map_name") or raw.get("map") or payload.get("map_name", "")
            node_ids = raw.get("node_ids") or raw.get("sample_nodes") or []
            structure = raw.get("structure", {})
            summary = raw.get("summary") or self._structure_summary(raw.get("community_id", ""), structure)
            applicable = raw.get("applicable_violations") or self._infer_applicable_violations(raw.get("community_id", ""), structure)
            record = CommunityRecord(
                community_id=raw.get("community_id", "unknown"),
                map_name=map_name,
                node_ids=node_ids,
                structure=structure,
                summary=summary,
                applicable_violations=applicable,
            )
            setattr(record, "limited_index", limited_index or "node_ids" not in raw)
            records.append(record)
        return records

    def _community_has_lane_pair(self, community: CommunityRecord, requirement: Dict[str, Any]) -> bool:
        try:
            nodes_by_id = self._graph_nodes_by_id(community.map_name)
            nodes = [nodes_by_id[node_id] for node_id in community.node_ids if node_id in nodes_by_id]
            direction = str(requirement.get("lane_change_direction", "left"))
            for source in nodes:
                for target in nodes:
                    if source is not target and self._same_direction_adjacent(source, target, direction):
                        return True
        except Exception:
            return False
        return False

    def _structural_fallback_match(self, requirement: Dict[str, Any], structure: Dict[str, Any]) -> bool:
        if requirement.get("same_direction_multi_lane") and int(structure.get("lane_count_max", 1)) >= 2:
            return True
        if requirement.get("has_traffic_light") and int(structure.get("traffic_lights", 0)) > 0:
            return True
        if requirement.get("needs_opposing_lanes") and int(structure.get("lane_count_max", 1)) >= 2:
            return True
        return False

    def _community_score(self, scenario_spec: Dict[str, Any], community: CommunityRecord) -> float:
        requirement = scenario_spec.get("road_requirement", {})
        structure = community.structure
        score = 0.0
        road_type = requirement.get("type", "RoadSegment")
        if road_type in {"Intersection", "T-intersection"}:
            score += 3.0 if structure.get("junctions", 0) > 0 else -2.0
        elif road_type == "Curve":
            score += 1.5
        else:
            score += 1.0
        effective_min_lanes = self._effective_min_lanes(requirement)
        if effective_min_lanes <= int(structure.get("lane_count_max", 1)):
            score += 2.0
        else:
            score -= 1.0
        if requirement.get("has_traffic_light"):
            score += 2.0 if structure.get("traffic_lights", 0) > 0 else -2.0
        elif road_type not in {"Intersection", "T-intersection"}:
            score += 0.5 if structure.get("traffic_lights", 0) == 0 else 0.0
        if requirement.get("needs_opposing_lanes") and structure.get("lane_count_max", 1) >= 2:
            score += 1.0
        if requirement.get("same_direction_multi_lane") and structure.get("lane_count_max", 1) >= 2:
            score += 1.0
            if self._community_has_lane_pair(community, requirement):
                score += 5.0
        summary_text = f"{community.summary} {community.community_id}".lower()
        violation_type = self._canonical_violation(scenario_spec.get("violation_type", ""))
        if violation_type and violation_type in [self._canonical_violation(item) for item in community.applicable_violations]:
            score += 2.0
        if "junction" in summary_text and road_type in {"Intersection", "T-intersection"}:
            score += 0.5
        if "multi_lane" in summary_text and effective_min_lanes >= 2:
            score += 0.5
        if "shoulder" in summary_text and requirement.get("has_shoulder"):
            score += 0.5
        return score

    def _match_requirement(self, attrs: Dict[str, Any], requirement: Dict[str, Any]) -> bool:
        matched, _, _ = self._score_node(attrs, requirement, {})
        return matched

    def _score_node(self, attrs: Dict[str, Any], requirement: Dict[str, Any], lane_index: Dict[int, List[int]]) -> Tuple[bool, Dict[str, List[str]], float]:
        matched: List[str] = []
        failed: List[str] = []
        score = 0.0
        expected_type = requirement.get("type", "RoadSegment")
        actual_type = self._road_type(attrs)
        if self._road_type_ok(expected_type, attrs):
            matched.append(f"road_type:{actual_type}")
            score += 2.0
        else:
            failed.append(f"road_type expected {expected_type}, actual {actual_type}")
        min_lanes = self._effective_min_lanes(requirement)
        if int(attrs.get("lane_count", 1)) >= min_lanes or (attrs.get("is_junction") and min_lanes <= 3):
            matched.append(f"lane_count>={min_lanes}")
            score += 1.5
        else:
            failed.append(f"lane_count<{min_lanes}")
        if requirement.get("has_traffic_light"):
            if attrs.get("has_traffic_light"):
                matched.append("has_traffic_light")
                score += 1.5
            else:
                failed.append("missing_traffic_light")
        if requirement.get("needs_opposing_lanes"):
            if self._has_opposing_lane(attrs, lane_index):
                matched.append("opposing_lanes")
                score += 1.0
            else:
                failed.append("missing_opposing_lanes")
        if requirement.get("same_direction_multi_lane"):
            if int(attrs.get("lane_count", 1)) >= 2 and attrs.get("lane_change") in {"Left", "Right", "Both"}:
                matched.append("same_direction_multi_lane")
                score += 1.0
            else:
                failed.append("missing_same_direction_multi_lane")
        required_length_m = float(requirement.get("min_segment_length_m", 25.0 if requirement.get("needs_long_straight") else 0.0))
        segment_length_m = self._segment_length(attrs)
        if required_length_m > 0.0:
            if segment_length_m >= required_length_m:
                matched.append(f"segment_length>={required_length_m:.1f}m")
                score += min(segment_length_m / max(required_length_m, 1.0), 3.0)
            else:
                failed.append(f"segment_length<{required_length_m:.1f}m")
        if requirement.get("needs_long_straight"):
            if not attrs.get("is_junction") and abs(float(attrs.get("curvature", 0.0))) < 0.05:
                matched.append("long_straight_candidate")
                score += 1.0
            else:
                failed.append("not_long_straight_candidate")
        if requirement.get("has_shoulder"):
            if attrs.get("has_shoulder"):
                matched.append("has_shoulder")
                score += 1.0
            else:
                failed.append("missing_shoulder")
        required_failures = [item for item in failed if not item.startswith("not_long_straight_candidate")]
        return not required_failures, {"matched": matched, "failed": failed}, score

    def _segment_length(self, attrs: Dict[str, Any]) -> float:
        start = attrs.get("start", {}) or {}
        end = attrs.get("end", {}) or {}
        try:
            return ((float(end.get("x", 0.0)) - float(start.get("x", 0.0))) ** 2 + (float(end.get("y", 0.0)) - float(start.get("y", 0.0))) ** 2) ** 0.5
        except (TypeError, ValueError):
            return 0.0

    def _effective_min_lanes(self, requirement: Dict[str, Any]) -> int:
        min_lanes = int(requirement.get("min_lanes", 1))
        if min_lanes >= 4:
            return max(2, min(3, (min_lanes + 1) // 2))
        return max(1, min_lanes)

    def _graph_nodes_by_id(self, map_name: str) -> Dict[str, Dict[str, Any]]:
        town = self._town_name(map_name)
        if town in self._node_cache:
            return self._node_cache[town]
        graph = self._load_graph(town)
        nodes = {node["id"]: node for node in graph.get("nodes", [])}
        self._node_cache[town] = nodes
        return nodes

    def _lane_index_by_road(self, map_name: str) -> Dict[int, List[int]]:
        lanes: Dict[int, List[int]] = {}
        for node in self._graph_nodes_by_id(map_name).values():
            lanes.setdefault(int(node.get("road_id", 0)), []).append(int(node.get("lane_id", 0)))
        return lanes

    def _load_graph(self, map_name: str) -> Dict[str, Any]:
        town = self._town_name(map_name)
        if town in self._graph_cache:
            return self._graph_cache[town]
        path = self.maps_dir / f"Carla_Maps_{town}_graph.json"
        if not path.exists():
            return {"nodes": []}
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._graph_cache[town] = payload
        return payload

    def _road_type(self, attrs: Dict[str, Any]) -> str:
        if attrs.get("is_junction"):
            return "Intersection"
        if abs(float(attrs.get("curvature", 0.0))) > 0.05:
            return "Curve"
        return "Straight"

    def _road_type_ok(self, expected: str, attrs: Dict[str, Any]) -> bool:
        actual = self._road_type(attrs)
        if expected in {"RoadSegment", "Any", "unknown"}:
            return not attrs.get("is_junction", False)
        if expected in {"Intersection", "T-intersection"}:
            return actual == "Intersection"
        if expected == "Curve":
            return actual == "Curve"
        if expected == "Straight":
            return actual == "Straight"
        return True

    def _has_opposing_lane(self, attrs: Dict[str, Any], lane_index: Dict[int, List[int]]) -> bool:
        road_id = int(attrs.get("road_id", 0))
        lane_id = int(attrs.get("lane_id", 0))
        if attrs.get("is_junction"):
            return True
        return any(item * lane_id < 0 for item in lane_index.get(road_id, []))

    def _infer_applicable_violations(self, community_id: str, structure: Dict[str, Any]) -> List[str]:
        labels = {"未保持安全距离", "未注意前方路况", "超速行驶"}
        if structure.get("junctions", 0) > 0:
            labels.update({"闯红灯", "未按规定让行", "违反交通信号（其他）"})
        if int(structure.get("lane_count_max", 1)) >= 2:
            labels.update({"违规变道", "违规超车", "逆行"})
        if structure.get("shoulders", 0) > 0 or "shoulder" in community_id:
            labels.add("违法占用应急车道")
        return sorted(labels)

    def _structure_summary(self, community_id: str, structure: Dict[str, Any]) -> str:
        return (
            f"{community_id}: junctions={structure.get('junctions', 0)}, "
            f"traffic_lights={structure.get('traffic_lights', 0)}, "
            f"lane_count_max={structure.get('lane_count_max', 1)}, "
            f"shoulders={structure.get('shoulders', 0)}"
        )

    def _community_export(self, community: CommunityRecord) -> Dict[str, Any]:
        return {
            "community_id": community.community_id,
            "map_name": community.map_name,
            "score": round(community.score, 4),
            "structure": community.structure,
            "summary": community.summary,
            "applicable_violations": community.applicable_violations,
            "node_count": len(community.node_ids),
            "limited_index": bool(getattr(community, "limited_index", False)),
        }

    def _canonical_violation(self, violation_type: str) -> str:
        if violation_type == "超速":
            return "超速行驶"
        if violation_type == "违反交通信号(其他)":
            return "违反交通信号（其他）"
        return violation_type

    def _town_name(self, map_name: str) -> str:
        return str(map_name).split("/")[-1]
