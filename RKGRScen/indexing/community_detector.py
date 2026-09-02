from typing import Dict, List, Sequence, Set

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from RKGRScen.models import CommunityRecord

try:
    from networkx.algorithms.community import louvain_communities
except ImportError:
    louvain_communities = None

class CommunityDetector:

    def __init__(self, method: str = "auto", resolution: float = 1.0, min_community_size: int = 5) -> None:
        self.method = method
        self.resolution = resolution
        self.min_community_size = min_community_size
        self.last_method = "uninitialized"

    def detect(self, graph: nx.DiGraph) -> List[CommunityRecord]:
        undirected = self._prepare_graph(graph)
        partitions = self._cluster(undirected)
        partitions = self._merge_small_communities(undirected, partitions)
        partitions.sort(key=len, reverse=True)
        map_name = graph.graph.get("map_name", "UnknownTown")

        communities: List[CommunityRecord] = []
        for index, nodes in enumerate(partitions, start=1):
            sorted_nodes = sorted(nodes)
            structure = self._summarize_structure(graph, sorted_nodes)
            label = self._label_community(sorted_nodes, structure)
            communities.append(
                CommunityRecord(
                    community_id=f"{map_name}_{label}_{index}",
                    map_name=map_name,
                    node_ids=sorted_nodes,
                    structure=structure,
                )
            )
        return communities

    def _prepare_graph(self, graph: nx.DiGraph) -> nx.Graph:
        undirected = nx.Graph()
        for node_id, attrs in graph.nodes(data=True):
            undirected.add_node(node_id, **attrs)
        for source, target, attrs in graph.edges(data=True):
            weight = self._edge_weight(graph, source, target, attrs)
            if undirected.has_edge(source, target):
                undirected[source][target]["weight"] += weight
            else:
                undirected.add_edge(source, target, weight=weight)
        return undirected

    def _cluster(self, graph: nx.Graph) -> List[Set[str]]:
        if graph.number_of_nodes() == 0:
            self.last_method = "empty"
            return []

        requested = self.method.lower()
        if requested in {"auto", "louvain"} and louvain_communities is not None:
            self.last_method = "networkx_louvain"
            return list(louvain_communities(graph, weight="weight", resolution=self.resolution, seed=42))

        self.last_method = "greedy_modularity"
        return [set(nodes) for nodes in greedy_modularity_communities(graph, weight="weight")]

    def _merge_small_communities(self, graph: nx.Graph, partitions: List[Set[str]]) -> List[Set[str]]:
        communities = [set(nodes) for nodes in partitions if nodes]
        if not communities:
            return []

        changed = True
        while changed:
            changed = False
            for index, community in enumerate(list(communities)):
                if len(community) >= self.min_community_size:
                    continue
                target_index = self._best_merge_target(graph, communities, index)
                if target_index is None:
                    continue
                communities[target_index].update(community)
                communities.pop(index)
                changed = True
                break
        return communities

    def _best_merge_target(self, graph: nx.Graph, communities: List[Set[str]], source_index: int) -> int:
        source_nodes = communities[source_index]
        best_index = None
        best_score = -1.0
        for target_index, target_nodes in enumerate(communities):
            if target_index == source_index:
                continue
            score = self._community_affinity(graph, source_nodes, target_nodes)
            if score > best_score:
                best_score = score
                best_index = target_index
        return best_index

    def _community_affinity(self, graph: nx.Graph, source_nodes: Set[str], target_nodes: Set[str]) -> float:
        score = 0.0
        for node in source_nodes:
            for neighbor, attrs in graph[node].items():
                if neighbor not in target_nodes:
                    continue
                score += float(attrs.get("weight", 1.0))
        return score

    def _edge_weight(self, graph: nx.DiGraph, source: str, target: str, attrs: Dict[str, object]) -> float:
        source_attrs = graph.nodes[source]
        target_attrs = graph.nodes[target]
        weight = float(attrs.get("weight", 1.0))
        if source_attrs.get("road_id") == target_attrs.get("road_id"):
            weight += 2.5
        if source_attrs.get("section_id") == target_attrs.get("section_id"):
            weight += 1.0
        if source_attrs.get("lane_id") == target_attrs.get("lane_id"):
            weight += 1.0
        if source_attrs.get("is_junction") and target_attrs.get("is_junction"):
            weight += 1.5
        if source_attrs.get("lane_count") == target_attrs.get("lane_count"):
            weight += 0.5
        if attrs.get("connection_type") == "junction":
            weight += 0.5
        return weight

    def _label_community(self, nodes: Sequence[str], structure: Dict[str, int]) -> str:
        node_count = max(structure.get("node_count", 1), 1)
        junction_ratio = structure.get("junctions", 0) / float(node_count)
        shoulder_ratio = structure.get("shoulders", 0) / float(node_count)
        lane_count_max = structure.get("lane_count_max", 1)
        if junction_ratio >= 0.6:
            return "junction_cluster"
        if shoulder_ratio >= 0.3:
            return "shoulder_cluster"
        if lane_count_max >= 3:
            return "multi_lane_cluster"
        return "road_cluster"

    def _summarize_structure(self, graph: nx.DiGraph, nodes: List[str]) -> Dict[str, int]:
        junctions = 0
        traffic_lights = 0
        shoulders = 0
        lane_count_max = 1
        for node in nodes:
            attrs = graph.nodes[node]
            junctions += int(bool(attrs.get("is_junction")))
            traffic_lights += int(bool(attrs.get("has_traffic_light")))
            shoulders += int(bool(attrs.get("has_shoulder")))
            lane_count_max = max(lane_count_max, int(attrs.get("lane_count", 1)))
        return {
            "junctions": junctions,
            "traffic_lights": traffic_lights,
            "shoulders": shoulders,
            "lane_count_max": lane_count_max,
            "node_count": len(nodes),
        }
