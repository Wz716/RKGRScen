import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from RKGRScen.indexing.community_detector import CommunityDetector
from RKGRScen.indexing.community_tagger import CommunityTagger
from RKGRScen.indexing.graph_builder import RoadGraphBuilder
from RKGRScen.models import CommunityRecord
from RKGRScen.query.constraint_solver import ConstraintSolver
from RKGRScen.query.retriever import GraphRetriever
from RKGRScen.query.scene_expander import SceneExpander
from RKGRScen.execution.carla_runner import CarlaScenarioRunner
from RKGRScen.execution.violation_detector import ViolationDetector

class RKGRScenPipeline:
    def __init__(self) -> None:
        self.graph_builder = RoadGraphBuilder()
        self.community_detector = CommunityDetector()
        self.community_tagger = CommunityTagger()
        self.scene_expander = SceneExpander()
        self.retriever = GraphRetriever()
        self.constraint_solver = ConstraintSolver()
        self.runner = CarlaScenarioRunner()
        self.detector = ViolationDetector()

    def build_index(self, map_name: str, lane_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        graph = self.graph_builder.build_from_records(map_name, lane_records)
        communities = self.community_detector.detect(graph)
        tagged = self.community_tagger.tag(communities)
        return {"graph": graph, "communities": tagged}

    def generate(self, logical_scenario: Dict[str, Any], graph, communities: List[CommunityRecord]) -> Dict[str, Any]:
        scene_spec = self.scene_expander.expand(logical_scenario)
        candidate_communities = self.retriever.global_search(scene_spec, communities)
        local_results = self.retriever.local_search(graph, scene_spec, candidate_communities)
        scenario = self.constraint_solver.solve(scene_spec, local_results)
        trace = self.runner.run(scenario)
        result = self.detector.detect(
            scene_spec["violation_type"],
            trace,
            scenario.expected_violation.get("params", {}),
        )
        return {
            "scene_spec": scene_spec,
            "llm_metadata": dict(self.scene_expander.audit_metadata),
            "scenario_config": scenario.to_dict(),
            "execution_trace": trace.to_dict(),
            "violation_result": result.to_dict(),
        }

def dump_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
