from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class CommunityRecord:
    community_id: str
    map_name: str
    node_ids: List[str]
    structure: Dict[str, Any]
    summary: str = ""
    applicable_violations: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RetrievalResult:
    community: CommunityRecord
    matched_nodes: List[Dict[str, Any]]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "community": self.community.to_dict(),
            "matched_nodes": self.matched_nodes,
            "score": self.score,
        }

@dataclass
class ScenarioConfiguration:
    scenario_id: str
    violation_type: str
    map_name: str
    ego: Dict[str, Any]
    npcs: List[Dict[str, Any]]
    conflict_point: Dict[str, Any]
    environment: Dict[str, Any]
    expected_violation: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExecutionTrace:
    scenario_id: str
    ticks: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ViolationResult:
    scenario_id: str
    violation_type: str
    detected: bool
    reason: str
    timestamp_s: Optional[float] = None
    location: Optional[Dict[str, float]] = None
    severity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
