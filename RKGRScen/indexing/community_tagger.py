from typing import Dict, Iterable, List

from RKGRScen.config import violation_map
from RKGRScen.llm_client import DeepSeekClient
from RKGRScen.models import CommunityRecord

class CommunityTagger:

    def __init__(self, use_llm: bool = False) -> None:
        self.violation_knowledge = violation_map()
        self.client = DeepSeekClient()
        self.use_llm = use_llm and self.client.enabled
        self.audit_metadata: List[Dict[str, object]] = []
        allowed = list(self.violation_knowledge.keys())
        self.output_schema: Dict[str, object] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "applicable_violations"],
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "applicable_violations": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {"type": "string", "enum": allowed},
                },
            },
        }

    def tag(self, communities: Iterable[CommunityRecord]) -> List[CommunityRecord]:
        tagged: List[CommunityRecord] = []
        for community in communities:
            structure = dict(community.structure)
            derived = self._derive_features(structure)
            structure.update(derived)
            summary = self._build_summary(structure)
            applicable = self._match_violations(structure)
            if self.use_llm:
                llm_output = self._llm_tag(community.map_name, community.community_id, structure)
                summary = str(llm_output["summary"])
                applicable = self._normalize_violations(llm_output["applicable_violations"])
                self.audit_metadata.append(dict(self.client.last_metadata))
            community.structure = structure
            community.summary = summary
            community.applicable_violations = applicable
            tagged.append(community)
        return tagged

    def build_index_records(self, communities: Iterable[CommunityRecord]) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for community in communities:
            records.append(
                {
                    "community_id": community.community_id,
                    "map": community.map_name,
                    "structure": community.structure,
                    "summary": community.summary,
                    "applicable_violations": community.applicable_violations,
                    "node_ids": community.node_ids,
                }
            )
        return records

    def _llm_tag(self, map_name: str, community_id: str, structure: Dict[str, object]) -> Dict[str, object]:
        allowed = list(self.violation_knowledge.keys())
        system_prompt = (
            "你是自动驾驶测试领域的道路社区语义标注器。"
            "你必须只输出 JSON，不要输出任何解释。"
            "applicable_violations 只能从给定候选集合中选择。"
        )
        user_prompt = (
            f"地图: {map_name}\n"
            f"社区ID: {community_id}\n"
            f"结构统计: {structure}\n"
            f"可选违规类型: {allowed}\n"
            "请输出 JSON，格式为: "
            '{"summary": "...", "applicable_violations": ["..."]}'
        )
        return self.client.generate_json(
            system_prompt,
            user_prompt,
            schema=self.output_schema,
            audit_metadata={"component": "community_tagger", "map": map_name, "community_id": community_id},
        )

    def _derive_features(self, structure: Dict[str, int]) -> Dict[str, object]:
        node_count = max(int(structure.get("node_count", 0)), 1)
        junctions = int(structure.get("junctions", 0))
        traffic_lights = int(structure.get("traffic_lights", 0))
        shoulders = int(structure.get("shoulders", 0))
        lane_count_max = int(structure.get("lane_count_max", 1))

        junction_ratio = round(junctions / float(node_count), 3)
        shoulder_ratio = round(shoulders / float(node_count), 3)
        traffic_ratio = round(traffic_lights / float(node_count), 3)
        has_complex_intersection = junction_ratio >= 0.5 and lane_count_max >= 2
        supports_lane_interaction = lane_count_max >= 2
        is_shoulder_dominant = shoulder_ratio >= 0.3

        return {
            "junction_ratio": junction_ratio,
            "shoulder_ratio": shoulder_ratio,
            "traffic_light_ratio": traffic_ratio,
            "has_complex_intersection": has_complex_intersection,
            "supports_lane_interaction": supports_lane_interaction,
            "is_shoulder_dominant": is_shoulder_dominant,
        }

    def _build_summary(self, structure: Dict[str, object]) -> str:
        parts = [f"该社区包含 {structure.get('node_count', 0)} 个车道段"]
        if structure.get("junctions", 0):
            parts.append(f"其中 {structure['junctions']} 个位于路口区域")
        if structure.get("traffic_lights", 0):
            parts.append(f"有 {structure['traffic_lights']} 个信号相关车道段")
        if structure.get("shoulders", 0):
            parts.append(f"包含 {structure['shoulders']} 个带路肩属性的车道段")
        parts.append(f"最大车道数为 {structure.get('lane_count_max', 1)}")
        if structure.get("has_complex_intersection"):
            parts.append("整体更接近复杂路口交互区域")
        elif structure.get("supports_lane_interaction"):
            parts.append("适合多车道并行、变道和超车类交互")
        else:
            parts.append("整体更接近普通道路或连接道路")
        if structure.get("is_shoulder_dominant"):
            parts.append("路肩或应急车道特征较明显")
        return "，".join(str(item) for item in parts) + "。"

    def _match_violations(self, structure: Dict[str, object]) -> List[str]:
        applicable: List[str] = []
        if structure.get("junctions", 0) > 0:
            applicable.extend(["未按规定让行", "闯红灯", "违规掉头", "违反交通信号(其他)"])
        if structure.get("lane_count_max", 1) >= 2:
            applicable.extend(["违规变道", "违规超车"])
        if structure.get("shoulders", 0) > 0:
            applicable.append("违法占用应急车道")
        applicable.extend(["超速行驶", "未保持安全距离", "未注意前方路况", "逆行"])
        return self._normalize_violations(applicable)

    def _normalize_violations(self, values: List[str]) -> List[str]:
        ordered: List[str] = []
        for item in values:
            if item in self.violation_knowledge and item not in ordered:
                ordered.append(item)
        return ordered
