import json
from typing import Any, Dict

from RKGRScen.llm_client import DeepSeekClient

class SemanticMatcher:
    def __init__(self) -> None:
        self.client = DeepSeekClient()

    def judge(self, organized_scenario: Dict[str, Any], scene_spec: Dict[str, Any], scenario_config: Dict[str, Any], violation_result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client.enabled:
            return {
                "enabled": False,
                "matched": None,
                "score": None,
                "reason": "DEEPSEEK_API_KEY 未设置，跳过大模型语义判定",
            }
        system_prompt = (
            "你是自动驾驶违规场景语义一致性审查器。"
            "你只输出 JSON，不输出额外解释。"
            "请判断生成的 CARLA 场景是否符合原始自然语言原因、法规依据和违规类型。"
            "输出字段必须包含 matched(boolean), score(0-1), reasons(array), risks(array), recommendation(string)。"
        )
        user_prompt = (
            f"原始 organized_scenario:\n{json.dumps(organized_scenario, ensure_ascii=False, indent=2)}\n"
            f"语义展开 scene_spec:\n{json.dumps(scene_spec, ensure_ascii=False, indent=2)}\n"
            f"生成配置 scenario_config:\n{json.dumps(scenario_config, ensure_ascii=False, indent=2)}\n"
            f"检测结果 violation_result:\n{json.dumps(violation_result, ensure_ascii=False, indent=2)}\n"
            "请重点检查：1) ego 是否是被诱发的自动驾驶车辆；2) 是否没有直接控制 ego 违规；"
            "3) 车辆运动是否符合物理约束；4) 是否贴合自然语言 reason 和 law。"
        )
        result = self.client.generate_json(system_prompt, user_prompt)
        return {
            "enabled": True,
            "matched": bool(result.get("matched", False)),
            "score": float(result.get("score", 0.0)),
            "reasons": result.get("reasons", []),
            "risks": result.get("risks", []),
            "recommendation": result.get("recommendation", ""),
        }
