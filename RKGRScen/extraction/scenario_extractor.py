"""LLM_extract: 从 NHTSA 事故叙事提取结构化逻辑场景。

对应论文 "Structured Extraction from Functional Scenarios"（第 155-160 行）。

输入为一段自然语言事故叙事，输出为逻辑场景四元组
    L = (V_type, A, R_req, E_cond)
映射到 organized_scenario 格式：
    - V_type  -> violation_type（目标违规类型）
    - A       -> dsl.actors（参与者：model/initial_position/actions/speed_limit）
    - R_req   -> dsl.road_network（粗粒度道路需求：type/lanes）
    - E_cond  -> dsl.env（天气与时间：weather/time）

论文约定：LLM_extract 最多 3 次重试（区别于 LLM_index 的 2 次）；
schema 校验失败在全部重试后仍失败者，记为 persistent extraction failure，
并从 benchmark 中排除。骨架代码不实现筛选/统计，仅暴露单条提取入口，
由调用方决定如何记录与汇总失败。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from RKGRScen.config import violation_map
from RKGRScen.llm_client import DeepSeekClient

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extract_prompt.txt"
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "extract_schema.json"


class ScenarioExtractor:
    """LLM_extract 的结构化提取器骨架。

    与 CommunityTagger / SceneExpander 保持一致的接口风格：
    `use_llm` 控制是否真正调用托管 LLM；`output_schema` 由外部
    JSON schema 文件加载，`prompt` 由外部模板加载，便于论文要求的
    "complete prompts and JSON schemas" 独立发布。
    """

    # 论文规定 LLM_extract 最多 3 次重试（LLM_index 为 2 次）。
    MAX_RETRIES = 3

    def __init__(self, use_llm: bool = False) -> None:
        self.knowledge = violation_map()
        self.client = DeepSeekClient(max_retries=self.MAX_RETRIES)
        self.use_llm = use_llm and self.client.enabled
        self.audit_metadata: Dict[str, Any] = {}
        self.output_schema: Dict[str, Any] = self._load_schema()
        self.system_prompt, self.user_prompt_template = self._load_prompt()

    def extract(
        self,
        narrative: str,
        *,
        report_id: str = "",
    ) -> Dict[str, Any]:
        """从单条事故叙事提取结构化逻辑场景。

        Args:
            narrative: 原始事故叙事文本。
            report_id: 源报告标识，仅用于审计记录，不进入输出。

        Returns:
            organized_scenario 格式的 dict（violation_type / likelihood /
            reason / law / dsl）。

        Raises:
            RuntimeError: 当 LLM 不可用或全部重试后仍失败时抛出；
                调用方应将其记为 persistent extraction failure。
        """
        if not self.use_llm:
            raise RuntimeError(
                "LLM_extract 需要已配置 DEEPSEEK_API_KEY 才能运行；"
                "未配置时无法提取，应记为 persistent extraction failure"
            )

        allowed = list(self.knowledge.keys())
        user_prompt = self.user_prompt_template.format(
            narrative=narrative,
            allowed_violation_types=allowed,
        )
        result = self.client.generate_json(
            self.system_prompt,
            user_prompt,
            schema=self.output_schema,
            audit_metadata={
                "component": "scenario_extractor",
                "report_id": report_id,
            },
        )
        self.audit_metadata = dict(self.client.last_metadata)
        return result

    # ------------------------------------------------------------------
    # 内部：加载 schema / prompt 文件
    # ------------------------------------------------------------------
    def _load_schema(self) -> Dict[str, Any]:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        # 与 violation_map.json 保持同步，避免 schema 文件与代码枚举漂移。
        schema["properties"]["violation_type"]["enum"] = list(self.knowledge.keys())
        return schema

    def _load_prompt(self) -> tuple[str, str]:
        text = _PROMPT_PATH.read_text(encoding="utf-8")
        system_lines: List[str] = []
        user_lines: List[str] = []
        current = system_lines
        for line in text.splitlines():
            if line.strip() == "SYSTEM:":
                current = system_lines
                continue
            if line.strip() == "USER:":
                current = user_lines
                continue
            current.append(line)
        return "\n".join(system_lines).strip(), "\n".join(user_lines).strip()
