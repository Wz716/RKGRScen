import json
import os
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

# 论文定义的 7 类违规类型（英文规范名，与冻结 benchmark 的 violation_type 字段一致）。
VIOLATION_TYPES = (
    "Inattention to the road ahead",
    "Failure to yield",
    "Speeding",
    "Failure to maintain safe following distance",
    "Wrong-way driving",
    "Illegal lane change",
    "Illegal overtaking",
)

# 英文规范名 -> 短键（用于 detector_thresholds.json / violation_map.json 的键名）。
VIOLATION_TYPE_KEYS = {
    "Inattention to the road ahead": "inattention",
    "Failure to yield": "failure_to_yield",
    "Speeding": "speeding",
    "Failure to maintain safe following distance": "following_distance",
    "Wrong-way driving": "wrong_way",
    "Illegal lane change": "illegal_lane_change",
    "Illegal overtaking": "illegal_overtaking",
}

# 旧中文类型名 -> 英文规范名，用于向后兼容历史输入。
VIOLATION_TYPE_ALIASES = {
    "超速": "Speeding",
    "超速行驶": "Speeding",
    "逆行": "Wrong-way driving",
    "违规变道": "Illegal lane change",
    "违规超车": "Illegal overtaking",
    "未按规定让行": "Failure to yield",
    "未保持安全距离": "Failure to maintain safe following distance",
    "未注意前方路况": "Inattention to the road ahead",
}


def canonical_violation_type(value: Any) -> str:
    """把任意输入（英文规范名或旧中文别名）归一化为论文 7 类英文名。"""
    text = str(value or "").strip()
    if text in VIOLATION_TYPES:
        return text
    return VIOLATION_TYPE_ALIASES.get(text, text)


def violation_key(value: Any) -> str:
    """返回违规类型的短键；未知类型返回原字符串。"""
    return VIOLATION_TYPE_KEYS.get(canonical_violation_type(value), str(value or ""))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def violation_map() -> Dict[str, Any]:
    return load_json(CONFIG_DIR / "violation_map.json")

def detector_thresholds() -> Dict[str, Any]:
    return load_json(CONFIG_DIR / "detector_thresholds.json")

def llm_settings() -> Dict[str, Any]:
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "model_version": os.getenv("DEEPSEEK_MODEL_VERSION", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        "temperature": 0.2,
        "timeout_s": float(os.getenv("DEEPSEEK_TIMEOUT_S", "60")),
        "max_retries": int(os.getenv("DEEPSEEK_MAX_RETRIES", "2")),
        "retry_backoff_s": float(os.getenv("DEEPSEEK_RETRY_BACKOFF_S", "0.5")),
        "audit_jsonl": os.getenv("DEEPSEEK_AUDIT_JSONL", ""),
        "enabled": bool(os.getenv("DEEPSEEK_API_KEY")),
    }
