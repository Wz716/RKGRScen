import json
import os
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

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
