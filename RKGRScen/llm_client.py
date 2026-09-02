import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from RKGRScen.config import llm_settings

class DeepSeekClient:
    TEMPERATURE = 0.2

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        audit_jsonl: Optional[Path] = None,
    ) -> None:
        settings = llm_settings()
        self.api_key = settings["api_key"]
        self.base_url = settings["base_url"].rstrip("/")
        self.model = model or settings["model"]
        self.model_version = settings["model_version"] if model is None else model
        if temperature is not None and float(temperature) != self.TEMPERATURE:
            raise ValueError("RQ1/RQ2 DeepSeek temperature 固定为 0.2")
        self.temperature = self.TEMPERATURE
        self.timeout_s = settings["timeout_s"]
        self.max_retries = settings["max_retries"] if max_retries is None else int(max_retries)
        self.retry_backoff_s = settings["retry_backoff_s"]
        configured_audit = settings.get("audit_jsonl", "")
        self.audit_jsonl = Path(audit_jsonl) if audit_jsonl else (Path(configured_audit) if configured_audit else None)
        self.enabled = settings["enabled"]
        self.last_metadata: Dict[str, Any] = {}

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            self.last_metadata = self._metadata(0, "disabled", 0.0, audit_metadata)
            self._audit(self.last_metadata)
            raise RuntimeError("DEEPSEEK_API_KEY 未设置，无法调用真实 LLM")
        validator = Draft202012Validator(schema) if schema else None
        last_error: Optional[Exception] = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                raw = self._request(system_prompt, user_prompt, schema)
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                result = self._extract_json(content)
                if validator:
                    validator.validate(result)
                elapsed = time.perf_counter() - started
                self.last_metadata = self._metadata(attempt, "success", elapsed, audit_metadata)
                self._audit(self.last_metadata)
                return result
            except (error.HTTPError, error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                status = self._failure_status(exc)
                metadata = self._metadata(attempt, status, time.perf_counter() - started, audit_metadata)
                metadata["error_type"] = type(exc).__name__
                metadata["error"] = self._safe_error(exc)
                self.last_metadata = metadata
                self._audit(metadata)
                if attempt < attempts and self.retry_backoff_s > 0:
                    time.sleep(self.retry_backoff_s * attempt)
        raise RuntimeError(f"DeepSeek 调用在 {attempts} 次有界尝试后失败: {self._safe_error(last_error)}") from last_error

    def _request(self, system_prompt: str, user_prompt: str, schema: Optional[Dict[str, Any]]) -> str:
        response_format: Dict[str, Any] = {"type": "json_object"}
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "RKGRScen_response", "strict": True, "schema": schema},
            }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": response_format,
        }
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise error.HTTPError(exc.url, exc.code, detail, exc.headers, exc.fp) from exc

    def _extract_json(self, content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        result = json.loads(content)
        if not isinstance(result, dict):
            raise TypeError("LLM JSON 顶层必须是对象")
        return result

    def _metadata(self, attempt: int, status: str, elapsed_s: float, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        metadata = dict(extra or {})
        purpose = str(metadata.pop("purpose", metadata.pop("component", "unspecified")))
        safe_metadata = {key: value for key, value in metadata.items() if "key" not in key.lower() and "secret" not in key.lower()}
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "purpose": purpose,
            "enabled": self.enabled,
            "attempts": attempt,
            "max_attempts": self.max_retries + 1,
            "elapsed_s": round(elapsed_s, 6),
            "success": status == "success",
            "status": status,
            **safe_metadata,
        }

    @staticmethod
    def _failure_status(exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            return "schema_error"
        if isinstance(exc, (json.JSONDecodeError, KeyError, IndexError, TypeError)):
            return "json_parse_error"
        return "network_error"

    def _audit(self, record: Dict[str, Any]) -> None:
        if not self.audit_jsonl:
            return
        self.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_error(exc: Optional[Exception]) -> str:
        if exc is None:
            return "unknown error"
        text = str(exc)
        return text[:1000]
