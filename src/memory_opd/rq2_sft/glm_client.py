"""Minimal GLM chat-completions client with strict memory validation."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping

import httpx


GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MEMORY_V1_KEYS = {"stable", "current", "changes"}
MEMORY_V2_KEYS = {"stable", "current", "past_experiences", "changes"}
EVIDENCE_V1_KEYS = {"evidence", "temporal_relations", "derived_facts"}


def validate_memory(value: object) -> dict[str, Any]:
    if isinstance(value, dict) and EVIDENCE_V1_KEYS <= set(value):
        normalized = {field: value[field] for field in EVIDENCE_V1_KEYS}
        for field in sorted(EVIDENCE_V1_KEYS):
            if not isinstance(normalized[field], list) or not all(
                isinstance(item, str) for item in normalized[field]
            ):
                raise ValueError(
                    f"evidence v1 {field} must be a list of strings")
        return normalized
    if not isinstance(value, dict) or set(value) not in (
        MEMORY_V1_KEYS,
        MEMORY_V2_KEYS,
    ):
        raise ValueError(
            "memory must match a supported schema; expected keys "
            f"{sorted(MEMORY_V1_KEYS)}, {sorted(MEMORY_V2_KEYS)}, or "
            f"{sorted(EVIDENCE_V1_KEYS)}"
        )
    if not isinstance(value["stable"], list) or not all(
        isinstance(item, str) for item in value["stable"]
    ):
        raise ValueError("stable must be a list of strings")
    if set(value) == MEMORY_V1_KEYS:
        if not isinstance(value["current"], list) or not all(
            isinstance(item, str) for item in value["current"]
        ):
            raise ValueError("v1 current must be a list of strings")
    else:
        fact_keys = {"topic", "fact"}
        if not isinstance(value["current"], list):
            raise ValueError("current must be a list")
        current_is_strings = all(isinstance(item, str)
                                 for item in value["current"])
        current_is_records = all(
            isinstance(item, dict)
            and set(item) == fact_keys
            and all(isinstance(item[key], str) for key in fact_keys)
            for item in value["current"]
        )
        if not current_is_strings and not current_is_records:
            raise ValueError(
                "v2 current must contain only strings or only topic/fact records")
        for field in ("past_experiences",):
            if not isinstance(value[field], list):
                raise ValueError(f"{field} must be a list")
            for item in value[field]:
                if not isinstance(item, dict) or set(item) != fact_keys:
                    raise ValueError(
                        f"each {field} item must contain exactly {sorted(fact_keys)}")
                if not all(isinstance(item[key], str) for key in fact_keys):
                    raise ValueError(f"{field} fields must be strings")
    if not isinstance(value["changes"], list):
        raise ValueError("changes must be a list")
    change_keys = {"topic", "previous", "current", "reason"}
    for change in value["changes"]:
        if not isinstance(change, dict) or set(change) != change_keys:
            raise ValueError(
                f"each change must contain exactly {sorted(change_keys)}")
        if not all(isinstance(change[key], str) for key in change_keys):
            raise ValueError("change fields must be strings")
    return value


def parse_glm_response(response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "GLM response is missing assistant content") from error
    if not isinstance(content, str):
        raise ValueError("GLM assistant content must be a string")
    try:
        memory = validate_memory(json.loads(content))
    except json.JSONDecodeError as error:
        raise ValueError("GLM assistant content is not valid JSON") from error
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return memory, dict(usage)


def call_glm(
    *,
    api_key: str,
    messages: list[dict[str, str]],
    model: str = "glm-5.2",
    max_tokens: int = 2048,
    temperature: float = 0.1,
    retries: int = 4,
    timeout_seconds: float = 180.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = httpx.post(
                GLM_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            raw = response.json()
            memory, usage = parse_glm_response(raw)
            metadata = {
                "model": raw.get("model", model),
                "request_id": raw.get("id"),
                "finish_reason": raw.get("choices", [{}])[0].get("finish_reason"),
                "attempts": attempt + 1,
            }
            return memory, usage, metadata
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as error:
            last_error = error
            retryable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
            }
            if attempt >= retries or not retryable:
                break
            sleep(min(2**attempt, 16))
    raise RuntimeError(
        f"GLM request failed after {retries + 1} attempts: {last_error}")
