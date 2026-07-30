"""Pluggable inference adapters with no provider-specific scoring behavior."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from canyonbench.exceptions import DataValidationError
from canyonbench.schemas import ModelConfig


@dataclass(frozen=True)
class AdapterResponse:
    content: str
    input_tokens: int
    output_tokens: int
    provider_request_id: str | None
    raw_finish_reason: str | None


class Adapter(ABC):
    @abstractmethod
    def complete(
        self,
        *,
        image_path: Path | None,
        system: str,
        prompt: str,
        json_schema: dict[str, Any],
        model: ModelConfig,
    ) -> AdapterResponse:
        """Return one raw model response."""


def encode_image(path: Path, max_side: int) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAICompatibleAdapter(Adapter):
    def __init__(self, config: ModelConfig) -> None:
        adapter = config.adapter
        if not adapter.base_url:
            raise DataValidationError("openai_compatible adapter requires base_url")
        if not adapter.api_key_env:
            raise DataValidationError("openai_compatible adapter requires api_key_env")
        api_key = os.environ.get(adapter.api_key_env)
        if not api_key:
            raise DataValidationError(
                f"Required API key environment variable is unset: {adapter.api_key_env}"
            )
        self.config = config
        self.client = httpx.Client(
            base_url=adapter.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=adapter.timeout_s,
        )

    def complete(
        self,
        *,
        image_path: Path | None,
        system: str,
        prompt: str,
        json_schema: dict[str, Any],
        model: ModelConfig,
    ) -> AdapterResponse:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            image_b64 = encode_image(image_path, model.image_max_side)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        }
        if model.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "canyonbench_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        retries = model.adapter.max_retries
        for attempt in range(retries + 1):
            try:
                response = self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                value = response.json()
                choice = value["choices"][0]
                usage = value.get("usage", {})
                return AdapterResponse(
                    content=str(choice["message"]["content"]),
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                    provider_request_id=response.headers.get("x-request-id") or value.get("id"),
                    raw_finish_reason=choice.get("finish_reason"),
                )
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                if attempt >= retries:
                    raise DataValidationError(
                        f"Inference failed after {retries + 1} attempts: {exc}"
                    ) from exc
                time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


class FixtureAdapter(Adapter):
    """Deterministic adapter for tests and offline harness validation."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses

    def complete(
        self,
        *,
        image_path: Path | None,
        system: str,
        prompt: str,
        json_schema: dict[str, Any],
        model: ModelConfig,
    ) -> AdapterResponse:
        key = f"{image_path.name if image_path else 'NO_IMAGE'}:{prompt}"
        content = self.responses.get(key, self.responses.get(prompt))
        if content is None:
            raise DataValidationError(f"No fixture response for {key}")
        return AdapterResponse(content, 0, 0, "fixture", "stop")


class HttpDetectorAdapter(Adapter):
    """Contract for a non-language detector/segmenter served over local or remote HTTP."""

    def __init__(self, config: ModelConfig) -> None:
        adapter = config.adapter
        if not adapter.base_url:
            raise DataValidationError("http_detector adapter requires base_url")
        headers: dict[str, str] = {}
        if adapter.api_key_env:
            api_key = os.environ.get(adapter.api_key_env)
            if not api_key:
                raise DataValidationError(
                    f"Required API key environment variable is unset: {adapter.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            base_url=adapter.base_url.rstrip("/"), headers=headers, timeout=adapter.timeout_s
        )

    def complete(
        self,
        *,
        image_path: Path | None,
        system: str,
        prompt: str,
        json_schema: dict[str, Any],
        model: ModelConfig,
    ) -> AdapterResponse:
        if image_path is None:
            raise DataValidationError("http_detector cannot execute a no-image query")
        payload = {
            "model": model.id,
            "prompt": prompt,
            "image_base64": encode_image(image_path, model.image_max_side),
            "response_schema": json_schema,
        }
        response = self.client.post("/predict", json=payload)
        try:
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"Detector request failed: {exc}") from exc
        content = value.get("response", value)
        return AdapterResponse(
            content=json.dumps(content) if not isinstance(content, str) else content,
            input_tokens=0,
            output_tokens=0,
            provider_request_id=response.headers.get("x-request-id"),
            raw_finish_reason="stop",
        )


def make_adapter(model: ModelConfig, fixture_responses: dict[str, str] | None = None) -> Adapter:
    if model.adapter.kind == "fixture":
        return FixtureAdapter(fixture_responses or {})
    if model.adapter.kind == "http_detector":
        return HttpDetectorAdapter(model)
    return OpenAICompatibleAdapter(model)
