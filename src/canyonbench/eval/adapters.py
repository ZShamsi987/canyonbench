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
        image_path: Path,
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
        image_path: Path,
        system: str,
        prompt: str,
        json_schema: dict[str, Any],
        model: ModelConfig,
    ) -> AdapterResponse:
        image_b64 = encode_image(image_path, model.image_max_side)
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
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
        image_path: Path,
        system: str,
        prompt: str,
        json_schema: dict[str, Any],
        model: ModelConfig,
    ) -> AdapterResponse:
        key = f"{image_path.name}:{prompt}"
        content = self.responses.get(key, self.responses.get(prompt))
        if content is None:
            raise DataValidationError(f"No fixture response for {key}")
        return AdapterResponse(content, 0, 0, "fixture", "stop")


def make_adapter(model: ModelConfig, fixture_responses: dict[str, str] | None = None) -> Adapter:
    if model.adapter.kind == "fixture":
        return FixtureAdapter(fixture_responses or {})
    return OpenAICompatibleAdapter(model)
