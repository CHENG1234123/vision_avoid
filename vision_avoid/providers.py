from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Protocol
from urllib.parse import urlparse

from vision_avoid.config import VlmConfig
from vision_avoid.scene import SCENE_SCANNER_PROMPT, SceneScan, scene_schema


class SceneScanError(RuntimeError):
    pass


class SceneScanner(Protocol):
    name: str

    def health_check(self) -> bool: ...

    def warmup(self) -> None: ...

    def unload(self) -> None: ...

    def scan(self, jpeg: bytes) -> SceneScan: ...


class OllamaSceneScanner:
    name = "ollama"

    def __init__(self, config: VlmConfig):
        self._config = config
        self._url = config.local_url.rstrip("/")
        host = urlparse(self._url).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise SceneScanError("local Ollama URL must use a loopback host")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.last_usage: dict[str, int] = {}

    def health_check(self) -> bool:
        try:
            with self._opener.open(f"{self._url}/api/tags", timeout=3) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def warmup(self) -> None:
        payload = {
            "model": self._config.local_model,
            "prompt": "Reply with OK.",
            "stream": False,
            "keep_alive": -1,
            "options": {"num_ctx": 4096, "num_predict": 4},
        }
        self._post("/api/generate", payload)
        loaded = self._get("/api/ps").get("models", [])
        model = next(
            (
                item
                for item in loaded
                if self._config.local_model
                in {item.get("name"), item.get("model")}
            ),
            None,
        )
        if model is None:
            raise SceneScanError("Ollama did not keep the local VLM loaded")
        size = int(model.get("size", 0))
        size_vram = int(model.get("size_vram", 0))
        if size <= 0 or size_vram < int(size * 0.95):
            raise SceneScanError("local VLM is not fully resident in GPU memory")

    def unload(self) -> None:
        self._post(
            "/api/generate",
            {
                "model": self._config.local_model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        )
        loaded = self._get("/api/ps").get("models", [])
        if any(
            self._config.local_model in {item.get("name"), item.get("model")}
            for item in loaded
        ):
            raise SceneScanError("Ollama did not unload the local VLM")

    def scan(self, jpeg: bytes) -> SceneScan:
        image_b64 = base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": self._config.local_model,
            "messages": [
                {"role": "system", "content": SCENE_SCANNER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Inspect the visible floor and obstacles for a small ground robot. "
                        "Return all required JSON fields."
                    ),
                    "images": [image_b64],
                },
            ],
            "format": scene_schema(),
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 256,
            },
        }
        response = self._post("/api/chat", payload)
        self.last_usage = {
            "input_tokens": int(response.get("prompt_eval_count", 0)),
            "output_tokens": int(response.get("eval_count", 0)),
        }
        try:
            content = response["message"]["content"]
            return SceneScan.model_validate_json(content)
        except (KeyError, TypeError, ValueError) as exc:
            raise SceneScanError("Ollama returned an invalid scene scan") from exc

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(
                request, timeout=self._config.local_timeout_s
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise SceneScanError(f"Ollama request failed: {type(exc).__name__}") from exc
        if not isinstance(data, dict):
            raise SceneScanError("Ollama returned a non-object response")
        return data

    def _get(self, path: str) -> dict:
        try:
            with self._opener.open(
                f"{self._url}{path}", timeout=self._config.local_timeout_s
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise SceneScanError(f"Ollama request failed: {type(exc).__name__}") from exc
        if not isinstance(data, dict):
            raise SceneScanError("Ollama returned a non-object response")
        return data


class OpenAICompatibleSceneScanner:
    name = "openai_compatible"

    def __init__(self, config: VlmConfig):
        if not config.api_enabled:
            raise SceneScanError("API provider is not explicitly enabled")
        if not config.api_base_url:
            raise SceneScanError("vlm.api_base_url is required")
        parsed_url = urlparse(config.api_base_url)
        if parsed_url.scheme != "https" and parsed_url.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            raise SceneScanError("remote API base URL must use HTTPS")
        self._config = config
        self._api_key = os.environ.get(config.api_key_env, "")
        if not self._api_key:
            raise SceneScanError(
                f"required API key environment variable is not set: {config.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SceneScanError("the openai package is not installed") from exc
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=config.api_base_url,
            timeout=config.api_timeout_s,
            max_retries=0,
        )
        self.last_usage: dict[str, int] = {}

    def health_check(self) -> bool:
        return True

    def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def scan(self, jpeg: bytes) -> SceneScan:
        image_b64 = base64.b64encode(jpeg).decode("ascii")
        schema = scene_schema()
        prompt = (
            f"{SCENE_SCANNER_PROMPT}\n"
            f"Your response must satisfy this JSON Schema:\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )
        if self._config.structured_output_mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_scan",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(
                model=self._config.api_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                response_format=response_format,
                temperature=0,
                max_tokens=256,
            )
            usage = getattr(response, "usage", None)
            self.last_usage = {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0)),
                "output_tokens": int(getattr(usage, "completion_tokens", 0)),
            }
            content = response.choices[0].message.content
            return SceneScan.model_validate_json(content or "")
        except Exception as exc:
            raise SceneScanError(f"API scene scan failed: {type(exc).__name__}") from exc


def create_scanner(config: VlmConfig) -> SceneScanner:
    if config.provider == "local":
        return OllamaSceneScanner(config)
    if config.provider == "api":
        return OpenAICompatibleSceneScanner(config)
    raise SceneScanError(f"unknown VLM provider: {config.provider}")
