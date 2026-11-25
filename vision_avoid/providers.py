from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import replace
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


class OpenAICompatibleSceneScanner:
    name = "openai_compatible"

    def __init__(
        self,
        config: VlmConfig,
        api_key: str | None = None,
        extra_body: dict[str, object] | None = None,
    ):
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
        self._extra_body = dict(extra_body or {})
        self._api_key = api_key or os.environ.get(config.api_key_env, "")
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
            f"返回内容必须符合以下 JSON Schema：\n"
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

        request_kwargs: dict[str, object] = {}
        if self._extra_body:
            request_kwargs["extra_body"] = self._extra_body
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
                **request_kwargs,
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


class LMStudioSceneScanner:
    name = "lmstudio"

    def __init__(self, config: VlmConfig):
        parsed = urlparse(config.lmstudio_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise SceneScanError("LM Studio URL must use a loopback host")
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise SceneScanError("LM Studio URL must end with /v1")
        self._config = config
        self._root_url = config.lmstudio_url.rstrip("/")[:-3]
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._instance_id: str | None = None
        delegate_config = replace(
            config,
            provider="api",
            api_enabled=True,
            api_model=config.lmstudio_model,
            api_base_url=config.lmstudio_url,
            api_timeout_s=config.local_timeout_s,
        )
        self._delegate = OpenAICompatibleSceneScanner(
            delegate_config, api_key="lm-studio"
        )
        self.last_usage: dict[str, int] = {}

    def health_check(self) -> bool:
        try:
            return isinstance(self._get_models(), dict)
        except SceneScanError:
            return False

    def warmup(self) -> None:
        loaded_ids = self._loaded_instance_ids()
        if loaded_ids:
            self._instance_id = loaded_ids[0]
            return
        try:
            loaded = self._request(
                "POST",
                "/api/v1/models/load",
                {
                    "model": self._config.lmstudio_model,
                    "context_length": 4096,
                    "echo_load_config": True,
                },
            )
        except SceneScanError as exc:
            raise SceneScanError(
                f"LM Studio could not load {self._config.lmstudio_model}: {exc}"
            ) from exc
        self._instance_id = str(loaded.get("instance_id", ""))
        if loaded.get("status") != "loaded" or not self._instance_id:
            raise SceneScanError("LM Studio did not confirm the local VLM is loaded")

    def unload(self) -> None:
        instance_ids = set(self._loaded_instance_ids())
        if self._instance_id:
            instance_ids.add(self._instance_id)
        for instance_id in instance_ids:
            self._request(
                "POST",
                "/api/v1/models/unload",
                {"instance_id": instance_id},
            )
        self._instance_id = None
        for _ in range(30):
            if not self._loaded_instance_ids():
                return
            time.sleep(0.1)
        raise SceneScanError("LM Studio did not unload the local VLM")

    def scan(self, jpeg: bytes) -> SceneScan:
        result = self._delegate.scan(jpeg)
        self.last_usage = dict(self._delegate.last_usage)
        return result

    def _loaded_instance_ids(self) -> list[str]:
        loaded: list[str] = []
        for model in self._get_models().get("models", []):
            if model.get("key") != self._config.lmstudio_model:
                continue
            loaded.extend(
                str(item.get("id"))
                for item in model.get("loaded_instances", [])
                if item.get("id")
            )
        return loaded

    def _get_models(self) -> dict:
        return self._request("GET", "/api/v1/models")

    def _request(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._root_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with self._opener.open(
                request, timeout=self._config.local_timeout_s
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise SceneScanError(
                f"LM Studio request failed: {type(exc).__name__}"
            ) from exc
        if not isinstance(data, dict):
            raise SceneScanError("LM Studio returned a non-object response")
        return data


def create_scanner(config: VlmConfig) -> SceneScanner:
    if config.provider == "local":
        return LMStudioSceneScanner(config)
    if config.provider == "api":
        return OpenAICompatibleSceneScanner(
            config,
            extra_body={"enable_thinking": config.api_enable_thinking},
        )
    raise SceneScanError(f"unknown VLM provider: {config.provider}")
