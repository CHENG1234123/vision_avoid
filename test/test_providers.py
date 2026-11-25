import json
import sys
import types

import pytest

from vision_avoid.config import VlmConfig
from vision_avoid.providers import (
    LMStudioSceneScanner,
    OpenAICompatibleSceneScanner,
    SceneScanError,
    create_scanner,
)


def vlm_config(
    *,
    provider="local",
    api_enabled=False,
) -> VlmConfig:
    return VlmConfig(
        provider=provider,
        local_timeout_s=1.0,
        scan_prior_s=2.0,
        api_enabled=api_enabled,
        api_model="qwen3-vl-flash",
        api_base_url="https://example.invalid/v1",
        api_key_env="TEST_VLM_KEY",
        api_timeout_s=1.0,
        structured_output_mode="json_schema",
        max_calls_per_hour=20,
        confidence_threshold=0.6,
        lmstudio_model="local/qwen3-vl-4b-test.gguf",
        lmstudio_url="http://127.0.0.1:1234/v1",
    )


VALID_SCAN = {
    "traversability": "clear",
    "safe_direction": "center",
    "summary": "Visible clear path",
    "hazards": [],
    "requires_operator": False,
    "confidence": 0.9,
}


def test_api_provider_requires_explicit_enable(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "secret")
    with pytest.raises(SceneScanError, match="not explicitly enabled"):
        OpenAICompatibleSceneScanner(vlm_config(provider="api"))


def test_api_provider_rejects_plaintext_remote_url(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "secret")
    config = vlm_config(provider="api", api_enabled=True)
    config = config.__class__(
        **{**config.__dict__, "api_base_url": "http://example.invalid/v1"}
    )
    with pytest.raises(SceneScanError, match="HTTPS"):
        OpenAICompatibleSceneScanner(config)


def test_api_provider_uses_json_schema_without_network(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "secret")
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content=json.dumps(VALID_SCAN))
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(
                completions=FakeCompletions()
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    scanner = OpenAICompatibleSceneScanner(
        vlm_config(provider="api", api_enabled=True)
    )
    result = scanner.scan(b"jpeg")
    assert result.traversability == "clear"
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["client"]["max_retries"] == 0
    assert "简体中文" in captured["messages"][0]["content"][0]["text"]
    assert "secret" not in json.dumps(captured["response_format"])


def test_api_provider_disables_thinking_by_default(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "secret")
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content=json.dumps(VALID_SCAN))
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    scanner = create_scanner(vlm_config(provider="api", api_enabled=True))
    scanner.scan(b"jpeg")
    assert captured["extra_body"] == {"enable_thinking": False}


def test_api_provider_json_object_mode_and_failure(monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "secret")
    captured = {}

    class FailingCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("simulated 429")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = types.SimpleNamespace(completions=FailingCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    config = vlm_config(provider="api", api_enabled=True)
    config = config.__class__(
        **{**config.__dict__, "structured_output_mode": "json_object"}
    )
    scanner = OpenAICompatibleSceneScanner(config)
    with pytest.raises(SceneScanError, match="API scene scan failed"):
        scanner.scan(b"jpeg")
    assert captured["response_format"] == {"type": "json_object"}


def test_lmstudio_provider_loads_scans_and_unloads(monkeypatch):
    config = vlm_config()
    scanner = LMStudioSceneScanner(config)
    requests = []

    def fake_request(method, path, payload=None):
        requests.append((method, path, payload))
        if path == "/api/v1/models/load":
            return {"status": "loaded", "instance_id": "instance-1"}
        if path == "/api/v1/models/unload":
            return {"instance_id": "instance-1"}
        return {"models": []}

    monkeypatch.setattr(scanner, "_request", fake_request)
    scanner.warmup()
    assert scanner._instance_id == "instance-1"
    scanner.unload()
    assert [path for _, path, _ in requests] == [
        "/api/v1/models",
        "/api/v1/models/load",
        "/api/v1/models",
        "/api/v1/models/unload",
        "/api/v1/models",
    ]


def test_lmstudio_is_selected_as_local_provider():
    scanner = create_scanner(vlm_config())
    assert isinstance(scanner, LMStudioSceneScanner)
