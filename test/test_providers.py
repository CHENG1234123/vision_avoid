import json
import sys
import types

import pytest

from vision_avoid.config import VlmConfig
from vision_avoid.providers import (
    OllamaSceneScanner,
    OpenAICompatibleSceneScanner,
    SceneScanError,
    create_scanner,
)


def vlm_config(*, provider="local", api_enabled=False) -> VlmConfig:
    return VlmConfig(
        provider=provider,
        local_model="qwen3-vl:4b-instruct",
        local_url="http://127.0.0.1:11434",
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
    )


VALID_SCAN = {
    "traversability": "clear",
    "safe_direction": "center",
    "summary": "Visible clear path",
    "hazards": [],
    "requires_operator": False,
    "confidence": 0.9,
}


def test_ollama_provider_validates_structured_response(monkeypatch):
    scanner = OllamaSceneScanner(vlm_config())
    captured = {}

    def fake_post(path, payload):
        del path
        captured.update(payload)
        return {"message": {"content": json.dumps(VALID_SCAN)}}

    monkeypatch.setattr(
        scanner,
        "_post",
        fake_post,
    )
    result = scanner.scan(b"jpeg")
    assert result.allows_motion(0.6)
    assert "简体中文" in captured["messages"][0]["content"]
    assert "简体中文" in captured["messages"][1]["content"]


def test_local_provider_rejects_non_loopback_url():
    config = vlm_config()
    config = config.__class__(
        **{**config.__dict__, "local_url": "http://203.0.113.10:11434"}
    )
    with pytest.raises(SceneScanError, match="loopback"):
        OllamaSceneScanner(config)


def test_ollama_warmup_requires_model_fully_on_gpu(monkeypatch):
    scanner = OllamaSceneScanner(vlm_config())
    monkeypatch.setattr(scanner, "_post", lambda path, payload: {})
    monkeypatch.setattr(
        scanner,
        "_get",
        lambda path: {
            "models": [
                {
                    "name": "qwen3-vl:4b-instruct",
                    "size": 4_000,
                    "size_vram": 4_000,
                }
            ]
        },
    )
    scanner.warmup()

    monkeypatch.setattr(
        scanner,
        "_get",
        lambda path: {
            "models": [
                {
                    "name": "qwen3-vl:4b-instruct",
                    "size": 4_000,
                    "size_vram": 2_000,
                }
            ]
        },
    )
    with pytest.raises(SceneScanError, match="not fully resident"):
        scanner.warmup()


def test_ollama_unload_requires_model_to_leave_gpu(monkeypatch):
    scanner = OllamaSceneScanner(vlm_config())
    monkeypatch.setattr(scanner, "_post", lambda path, payload: {})
    monkeypatch.setattr(scanner, "_get", lambda path: {"models": []})
    scanner.unload()

    monkeypatch.setattr(
        scanner,
        "_get",
        lambda path: {
            "models": [{"name": "qwen3-vl:4b-instruct", "size_vram": 4_000}]
        },
    )
    with pytest.raises(SceneScanError, match="did not unload"):
        scanner.unload()


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


def test_local_provider_does_not_import_api_client(monkeypatch):
    def fail_import(name):
        if name == "openai":
            raise AssertionError("local provider imported openai")
        return original_import(name)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)
    scanner = create_scanner(vlm_config())
    assert isinstance(scanner, OllamaSceneScanner)
