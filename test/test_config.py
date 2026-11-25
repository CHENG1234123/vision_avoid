from pathlib import Path

from vision_avoid.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_indoor_config_loads():
    config = load_config(ROOT / "config" / "indoor.yaml")
    assert config.profile == "indoor"
    assert config.vlm.provider == "local"
    assert config.vlm.lmstudio_model
    assert config.robot.caution_linear_mps < config.robot.max_linear_mps


def test_outdoor_config_uses_outdoor_depth():
    config = load_config(ROOT / "config" / "outdoor.yaml")
    assert "outdoor" in config.depth_model


def test_environment_enables_api_explicitly(monkeypatch):
    monkeypatch.setenv("VLM_PROVIDER", "api")
    monkeypatch.setenv("VLM_API_ENABLED", "true")
    monkeypatch.setenv("VLM_API_BASE_URL", "https://example.invalid/v1")
    config = load_config(ROOT / "config" / "indoor.yaml")
    assert config.vlm.provider == "api"
    assert config.vlm.api_enabled


def test_models_directory_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_MODELS_DIR", str(tmp_path))
    config = load_config(ROOT / "config" / "indoor.yaml")
    assert config.models_root == tmp_path.resolve()


def test_lmstudio_url_can_be_configured(monkeypatch):
    monkeypatch.setenv("VLM_LMSTUDIO_URL", "http://127.0.0.1:1235/v1")
    config = load_config(ROOT / "config" / "indoor.yaml")
    assert config.vlm.lmstudio_url == "http://127.0.0.1:1235/v1"
