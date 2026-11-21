import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_avoid.config import load_config
from vision_avoid.node import encode_snapshot
from vision_avoid.perception import LocalPerception
from vision_avoid.providers import OllamaSceneScanner


ROOT = Path(__file__).resolve().parents[1]


def synthetic_robot_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for row in range(240, 480):
        value = int(70 + (row - 240) * 0.45)
        frame[row, :] = (value, value, value)
    cv2.rectangle(frame, (210, 130), (430, 350), (20, 30, 210), -1)
    for x in range(0, 640, 80):
        cv2.line(frame, (x, 240), (x + 100, 479), (110, 110, 110), 2)
    return frame


def test_snapshot_preserves_camera_aspect_ratio():
    encoded = np.frombuffer(encode_snapshot(synthetic_robot_frame()), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert image.shape[:2] == (768, 1024)


def test_real_local_models_smoke():
    pytest.importorskip("torch")
    config = load_config(ROOT / "config" / "indoor.yaml")
    if not (config.models_root / config.detection_model).exists():
        pytest.skip("local model weights have not been downloaded")

    perception = LocalPerception(config)
    output = perception.infer(synthetic_robot_frame())
    assert perception.device in {"cuda", "cpu"}
    assert output.depth_m.shape == (config.camera.height, config.camera.width)
    assert np.isfinite(output.depth_m).all()
    assert output.occupancy.cells.shape == (20, 40)
    assert output.depth_m.min() >= 0.1


def test_local_qwen_smoke_when_available():
    config = load_config(ROOT / "config" / "indoor.yaml")
    scanner = OllamaSceneScanner(config.vlm)
    if not scanner.health_check():
        pytest.skip("Ollama is not running")
    try:
        models = scanner._get("/api/tags").get("models", [])
    except Exception:
        pytest.skip("Ollama model list is unavailable")
    if not any(config.vlm.local_model in {item.get("name"), item.get("model")} for item in models):
        pytest.skip("local Qwen model has not been pulled")
    models_root = Path(
        os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models")
    )
    manifest_path = (
        models_root
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "qwen3-vl"
        / "4b-instruct"
    )
    if not manifest_path.exists():
        pytest.skip("local Qwen manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    large_layers = [
        layer
        for layer in manifest["layers"]
        if layer["mediaType"]
        in {
            "application/vnd.ollama.image.model",
            "application/vnd.ollama.image.projector",
        }
    ]
    for layer in large_layers:
        blob = models_root / "blobs" / layer["digest"].replace(":", "-")
        if not blob.exists() or blob.stat().st_size != layer["size"]:
            pytest.skip("local Qwen blobs are still downloading")

    ok, jpeg = cv2.imencode(".jpg", synthetic_robot_frame())
    assert ok
    scanner.warmup()
    result = scanner.scan(jpeg.tobytes())
    assert result.summary
    assert 0.0 <= result.confidence <= 1.0
