import cv2
import numpy as np

from vision_avoid.config import SceneChangeConfig
from vision_avoid.scene_change import SceneChangeDetector, SceneEvent


CONFIG = SceneChangeConfig(
    sample_period_s=0.0,
    hamming_threshold=8,
    hold_seconds=2.0,
    cooldown_seconds=45.0,
    min_brightness=5.0,
    min_contrast=5.0,
    min_sharpness=1.0,
)


def pattern(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    return cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)


def test_new_scene_requires_hold_time_and_cooldown():
    detector = SceneChangeDetector(CONFIG)
    assert detector.update(pattern(1), 0.0).event == SceneEvent.INITIALIZED
    assert detector.update(pattern(2), 1.0).event == SceneEvent.UNCHANGED
    assert detector.update(pattern(2), 3.1).event == SceneEvent.NEW_SCENE
    assert detector.update(pattern(3), 4.0).event == SceneEvent.UNCHANGED
    assert detector.update(pattern(3), 6.1).event == SceneEvent.UNCHANGED


def test_dark_frame_is_quality_rejected():
    detector = SceneChangeDetector(CONFIG)
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detector.update(dark, 0.0).event == SceneEvent.QUALITY_REJECTED
