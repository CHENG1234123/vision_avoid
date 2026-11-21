from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from vision_avoid.config import SceneChangeConfig


class SceneEvent(str, Enum):
    INITIALIZED = "initialized"
    UNCHANGED = "unchanged"
    QUALITY_REJECTED = "quality_rejected"
    NEW_SCENE = "new_scene"


@dataclass(frozen=True)
class SceneObservation:
    event: SceneEvent
    hamming_distance: int
    brightness: float
    contrast: float
    sharpness: float


def perceptual_hash(frame_bgr: np.ndarray) -> np.ndarray:
    # shortcut: appearance hashing is not room recognition, upgrade for mapped navigation.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8].flatten()
    coefficients[0] = 0.0
    threshold = float(np.median(coefficients))
    return coefficients > threshold


class SceneChangeDetector:
    def __init__(self, config: SceneChangeConfig):
        self._config = config
        self._baseline: np.ndarray | None = None
        self._latest: np.ndarray | None = None
        self._changed_since: float | None = None
        self._cooldown_until = 0.0
        self._sampled_at = -1.0

    def update(
        self,
        frame_bgr: np.ndarray,
        now_s: float,
        force_sample: bool = False,
    ) -> SceneObservation:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        if not force_sample and now_s - self._sampled_at < self._config.sample_period_s:
            return self._observation(
                SceneEvent.UNCHANGED, brightness, contrast, sharpness
            )
        self._sampled_at = now_s

        if (
            brightness < self._config.min_brightness
            or contrast < self._config.min_contrast
            or sharpness < self._config.min_sharpness
        ):
            self._changed_since = None
            return self._observation(
                SceneEvent.QUALITY_REJECTED, brightness, contrast, sharpness
            )

        current = perceptual_hash(frame_bgr)
        self._latest = current
        if self._baseline is None:
            self._baseline = current
            return self._observation(
                SceneEvent.INITIALIZED, brightness, contrast, sharpness
            )

        distance = int(np.count_nonzero(current != self._baseline))
        if distance < self._config.hamming_threshold:
            self._changed_since = None
            return self._observation(
                SceneEvent.UNCHANGED, brightness, contrast, sharpness
            )

        if self._changed_since is None:
            self._changed_since = now_s
        changed_for = now_s - self._changed_since
        if changed_for >= self._config.hold_seconds and now_s >= self._cooldown_until:
            self.mark_scanned(now_s)
            return self._observation(
                SceneEvent.NEW_SCENE, brightness, contrast, sharpness
            )
        return self._observation(
            SceneEvent.UNCHANGED, brightness, contrast, sharpness
        )

    def mark_scanned(self, now_s: float) -> None:
        if self._latest is not None:
            self._baseline = self._latest.copy()
        self._changed_since = None
        self._cooldown_until = now_s + self._config.cooldown_seconds

    def _observation(
        self,
        event: SceneEvent,
        brightness: float,
        contrast: float,
        sharpness: float,
    ) -> SceneObservation:
        if self._baseline is None or self._latest is None:
            distance = 0
        else:
            distance = int(np.count_nonzero(self._latest != self._baseline))
        return SceneObservation(event, distance, brightness, contrast, sharpness)
