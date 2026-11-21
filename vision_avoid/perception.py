from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_avoid.config import AppConfig
from vision_avoid.geometry import OccupancyGrid, build_occupancy, depth_to_body_points


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class PerceptionOutput:
    detections: tuple[Detection, ...]
    depth_m: np.ndarray
    occupancy: OccupancyGrid


def depth_colormap(depth_m: np.ndarray, width: int, height: int) -> np.ndarray:
    normalized = cv2.normalize(
        depth_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    near_is_warm = 255 - normalized
    colored = cv2.applyColorMap(near_is_warm, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST)


class LocalPerception:
    """Loads the two real-time models from a local model directory."""

    def __init__(self, config: AppConfig):
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
                DFineForObjectDetection,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch/Transformers are missing; install requirements-cu128.txt"
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        detection_path = (
            config.models_root / config.detection_model
        ).resolve()
        depth_path = (config.models_root / config.depth_model).resolve()
        for path in (detection_path, depth_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"model directory not found: {path}; run vision-download-models"
                )

        self._depth_processor = AutoImageProcessor.from_pretrained(
            str(depth_path), local_files_only=True, use_fast=False
        )
        self._depth_model = (
            AutoModelForDepthEstimation.from_pretrained(
                str(depth_path), local_files_only=True
            )
            .to(self._device)
            .eval()
        )
        self._detection_processor = AutoImageProcessor.from_pretrained(
            str(detection_path), local_files_only=True, use_fast=False
        )
        self._detection_model = (
            DFineForObjectDetection.from_pretrained(
                str(detection_path), local_files_only=True
            )
            .to(self._device)
            .eval()
        )
        self._config = config

    @property
    def device(self) -> str:
        return self._device

    def infer(self, frame_bgr: np.ndarray) -> PerceptionOutput:
        from PIL import Image

        resized = cv2.resize(
            frame_bgr,
            (self._config.camera.width, self._config.camera.height),
            interpolation=cv2.INTER_AREA,
        )
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        with self._torch.inference_mode():
            depth_inputs = self._depth_processor(
                images=image, return_tensors="pt"
            ).to(self._device)
            predicted_depth = self._depth_model(**depth_inputs).predicted_depth
            depth_batch = (
                predicted_depth.unsqueeze(0).unsqueeze(0)
                if predicted_depth.ndim == 2
                else predicted_depth.unsqueeze(1)
            )
            depth = self._torch.nn.functional.interpolate(
                depth_batch,
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            depth_m = depth.detach().float().cpu().numpy()
            depth_m = np.clip(depth_m, 0.1, self._config.planner.grid_length_m * 2)

            detection_inputs = self._detection_processor(
                images=image, return_tensors="pt"
            ).to(self._device)
            outputs = self._detection_model(**detection_inputs)
            results = self._detection_processor.post_process_object_detection(
                outputs,
                target_sizes=self._torch.tensor(
                    [image.size[::-1]]
                ),
                threshold=0.3,
            )[0]

        detections: list[Detection] = []
        id2label = self._detection_model.config.id2label
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box_values = box.detach().cpu().numpy().astype(int)
            detections.append(
                Detection(
                    label=id2label[int(label_id)],
                    confidence=float(score),
                    box=tuple(int(v) for v in box_values),
                )
            )

        points = depth_to_body_points(
            depth_m,
            self._config.camera,
            max_range_m=self._config.planner.grid_length_m * 2,
            stride=2,
        )
        occupancy = build_occupancy(
            points, self._config.camera, self._config.robot, self._config.planner
        )
        return PerceptionOutput(tuple(detections), depth_m, occupancy)


def render_debug(
    frame_bgr: np.ndarray,
    depth_m: np.ndarray,
    occupancy: OccupancyGrid,
    detections: tuple[Detection, ...],
    lines: list[str],
) -> np.ndarray:
    canvas = np.full((540, 960, 3), (28, 31, 36), dtype=np.uint8)
    frame = cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_AREA)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        scale_x = 640.0 / frame_bgr.shape[1]
        scale_y = 480.0 / frame_bgr.shape[0]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        color = (45, 220, 95)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{detection.label}  {detection.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_top = max(0, y1 - text_h - 8)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_w + 8, label_top + text_h + 7),
            (18, 92, 42),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 4, label_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 255, 245),
            1,
            cv2.LINE_AA,
        )

    canvas[0:480, 0:640] = frame
    cv2.rectangle(canvas, (0, 480), (959, 539), (18, 21, 26), -1)
    cv2.line(canvas, (640, 0), (640, 539), (65, 70, 80), 1)

    state_text = lines[0] if lines else "state=UNKNOWN"
    state_name = (
        state_text.split("state=", 1)[1].split()[0]
        if "state=" in state_text
        else "UNKNOWN"
    )
    gate_text = "OPEN" if "gate=open" in state_text else "CLOSED"
    state_colors = {
        "RUNNING": (55, 205, 105),
        "READY": (55, 205, 105),
        "SCANNING": (40, 200, 240),
        "BLOCKED": (20, 120, 240),
        "FAULT": (50, 50, 230),
    }
    state_color = state_colors.get(state_name, (180, 185, 195))

    cv2.putText(
        canvas,
        f"VISION AVOID  |  {state_name}  |  GATE {gate_text}",
        (14, 514),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        state_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"detections={len(detections)}  depth={depth_m.min():.1f}-{depth_m.max():.1f}m",
        (14, 532),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 195, 205),
        1,
        cv2.LINE_AA,
    )

    sidebar_x, sidebar_w = 650, 300
    canvas[38:173, sidebar_x : sidebar_x + sidebar_w] = depth_colormap(
        depth_m, sidebar_w, 135
    )

    occ_display = cv2.rotate(occupancy.cells * 255, cv2.ROTATE_90_CLOCKWISE)
    occ_display = cv2.resize(
        occ_display,
        (sidebar_w, 135),
        interpolation=cv2.INTER_NEAREST,
    )
    occ_color = cv2.applyColorMap(occ_display.astype(np.uint8), cv2.COLORMAP_INFERNO)
    canvas[211:346, sidebar_x : sidebar_x + sidebar_w] = occ_color
    robot_x = sidebar_x + sidebar_w // 2
    cv2.rectangle(canvas, (robot_x - 11, 333), (robot_x + 11, 344), (0, 220, 255), -1)
    cv2.arrowedLine(
        canvas,
        (robot_x, 333),
        (robot_x, 316),
        (0, 220, 255),
        2,
        tipLength=0.35,
    )

    panel_titles = [
        ("DEPTH  NEAR=RED / FAR=BLUE", 20),
        ("OBSTACLES  FORWARD=UP", 193),
    ]
    for title, y in panel_titles:
        cv2.rectangle(
            canvas,
            (sidebar_x, y),
            (sidebar_x + sidebar_w, y + 20),
            (32, 36, 44),
            -1,
        )
        cv2.putText(
            canvas,
            title,
            (sidebar_x + 8, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (225, 228, 235),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (sidebar_x, 364), (949, 529), (22, 25, 31), -1)
    cv2.putText(
        canvas,
        "STATUS",
        (sidebar_x + 8, 383),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        state_color,
        1,
        cv2.LINE_AA,
    )

    detail_lines = lines[1:9]
    for index, line in enumerate(detail_lines):
        y = 404 + index * 15
        display = line if len(line) <= 43 else line[:40] + "..."
        cv2.putText(
            canvas,
            display,
            (sidebar_x + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (205, 210, 220),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (0, 0), (959, 539), state_color, 3)
    return canvas
