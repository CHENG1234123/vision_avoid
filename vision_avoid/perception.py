from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision_avoid.config import AppConfig
from vision_avoid.geometry import OccupancyGrid, build_occupancy, depth_to_body_points


COCO_LABELS_ZH = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "motorbike": "摩托车",
    "airplane": "飞机",
    "aeroplane": "飞机",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "交通信号灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "单板滑雪板",
    "sports ball": "球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "sofa": "沙发",
    "potted plant": "盆栽",
    "pottedplant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "diningtable": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "tvmonitor": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "时钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷",
    "box": "箱子",
    "cable": "线缆",
    "cone": "锥桶",
}


def localize_detection_label(label: str) -> str:
    """Return a Chinese display label while keeping unknown classes generic."""
    normalized = " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())
    translated = COCO_LABELS_ZH.get(normalized)
    if translated is not None:
        return translated
    if any("\u4e00" <= character <= "\u9fff" for character in label):
        return label.strip()
    return "未知物体"


@lru_cache(maxsize=4)
def _cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
    )
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size=size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


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


def _draw_detections(
    frame: np.ndarray,
    detections: tuple[Detection, ...],
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    scale_x = render_width / source_width
    scale_y = render_height / source_height
    if not detections:
        return
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _cjk_font(16)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        x1 = int(x1 * scale_x) + offset_x
        x2 = int(x2 * scale_x) + offset_x
        y1 = int(y1 * scale_y) + offset_y
        y2 = int(y2 * scale_y) + offset_y
        label = localize_detection_label(detection.label)
        text = f"{label}  {detection.confidence:.2f}"
        text_left, text_top, text_right, text_bottom = draw.textbbox(
            (0, 0), text, font=font
        )
        text_width = text_right - text_left
        text_height = text_bottom - text_top
        label_top = max(0, y1 - text_height - 10)
        draw.rectangle(
            (x1, label_top, x1 + text_width + 10, label_top + text_height + 9),
            fill=(18, 92, 42),
        )
        draw.rectangle((x1, y1, x2, y2), outline=(45, 220, 95), width=2)
        draw.text(
            (x1 + 5, label_top + 4),
            text,
            font=font,
            fill=(245, 255, 245),
        )
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_cjk_text(
    frame: np.ndarray,
    position: tuple[int, int],
    text: str,
    size: int,
    color: tuple[int, int, int],
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text(
        position,
        text,
        font=_cjk_font(size),
        fill=(color[2], color[1], color[0]),
    )
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_camera_view(
    frame_bgr: np.ndarray,
    detections: tuple[Detection, ...],
) -> np.ndarray:
    width, height = 640, 360
    source_height, source_width = frame_bgr.shape[:2]
    scale = min(width / source_width, height / source_height)
    render_width = max(1, round(source_width * scale))
    render_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        frame_bgr,
        (render_width, render_height),
        interpolation=cv2.INTER_AREA,
    )
    frame = np.full((height, width, 3), (226, 230, 236), dtype=np.uint8)
    offset_x = (width - render_width) // 2
    offset_y = (height - render_height) // 2
    frame[offset_y : offset_y + render_height, offset_x : offset_x + render_width] = (
        resized
    )
    _draw_detections(
        frame,
        detections,
        source_width,
        source_height,
        render_width,
        render_height,
        offset_x,
        offset_y,
    )
    return frame


def render_depth_view(depth_m: np.ndarray) -> np.ndarray:
    return depth_colormap(depth_m, 480, 270)


def render_bev_view(occupancy: OccupancyGrid) -> np.ndarray:
    view = np.full((270, 480, 3), (246, 248, 250), dtype=np.uint8)
    cells = cv2.rotate(occupancy.cells * 255, cv2.ROTATE_90_CLOCKWISE)
    cells = cv2.resize(cells, (450, 240), interpolation=cv2.INTER_NEAREST)
    occupied = cells > 0
    view[15:255, 15:465][occupied] = (45, 45, 220)
    for distance_m in (1, 2, 3):
        y = 255 - int(distance_m * 60)
        cv2.line(view, (15, y), (465, y), (205, 210, 218), 1)
        cv2.putText(
            view,
            f"{distance_m}m",
            (467, min(266, y + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (95, 105, 120),
            1,
            cv2.LINE_AA,
        )
    robot_x = 240
    cv2.rectangle(view, (robot_x - 11, 244), (robot_x + 11, 255), (210, 105, 20), -1)
    cv2.arrowedLine(
        view,
        (robot_x, 244),
        (robot_x, 218),
        (210, 105, 20),
        2,
        tipLength=0.35,
    )
    return view


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
                "PyTorch/Transformers are unavailable "
                f"({type(exc).__name__}: {exc}); install requirements-cu128.txt"
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
                    label=localize_detection_label(id2label[int(label_id)]),
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
    _draw_detections(
        frame,
        detections,
        frame_bgr.shape[1],
        frame_bgr.shape[0],
        frame.shape[1],
        frame.shape[0],
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
    gate_text = "开启" if "gate=open" in state_text else "关闭"
    state_names_zh = {
        "INIT": "初始化",
        "RUNNING": "行驶中",
        "READY": "就绪",
        "SCANNING": "扫描中",
        "BLOCKED": "等待放行",
        "FAULT": "故障",
        "UNKNOWN": "未知",
    }
    state_display = state_names_zh.get(state_name, "未知")
    state_colors = {
        "INIT": (180, 185, 195),
        "RUNNING": (55, 205, 105),
        "READY": (55, 205, 105),
        "SCANNING": (40, 200, 240),
        "BLOCKED": (20, 120, 240),
        "FAULT": (50, 50, 230),
    }
    state_color = state_colors.get(state_name, (180, 185, 195))

    _draw_cjk_text(
        canvas,
        (14, 487),
        f"视觉避障  |  {state_display}  |  门禁{gate_text}",
        18,
        state_color,
    )
    _draw_cjk_text(
        canvas,
        (14, 514),
        (
            f"识别目标 {len(detections)} 个  "
            f"深度 {depth_m.min():.1f}-{depth_m.max():.1f} 米"
        ),
        11,
        (190, 195, 205),
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
        ("单目深度  近红 / 远蓝", 20),
        ("障碍俯视  前方向上", 193),
    ]
    for title, y in panel_titles:
        cv2.rectangle(
            canvas,
            (sidebar_x, y),
            (sidebar_x + sidebar_w, y + 20),
            (32, 36, 44),
            -1,
        )
        _draw_cjk_text(
            canvas,
            (sidebar_x + 8, y + 15),
            title,
            12,
            (225, 228, 235),
        )

    cv2.rectangle(canvas, (sidebar_x, 364), (949, 529), (22, 25, 31), -1)
    _draw_cjk_text(
        canvas,
        (sidebar_x + 8, 383),
        "状态",
        14,
        state_color,
    )

    detail_lines = lines[1:9]
    for index, line in enumerate(detail_lines):
        y = 404 + index * 15
        display = line if len(line) <= 43 else line[:40] + "..."
        _draw_cjk_text(
            canvas,
            (sidebar_x + 8, y),
            display,
            10,
            (205, 210, 220),
        )

    cv2.rectangle(canvas, (0, 0), (959, 539), state_color, 3)
    return canvas
