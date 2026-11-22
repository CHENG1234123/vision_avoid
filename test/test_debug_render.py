import cv2
import numpy as np

from vision_avoid.geometry import OccupancyGrid
from vision_avoid.perception import (
    Detection,
    depth_colormap,
    localize_detection_label,
    render_bev_view,
    render_camera_view,
    render_debug,
    render_depth_view,
)


def test_debug_dashboard_has_fixed_layout():
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    depth = np.linspace(0.5, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    grid = OccupancyGrid(np.zeros((20, 40), dtype=np.uint8), 0.1, -1.0)
    grid.cells[8:12, 20:25] = 1
    output = render_debug(
        frame,
        depth,
        grid,
        (Detection("box", 0.9, (100, 120, 300, 360)),),
        [
            "state=RUNNING gate=open",
            "cmd=(0.15,0.00) scan=caution",
            "device=cuda",
            "fps=15.0 inference=65ms",
        ],
    )
    assert output.shape == (540, 960, 3)
    assert output.dtype == np.uint8
    assert output.mean() > 0
    assert cv2.imencode(".jpg", output)[0]


def test_depth_colormap_is_warm_near_and_cool_far():
    depth = np.array([[0.5, 5.0]], dtype=np.float32)
    colored = depth_colormap(depth, 2, 1)
    near_bgr = colored[0, 0]
    far_bgr = colored[0, 1]
    assert near_bgr[2] > near_bgr[0]
    assert far_bgr[0] > far_bgr[2]


def test_independent_debug_views_have_expected_sizes():
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    depth = np.linspace(0.5, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    grid = OccupancyGrid(np.zeros((20, 40), dtype=np.uint8), 0.1, -1.0)
    detections = (Detection("box", 0.9, (100, 120, 300, 360)),)
    assert render_camera_view(frame, detections).shape == (360, 640, 3)
    assert render_depth_view(depth).shape == (270, 480, 3)
    assert render_bev_view(grid).shape == (270, 480, 3)


def test_detection_labels_are_localized_to_chinese():
    assert localize_detection_label("person") == "人"
    assert localize_detection_label("traffic_light") == "交通信号灯"
    assert localize_detection_label("aeroplane") == "飞机"
    assert localize_detection_label("motorbike") == "摩托车"
    assert localize_detection_label("diningtable") == "餐桌"
    assert localize_detection_label("pottedplant") == "盆栽"
    assert localize_detection_label("sofa") == "沙发"
    assert localize_detection_label("tvmonitor") == "电视"
    assert localize_detection_label("box") == "箱子"
    assert localize_detection_label("未知") == "未知"
    assert localize_detection_label("not-a-coco-class") == "未知物体"
