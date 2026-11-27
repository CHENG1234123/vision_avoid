from pathlib import Path

import numpy as np
import pytest
import yaml

from vision_avoid.config import load_config
from vision_avoid.geometry import CommandSelector, OccupancyGrid, build_occupancy, front_clearance
from vision_avoid.vehicle import measured, steering

ROOT = Path(__file__).resolve().parents[1]


def calibrated(tmp_path):
    car = yaml.safe_load((ROOT / "config/vehicle.example.yaml").read_text())
    car.update(
        width_m=0.39,
        front_overhang_m=0.065,
        camera_forward_m=0.28,
        camera_left_m=0.0,
        braking_distance_m=0.025,
        estop_verified=True,
        intrinsics_file="lens.yaml",
        zero_tilt_ack=True,
        precleared_forward_m=1.2,
    )
    (tmp_path / "lens.yaml").write_text(
        yaml.safe_dump(
            {
                "image_width": 640,
                "image_height": 480,
                "fx": 555.0,
                "fy": 555.0,
                "cx": 320.0,
                "cy": 240.0,
                "distortion": [0.0] * 5,
            }
        )
    )
    path = tmp_path / "car.yaml"
    path.write_text(yaml.safe_dump(car))
    return path


def test_template_and_powered_mode_reject_guesses(tmp_path):
    base = load_config(ROOT / "config/indoor.yaml")
    with pytest.raises(ValueError, match="braking_distance_m"):
        measured(ROOT / "config/vehicle.example.yaml", base)


def test_unknown_ground_and_short_edges_stop(tmp_path):
    car = measured(calibrated(tmp_path), load_config(ROOT / "config/indoor.yaml"))
    select = CommandSelector(car.planner).select
    empty = np.empty((0, 3), dtype=np.float32)
    unknown = build_occupancy(empty, car.camera, car.robot, car.planner)
    assert select(unknown, car.robot, "left", 0.0, 0.06).is_stop
    edge = OccupancyGrid(np.zeros((8, 6), dtype=np.uint8), 0.05, -0.2)
    assert select(edge, car.robot, "left", 0.1, 0.06).is_stop


def test_observed_floor_turns_with_a_forward_arc(tmp_path):
    car = measured(calibrated(tmp_path), load_config(ROOT / "config/indoor.yaml"))
    x, y = np.meshgrid(np.arange(0.025, 4.0, 0.05), np.arange(-1.575, 1.6, 0.05))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    grid = build_occupancy(points, car.camera, car.robot, car.planner)
    select = CommandSelector(car.planner).select
    straight = select(grid, car.robot, "right", 0.0, 0.06)
    assert straight.linear_x == 0.06 and straight.angular_z == 0.0
    grid.cells[27, 22] = 1  # Intrudes into the right edge of the straight corridor.
    turn = select(grid, car.robot, "left", 0.2, 0.06)
    assert turn.linear_x == 0.06
    assert 0 < turn.angular_z < 0.06 / 0.85


def test_real_vehicle_selection_never_requests_reverse_or_spin(tmp_path):
    car = measured(calibrated(tmp_path), load_config(ROOT / "config/indoor.yaml"))
    select = CommandSelector(car.planner).select
    x, y = np.meshgrid(np.arange(0.025, 4.0, 0.05), np.arange(-1.575, 1.6, 0.05))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    grid = build_occupancy(points, car.camera, car.robot, car.planner)
    grid.cells[27, 22] = 1
    for preferred in ("left", "right", "unknown"):
        command = select(grid, car.robot, preferred, 0.3, 0.06)
        if not command.is_stop:
            assert command.linear_x > 0.0
            assert abs(steering(command.linear_x, command.angular_z)) <= 18.0


def test_level_camera_requires_explicit_preclear(tmp_path):
    path = calibrated(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["zero_tilt_ack"] = False
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="zero_tilt_ack"):
        measured(path, load_config(ROOT / "config/indoor.yaml"))


def test_preclear_allows_unknown_only_inside_the_checked_corridor(tmp_path):
    car = measured(calibrated(tmp_path), load_config(ROOT / "config/indoor.yaml"))
    shape = (64, 80)
    cells = np.zeros(shape, dtype=np.uint8)
    res = 0.05
    origin = -1.6
    xs = (np.arange(shape[1]) + 1) * res
    ys = origin + (np.arange(shape[0]) + 0.5) * res
    corridor = (xs >= car.robot.vehicle.nose) & (
        xs <= car.robot.vehicle.nose + car.robot.vehicle.blind_ground_m
    )
    lane = np.abs(ys) <= car.robot.vehicle.width / 2.0 + car.robot.safety_margin_m
    cells[np.ix_(lane, corridor)] = 2
    grid = OccupancyGrid(cells, res, origin)
    command = CommandSelector(car.planner).select(grid, car.robot, "left", 0.0, 0.06)
    assert command.linear_x == 0.06
    cells[30, 12] = 1
    blocked = OccupancyGrid(cells, res, origin)
    assert CommandSelector(car.planner).select(
        blocked, car.robot, "left", 0.1, 0.06
    ).is_stop


def test_level_camera_straight_lane_uses_actual_swept_width(tmp_path):
    path = calibrated(tmp_path)
    data = yaml.safe_load(path.read_text())
    data.update(width_m=0.40, safety_margin_m=0.05)
    path.write_text(yaml.safe_dump(data))
    car = measured(path, load_config(ROOT / "config/indoor.yaml"))
    # Level camera: near floor is invisible. Beyond it, floor is visible in
    # the 50 cm corridor, with obstacles immediately outside the margin.
    cells = np.ones((64, 80), dtype=np.uint8)
    cells[27:37, :21] = 2
    cells[27:37, 21:] = 0
    grid = OccupancyGrid(cells, 0.05, -1.6)
    select = CommandSelector(car.planner).select
    cmd = select(grid, car.robot, "left", 0.0, 0.06)
    assert (cmd.linear_x, cmd.angular_z) == (0.06, 0.0)
    # A real obstacle at the edge of the pre-cleared strip still wins.
    cells[36, 12] = 1
    assert select(grid, car.robot, "left", 0.0, 0.06).is_stop
    cells[36, 12] = 2
    # Unknown across the lane beyond the declared blind zone still blocks.
    cells[27:37, 34] = 2
    assert select(grid, car.robot, "left", 0.0, 0.06).is_stop

    data.update(precleared_forward_m=5.0)
    path.write_text(yaml.safe_dump(data))
    long_preclear = measured(path, load_config(ROOT / "config/indoor.yaml"))
    cells[27:37, :] = 2
    assert select(grid, long_preclear.robot, "left", 0.0, 0.06).is_stop


@pytest.mark.parametrize("powered", [False, True])
def test_calibrated_blind_strip_ends_at_first_visible_cell(tmp_path, powered):
    path = calibrated(tmp_path)
    data = yaml.safe_load(path.read_text())
    data.update(camera_forward_m=0.335, front_overhang_m=0.0615)
    if not powered:
        del data["precleared_forward_m"]  # optional; powered case has the old 1.2 m
    path.write_text(yaml.safe_dump(data))
    lens = tmp_path / "lens.yaml"
    intr = yaml.safe_load(lens.read_text())
    intr.update(cy=228.758098, fy=514.498911)
    lens.write_text(yaml.safe_dump(intr))
    car = measured(path, load_config(ROOT / "config/indoor.yaml"), powered=powered)
    robot = car.robot
    assert robot.vehicle.blind_ground_m == pytest.approx(0.6681, abs=0.001)
    cells = np.zeros((64, 80), dtype=np.uint8)
    grid = OccupancyGrid(cells, 0.05, -1.6)
    select = CommandSelector(car.planner).select
    cells[:, :21] = 2  # ends at x=1.05, just before the first ground ray
    assert select(grid, robot, "center", 0, 0.06).linear_x == 0.06
    assert front_clearance(grid, robot) == pytest.approx(4 - robot.vehicle.nose)
    cells[:, 21] = 2  # first partially visible column; may not be skipped
    assert select(grid, robot, "center", 0, 0.06).is_stop
    assert front_clearance(grid, robot) == pytest.approx(1.05 - robot.vehicle.nose)
    cells[:, 21] = 0
    cells[32, 12] = 1  # tall obstacle within the ground blind strip
    assert select(grid, robot, "center", 0, 0.06).is_stop
    assert front_clearance(grid, robot) == pytest.approx(0.60 - robot.vehicle.nose)
