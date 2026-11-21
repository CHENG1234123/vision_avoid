import math

import numpy as np

from vision_avoid.config import CameraConfig, PlannerConfig, RobotConfig
from vision_avoid.geometry import (
    CommandSelector,
    OccupancyGrid,
    apply_scan_constraints,
    build_occupancy,
    depth_to_body_points,
    front_clearance,
    path_is_clear,
    plan_command,
    rotation_command,
    side_clearance,
)
from vision_avoid.scene import Hazard, SceneScan


CAMERA = CameraConfig(
    width=5,
    height=5,
    fx=2.0,
    fy=2.0,
    cx=2.0,
    cy=2.0,
    mount_height_m=1.0,
    pitch_deg=0.0,
)

ROBOT = RobotConfig(
    half_width_m=0.1,
    safety_margin_m=0.0,
    max_linear_mps=0.25,
    caution_linear_mps=0.15,
    max_angular_rps=1.0,
    stop_distance_m=0.35,
    ground_tolerance_m=0.08,
    obstacle_max_height_m=1.2,
)

PLANNER = PlannerConfig(
    grid_length_m=4.0,
    grid_width_m=2.0,
    resolution_m=0.1,
    lookahead_m=1.8,
    time_step_s=0.1,
    turn_duration_s=1.0,
    yaw_rates=(0.0, 0.45, -0.45, 0.9, -0.9),
    hysteresis_s=0.5,
)


def empty_grid() -> OccupancyGrid:
    return OccupancyGrid(np.zeros((20, 40), dtype=np.uint8), 0.1, -1.0)


def test_center_pixel_projects_to_expected_body_point():
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    points = depth_to_body_points(depth, CAMERA, max_range_m=4.0, stride=1)
    center = points[
        (points[:, 0] == 2.0) & (points[:, 1] == 0.0) & (points[:, 2] == 1.0)
    ]
    assert len(center) == 1


def test_ground_points_are_filtered_from_occupancy():
    points = np.array(
        [[1.0, 0.0, 1.0], [2.0, 0.0, 0.02], [3.0, 0.0, 1.0]], dtype=np.float32
    )
    grid = build_occupancy(points, CAMERA, ROBOT, PLANNER)
    assert grid.cells.sum() > 0
    assert not grid.cells[10, 20]


def test_planner_turns_around_center_obstacle():
    grid = empty_grid()
    grid.cells[8:12, 14:18] = 1
    command = plan_command(grid, ROBOT, PLANNER)
    assert command.linear_x > 0
    assert abs(command.angular_z) > 0


def test_planner_stops_when_all_paths_are_blocked():
    grid = empty_grid()
    grid.cells[:, 4:] = 1
    command = plan_command(grid, ROBOT, PLANNER)
    assert command.is_stop


def test_stop_distance_halts_before_turning():
    grid = empty_grid()
    grid.cells[8:12, 3] = 1
    assert front_clearance(grid, ROBOT) <= ROBOT.stop_distance_m
    assert plan_command(grid, ROBOT, PLANNER).is_stop


def test_high_severity_vlm_hazard_blocks_center_path():
    scan = SceneScan(
        traversability="caution",
        safe_direction="left",
        summary="Transparent hazard ahead",
        hazards=[
            Hazard(
                label="glass",
                direction="center",
                distance_band="near",
                severity="high",
            )
        ],
        confidence=0.9,
    )
    fused = apply_scan_constraints(empty_grid(), scan, ROBOT)
    assert path_is_clear(empty_grid(), 0.0, ROBOT.caution_linear_mps, PLANNER)
    assert not path_is_clear(fused, 0.0, ROBOT.caution_linear_mps, PLANNER)


def test_selector_keeps_forward_when_clear():
    selector = CommandSelector(PLANNER)
    command = selector.select(empty_grid(), ROBOT, "left", 1.0, 0.25)
    assert command.linear_x == ROBOT.max_linear_mps
    assert command.angular_z == 0.0
    assert math.isfinite(command.linear_x)


def test_rotation_fallback_turns_toward_safe_side():
    grid = empty_grid()
    grid.cells[8:12, 5:] = 1
    command = rotation_command(grid, ROBOT, "left")
    assert command.linear_x == 0.0
    assert command.angular_z > 0.0
    assert side_clearance(grid, ROBOT, "left") >= 0.6
