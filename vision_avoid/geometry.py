from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vision_avoid.config import CameraConfig, PlannerConfig, RobotConfig
from vision_avoid.scene import SceneScan


@dataclass(frozen=True)
class OccupancyGrid:
    cells: np.ndarray
    resolution_m: float
    origin_y_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.cells.shape


@dataclass(frozen=True)
class MotionCommand:
    linear_x: float = 0.0
    angular_z: float = 0.0

    @property
    def is_stop(self) -> bool:
        return self.linear_x == 0.0 and self.angular_z == 0.0


def depth_to_body_points(
    depth_m: np.ndarray,
    camera: CameraConfig,
    max_range_m: float,
    stride: int = 2,
) -> np.ndarray:
    """Project pixel depth into forward/left/up body coordinates."""
    if depth_m.ndim != 2:
        raise ValueError("depth image must be 2-D")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    pixels = depth_m[::stride, ::stride]
    rows, cols = pixels.shape
    if rows == 0 or cols == 0:
        return np.empty((0, 3), dtype=np.float32)

    scale_x = camera.width / float(depth_m.shape[1])
    scale_y = camera.height / float(depth_m.shape[0])
    v, u = np.mgrid[0:rows, 0:cols]
    u = u.astype(np.float32) * stride * scale_x
    v = v.astype(np.float32) * stride * scale_y
    z_cam = pixels.astype(np.float32, copy=False)
    x_cam = (u - camera.cx) * z_cam / camera.fx
    y_cam = (v - camera.cy) * z_cam / camera.fy

    pitch = math.radians(camera.pitch_deg)
    c, s = math.cos(pitch), math.sin(pitch)
    x_body = -y_cam * s + z_cam * c + camera.forward_m
    y_body = -x_cam + camera.left_m
    z_body = -y_cam * c - z_cam * s + camera.mount_height_m

    valid = (
        np.isfinite(z_cam)
        & (z_cam > 0.1)
        & (z_cam < max_range_m)
        & (x_body > 0.0)
    )
    return np.column_stack((x_body[valid], y_body[valid], z_body[valid]))


def build_occupancy(
    body_points: np.ndarray,
    camera: CameraConfig,
    robot: RobotConfig,
    planner: PlannerConfig,
) -> OccupancyGrid:
    nx = int(math.ceil(planner.grid_length_m / planner.resolution_m))
    ny = int(math.ceil(planner.grid_width_m / planner.resolution_m))
    cells = np.zeros((ny, nx), dtype=np.uint8)
    if robot.vehicle is not None:
        from vision_avoid.vehicle_path import measured_grid
        cells = measured_grid(body_points, robot, planner, (ny, nx))
        return OccupancyGrid(cells, planner.resolution_m, -planner.grid_width_m / 2)
    if len(body_points) == 0:
        return OccupancyGrid(cells, planner.resolution_m, -planner.grid_width_m / 2)

    x = body_points[:, 0]
    y = body_points[:, 1]
    z = body_points[:, 2]
    keep = (
        (z > robot.ground_tolerance_m)
        & (z < robot.obstacle_max_height_m)
        & (x >= 0.1)
        & (x < planner.grid_length_m)
        & (np.abs(y) < planner.grid_width_m / 2)
    )
    xi = (x[keep] / planner.resolution_m).astype(np.int32)
    yi = ((y[keep] + planner.grid_width_m / 2) / planner.resolution_m).astype(np.int32)
    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    cells[yi[valid], xi[valid]] = 1

    dilation_cells = int(
        math.ceil((robot.half_width_m + robot.safety_margin_m) / planner.resolution_m)
    )
    return OccupancyGrid(
        _dilate(cells, dilation_cells),
        planner.resolution_m,
        -planner.grid_width_m / 2,
    )


def _dilate(cells: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not cells.any():
        return cells
    result = np.zeros_like(cells)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            src_y = slice(max(0, -dy), min(cells.shape[0], cells.shape[0] - dy))
            dst_y = slice(max(0, dy), min(cells.shape[0], cells.shape[0] + dy))
            src_x = slice(max(0, -dx), min(cells.shape[1], cells.shape[1] - dx))
            dst_x = slice(max(0, dx), min(cells.shape[1], cells.shape[1] + dx))
            result[dst_y, dst_x] = np.maximum(
                result[dst_y, dst_x], cells[src_y, src_x]
            )
    return result


def _point_blocked(grid: OccupancyGrid, x: float, y: float) -> bool:
    xi = int(x / grid.resolution_m)
    yi = int((y - grid.origin_y_m) / grid.resolution_m)
    if xi < 0 or yi < 0 or yi >= grid.shape[0] or xi >= grid.shape[1]:
        return True
    return bool(grid.cells[yi, xi])


def path_is_clear(
    grid: OccupancyGrid,
    yaw_rate: float,
    linear_speed: float,
    planner: PlannerConfig,
) -> bool:
    # shortcut: fixed-speed kinematic lookahead, upgrade after wheel dynamics are measured.
    x = 0.0
    y = 0.0
    theta = 0.0
    travelled = 0.0
    elapsed = 0.0
    while travelled < planner.lookahead_m:
        travelled += linear_speed * planner.time_step_s
        x += linear_speed * math.cos(theta) * planner.time_step_s
        y += linear_speed * math.sin(theta) * planner.time_step_s
        elapsed += planner.time_step_s
        if elapsed <= planner.turn_duration_s:
            theta += yaw_rate * planner.time_step_s
        if _point_blocked(grid, x, y):
            return False
    return True


def front_clearance(grid: OccupancyGrid, robot: RobotConfig) -> float:
    if robot.vehicle is not None:
        car = robot.vehicle
        res = grid.resolution_m
        start = int(car.nose / res)
        y = grid.origin_y_m + np.arange(grid.shape[0]) * res
        half = robot.half_width_m + robot.safety_margin_m
        rows = (y < half - 1e-9) & (y + res > -half + 1e-9)
        band = grid.cells[rows, start:]
        blind = car.blind_columns(np.arange(start, grid.shape[1]), res)
        stops = np.flatnonzero(((band == 1) | ((band == 2) & ~blind)).any(axis=0))
        end_col = start + stops[0] if stops.size else grid.shape[1]
        return max(0.0, end_col * res - car.nose)
    y0 = int(math.floor(((-robot.half_width_m) - grid.origin_y_m) / grid.resolution_m))
    y1 = int(math.ceil(((robot.half_width_m) - grid.origin_y_m) / grid.resolution_m)) + 1
    nose = 0.0
    start = int(nose / grid.resolution_m)
    occupied_x = np.flatnonzero(grid.cells[max(0, y0) : y1, start:].any(axis=0))
    if occupied_x.size == 0:
        return max(0.0, grid.shape[1] * grid.resolution_m - nose)
    return max(0.0, (start + occupied_x[0]) * grid.resolution_m - nose)


def side_clearance(grid: OccupancyGrid, robot: RobotConfig, side: str) -> float:
    if side == "left":
        y_min = robot.half_width_m + grid.resolution_m
        y_max = y_min + 0.5
    elif side == "right":
        y_max = -robot.half_width_m - grid.resolution_m
        y_min = y_max - 0.5
    else:
        raise ValueError("side must be left or right")
    y0 = int(math.floor((y_min - grid.origin_y_m) / grid.resolution_m + 1e-9))
    y1 = int(math.ceil((y_max - grid.origin_y_m) / grid.resolution_m - 1e-9)) + 1
    occupied_x = np.flatnonzero(grid.cells[max(0, y0) : y1].any(axis=0))
    if occupied_x.size == 0:
        return grid.shape[1] * grid.resolution_m
    return float(occupied_x[0] * grid.resolution_m)


def apply_scan_constraints(
    grid: OccupancyGrid,
    scan: SceneScan,
    robot: RobotConfig,
) -> OccupancyGrid:
    cells = grid.cells.copy()
    nx = grid.shape[1]
    half_width = grid.shape[0] * grid.resolution_m / 2
    lateral = {
        "center": (
            -robot.half_width_m,
            robot.half_width_m,
        ),
        "left": (robot.half_width_m, half_width),
        "right": (-half_width, -robot.half_width_m),
        "unknown": (-half_width, half_width),
    }
    distance = {
        "near": (robot.stop_distance_m + grid.resolution_m, 1.2),
        "medium": (1.2, 2.4),
        "far": (2.4, grid.shape[1] * grid.resolution_m),
        "unknown": (0.0, grid.shape[1] * grid.resolution_m),
    }
    for hazard in scan.hazards:
        if hazard.severity == "low":
            continue
        y_min, y_max = lateral[hazard.direction]
        x_min, x_max = distance[hazard.distance_band]
        y0 = int((y_min - grid.origin_y_m) / grid.resolution_m)
        y1 = int((y_max - grid.origin_y_m) / grid.resolution_m) + 1
        x0 = int(x_min / grid.resolution_m)
        x1 = min(nx, int(x_max / grid.resolution_m) + 1)
        cells[max(0, y0) : min(grid.shape[0], y1), max(0, x0) : x1] = 1
    return OccupancyGrid(cells, grid.resolution_m, grid.origin_y_m)


def rotation_command(
    grid: OccupancyGrid,
    robot: RobotConfig,
    preferred_direction: str,
) -> MotionCommand:
    if preferred_direction not in {"left", "right"}:
        return MotionCommand()
    if side_clearance(grid, robot, preferred_direction) < 0.6:
        return MotionCommand()
    if front_clearance(grid, robot) <= robot.stop_distance_m:
        return MotionCommand()
    direction = 1.0 if preferred_direction == "left" else -1.0
    return MotionCommand(0.0, direction * robot.max_angular_rps * 0.6)


def plan_command(
    grid: OccupancyGrid,
    robot: RobotConfig,
    planner: PlannerConfig,
    preferred_direction: str = "unknown",
) -> MotionCommand:
    if front_clearance(grid, robot) <= robot.stop_distance_m:
        return MotionCommand()
    rates = list(planner.yaw_rates)

    def order(rate: float) -> tuple[int, int, float]:
        prefers_left = preferred_direction == "left" and rate > 0
        prefers_right = preferred_direction == "right" and rate < 0
        return (0 if rate == 0.0 else 1, 0 if prefers_left or prefers_right else 1, abs(rate))

    for rate in sorted(rates, key=order):
        if path_is_clear(grid, rate, robot.max_linear_mps, planner):
            return MotionCommand(robot.max_linear_mps, rate)
    return MotionCommand()


class CommandSelector:
    """Prefer a short-held turn while still trying straight travel first."""

    def __init__(self, planner: PlannerConfig):
        self._planner = planner
        self._last_turn = 0.0
        self._hold_until = 0.0

    def select(
        self,
        grid: OccupancyGrid,
        robot: RobotConfig,
        preferred_direction: str,
        now_s: float,
        speed_cap: float,
    ) -> MotionCommand:
        if robot.vehicle is not None:
            from vision_avoid.vehicle_path import choose
            return choose(grid, robot, self._planner, preferred_direction, speed_cap)
        if front_clearance(grid, robot) <= robot.stop_distance_m:
            return MotionCommand()
        rates = list(self._planner.yaw_rates)
        if now_s < self._hold_until and self._last_turn != 0.0:
            turn = self._last_turn
            rates = [0.0, turn] + [rate for rate in rates if rate not in {0.0, turn}]
        else:
            rates = [0.0] + sorted(
                [rate for rate in rates if rate != 0.0],
                key=lambda rate: (
                    0
                    if (
                        (preferred_direction == "left" and rate > 0)
                        or (preferred_direction == "right" and rate < 0)
                    )
                    else 1,
                    abs(rate),
                ),
            )

        for rate in rates:
            if path_is_clear(grid, rate, speed_cap, self._planner):
                command = MotionCommand(speed_cap, rate)
                if rate != 0.0:
                    self._last_turn = rate
                    self._hold_until = now_s + self._planner.hysteresis_s
                return command
        return rotation_command(grid, robot, preferred_direction)
