"""Rear-axle trajectories on a measured ground/obstacle/unknown grid."""
from functools import lru_cache
import math

import numpy as np

from vision_avoid.vehicle import WHEELBASE


def measured_grid(points, robot, planner, shape):
    res = planner.resolution_m
    rows, cols = shape
    cells = np.full(shape, 2, dtype=np.uint8)  # unknown unless we actually saw floor
    if len(points):
        x, y, z = points.T
        ix = np.floor(x / res).astype(int)
        iy = np.floor(y / res + rows / 2).astype(int)
        valid = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        floor = valid & (np.abs(z) <= robot.ground_tolerance_m)
        cells[iy[floor], ix[floor]] = 0
        obstacle = valid & (z > robot.ground_tolerance_m) & (z < robot.obstacle_max_height_m)
        cells[iy[obstacle], ix[obstacle]] = 1
    return cells


@lru_cache(maxsize=128)
def _straight_sweep(shape, res, origin_y, nose, width, margin, distance):
    # Straight translation sweeps one axis-aligned rectangle exactly. Count
    # cells with positive area inside it; merely sharing an edge is fine.
    half = width / 2 + margin
    end = nose + distance + margin
    eps = res * 1e-9
    if (end > shape[1] * res + eps or -half < origin_y - eps
            or half > origin_y + shape[0] * res + eps):
        return None
    x = np.arange(shape[1]) * res
    y = origin_y + np.arange(shape[0]) * res
    rows = (y < half - eps) & (y + res > -half + eps)
    collision = rows[:, None] & (x < end - eps)
    observed = collision & (x + res > nose + eps)
    return np.flatnonzero(collision), np.flatnonzero(observed)


@lru_cache(maxsize=128)
def sweep(shape, res, origin_y, nose, tail, width, margin, distance, steer):
    if steer == 0:
        return _straight_sweep(shape, res, origin_y, nose, width, margin, distance)
    # Sample constant curvature; include cell half diagonal + sampling motion in the envelope.
    k = math.tan(math.radians(steer)) / WHEELBASE
    step = res / 3
    pad = res / math.sqrt(2) + step * (1 + abs(k) * (nose + width))
    back = math.ceil((tail + margin + pad + width) / res)
    ix, iy = np.meshgrid(np.arange(-back, shape[1]), np.arange(shape[0]))
    x, y = (ix + 0.5) * res, origin_y + (iy + 0.5) * res
    obstacle = np.zeros(ix.shape, bool)
    known = np.zeros(ix.shape, bool)
    for s in np.linspace(0, distance, math.ceil(distance / step) + 1):
        theta = s * k
        ct, st = math.cos(theta), math.sin(theta)
        px = st / k if k else s
        py = (1 - ct) / k if k else 0.0
        along, cross = (x - px) * ct + (y - py) * st, -(x - px) * st + (y - py) * ct
        # A trajectory outside the mapped front/sides can't be approved.
        for end in (-tail-margin-pad, nose+margin+pad):
            for side in (-width/2-margin-pad, width/2+margin+pad):
                cx, cy = px + ct*end - st*side, py + st*end + ct*side
                if cx >= shape[1]*res or not origin_y <= cy < origin_y + shape[0]*res:
                    return None
        swept = ((along >= -tail-margin-pad) & (along <= nose+margin+pad)
                 & (np.abs(cross) <= width/2+margin+pad))
        obstacle |= swept
        actual = ((along >= -tail-pad) & (along <= nose+pad)
                  & (np.abs(cross) <= width/2+pad))
        known |= np.where(ix < 0, actual, swept)
    # Forward camera cannot see under the car. Only its initial, operator-cleared
    # envelope may be unknown. New front/side space must be observed ground.
    # At the rear, test the actual swept body against the pre-cleared side buffer.
    initial = ((x >= -tail-margin-pad) & (x <= nose)
               & (np.abs(y) <= width/2+margin+pad))
    known &= ~initial
    if np.any(known & (ix < 0)):
        return None
    return (np.flatnonzero(obstacle[:, back:]), np.flatnonzero(known[:, back:]))


def _blind_unknown_allowed(grid, robot):
    car = robot.vehicle
    if car.blind_ground_m <= 0.0:
        return None
    res = grid.resolution_m
    ys = grid.origin_y_m + (np.arange(grid.shape[0], dtype=float) + 0.5) * res
    x_ok = car.blind_columns(np.arange(grid.shape[1]), res)
    y_ok = np.abs(ys) + res / 2 <= car.width / 2.0 + robot.safety_margin_m + 1e-9
    return (y_ok[:, None] & x_ok[None, :]).ravel()


def _straight_blocked(grid, robot, masks, *, require_visible_end):
    collision, observed = masks
    cells = grid.cells.ravel()
    if (cells[collision] == 1).any() or not observed.size:
        return True
    cols = observed % grid.shape[1]
    # The part of a nose-straddling cell before the nose is under the
    # initial envelope. The part ahead may use the checked straight strip.
    car = robot.vehicle
    need_floor = ~car.blind_columns(cols, grid.resolution_m)
    if require_visible_end:
        # Even if the horizon falls in the blind zone, require visible floor.
        need_floor |= cols == cols.max()
    return bool((cells[observed[need_floor]] == 2).any())


def choose(grid, robot, planner, preferred, speed):
    from vision_avoid.geometry import MotionCommand

    car = robot.vehicle
    if not math.isfinite(speed) or not 0 < speed <= robot.max_linear_mps:
        return MotionCommand()
    sign = -1 if preferred == "right" else 1
    cells = grid.cells.ravel()
    allowed_unknown = _blind_unknown_allowed(grid, robot)
    stop = sweep(grid.shape, grid.resolution_m, grid.origin_y_m, car.nose, car.tail,
                 car.width, robot.safety_margin_m, robot.stop_distance_m, 0)
    if stop is None or _straight_blocked(grid, robot, stop, require_visible_end=False):
        return MotionCommand()
    for angle in (0, 8*sign, -8*sign, 16*sign, -16*sign):
        masks = sweep(grid.shape, grid.resolution_m, grid.origin_y_m, car.nose, car.tail,
                      car.width, robot.safety_margin_m, planner.lookahead_m, angle)
        if masks is None:
            continue
        if angle == 0:
            if _straight_blocked(grid, robot, masks, require_visible_end=True):
                continue
        else:
            collision, observed = masks
            unknown = cells[observed] == 2
            if allowed_unknown is not None:
                unknown &= ~allowed_unknown[observed]
            if (cells[collision] == 1).any() or unknown.any():
                continue
        return MotionCommand(speed, speed * math.tan(math.radians(angle)) / WHEELBASE)
    return MotionCommand()
