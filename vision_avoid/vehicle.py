"""Measured vehicle geometry and conservative forward-only motion limits."""
from dataclasses import dataclass, replace
import math
from pathlib import Path

import yaml


WHEELBASE = 0.325
TRACK = 0.322
WHEEL_DIAMETER = 0.123
MAX_STEER = 18.0
MAX_SPEED = 0.20
CAUTION_SPEED = 0.12


@dataclass(frozen=True)
class Vehicle:
    width: float
    nose: float  # rear axle to front bumper
    tail: float
    brake: float
    distortion: tuple
    blind_ground_m: float = 0.0  # forward of nose, for missing ground only

    def blind_columns(self, columns, resolution):
        # Require observation in a cell that straddles the first visible ground.
        far_edge = (columns + 1) * resolution
        return (far_edge > self.nose) & (far_edge <= self.nose + self.blind_ground_m + 1e-9)


def measured(path, base, *, powered=True):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("vehicle file must be a mapping")

    def number(key, low, high, obj=data):
        v = obj.get(key)
        if type(v) not in (float, int) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"vehicle calibration: measure {key} in [{low}, {high}]")
        return float(v)

    width = number("width_m", TRACK, 1.0)
    margin = (
        number("safety_margin_m", 0.0, 0.3)
        if data.get("safety_margin_m") is not None
        else 0.10
    )
    nose = WHEELBASE + number("front_overhang_m", 0, 0.6)
    tail = number("rear_overhang_m", 0, 0.6)
    brake = (number("braking_distance_m", 0.001, 1.0)
             if powered or data.get("braking_distance_m") is not None else 0.35)
    # Older vehicle files used 0.10 m/s. Reserve more braking room when
    # raising that speed, without ever shortening the configured distance.
    brake_reference = (number("braking_reference_mps", 0.01, 1.0)
                       if "braking_reference_mps" in data else 0.10)
    brake_allowance = brake * max(1.0, (MAX_SPEED / brake_reference) ** 2)
    if powered and data.get("estop_verified") is not True:
        raise ValueError("vehicle calibration: independent motor stop has not been verified")
    if not isinstance(data.get("intrinsics_file"), str) or not data["intrinsics_file"].strip():
        raise ValueError("vehicle calibration: intrinsics_file is required")
    intr = yaml.safe_load((path.parent / data["intrinsics_file"]).read_text())
    if (intr.get("image_width"), intr.get("image_height")) != (640, 480):
        raise ValueError("vehicle calibration requires intrinsics measured at 640 x 480")
    dist = intr.get("distortion")
    if not isinstance(dist, list) or len(dist) not in (4, 5, 8, 12, 14) or not all(
        type(x) in (int, float) and math.isfinite(x) for x in dist
    ):
        raise ValueError("vehicle calibration: invalid lens distortion")
    pitch = number("pitch_deg", -15, 80)
    forward = number("camera_forward_m", -tail, nose)
    mount = number("mount_height_m", .02, 2)
    cy = number("cy", 0, 479, intr)
    fy = number("fy", 1, 5000, intr)
    blind = 0.0
    if abs(pitch) <= 1.0:
        acknowledged = data.get("zero_tilt_ack") is True
        if powered and not acknowledged:
            raise ValueError("vehicle calibration: zero_tilt_ack is required for a level camera")
        if acknowledged:
            down = math.radians(pitch) + math.atan((intr["image_height"] - 1 - cy) / fy)
            if down <= 0:
                raise ValueError("vehicle calibration: bottom camera ray cannot see the floor")
            blind = max(0.0, forward + mount / math.tan(down) - nose)
            if data.get("precleared_forward_m") is not None:
                blind = min(blind, number("precleared_forward_m", 0, 6.0))
    vehicle = Vehicle(width, nose, tail, brake, tuple(dist), blind)
    camera = replace(base.camera, width=640, height=480, fx=number("fx", 1, 5000, intr),
                     fy=fy, cx=number("cx", 0, 639, intr),
                     cy=cy, mount_height_m=mount,
                     pitch_deg=pitch, forward_m=forward,
                     left_m=number("camera_left_m", -width / 2, width / 2))
    robot = replace(base.robot, vehicle=vehicle, half_width_m=width / 2, safety_margin_m=margin,
                    max_linear_mps=MAX_SPEED, caution_linear_mps=CAUTION_SPEED,
                    stop_distance_m=brake_allowance + 0.1 + MAX_SPEED * 0.5)
    planner = replace(base.planner, grid_width_m=3.2, resolution_m=0.05,
                      lookahead_m=max(1.40, robot.stop_distance_m))
    return replace(base, camera=camera, robot=robot, planner=planner)


def steering(v, omega):
    if not (math.isfinite(v) and math.isfinite(omega) and 0 < v <= MAX_SPEED):
        raise ValueError("invalid physical speed")
    angle = math.degrees(math.atan(WHEELBASE * omega / v))
    if abs(angle) > MAX_STEER + 1e-6:
        raise ValueError("unachievable physical curvature")
    return angle
