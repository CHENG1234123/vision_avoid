from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Traversability(str, Enum):
    CLEAR = "clear"
    CAUTION = "caution"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class Hazard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80)
    direction: Literal["left", "center", "right", "unknown"]
    distance_band: Literal["near", "medium", "far", "unknown"]
    severity: Literal["low", "medium", "high"]


class SceneScan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    traversability: Literal["clear", "caution", "blocked", "unknown"]
    safe_direction: Literal["left", "center", "right", "stop", "unknown"]
    summary: str = Field(min_length=1, max_length=500)
    hazards: list[Hazard] = Field(default_factory=list, max_length=20)
    requires_operator: bool = False
    confidence: float = Field(ge=0.0, le=1.0)

    def allows_motion(self, threshold: float) -> bool:
        valid_directions = {"left", "center", "right"}
        if self.traversability == Traversability.CAUTION.value:
            valid_directions.add("unknown")
        return (
            self.traversability in {Traversability.CLEAR.value, Traversability.CAUTION.value}
            and self.confidence >= threshold
            and not self.requires_operator
            and not any(hazard.severity == "high" for hazard in self.hazards)
            and self.safe_direction in valid_directions
            and (
                self.traversability != Traversability.CAUTION.value
                or bool(self.hazards)
            )
        )

    def speed_cap(self, normal: float, caution: float) -> float:
        if self.traversability == Traversability.CAUTION.value:
            return min(normal, caution)
        return normal


SCENE_SCANNER_PROMPT = """\
You inspect one forward-facing ground-robot camera frame before the robot enters an area.
Look at the visible floor, obstacles and free space.
Use clear when the center path is visibly open, caution when an obstacle exists but a
visible side path exists, blocked when no visible route remains, and unknown only when
the image itself is unusable or the floor cannot be judged.
If the center is obstructed and floor is visible on either side, use caution and set
safe_direction to the side with more visible floor. If both sides look similar, choose left.
Use blocked only when the obstacle spans the route and no floor is visible on either side.
Do not use blocked just because the center is obstructed. Do not use unknown merely
because metric distance cannot be measured.
Report a box, cable, step, drop, puddle, reflective area, narrow passage, person or
vehicle only when it is visible. Do not estimate metric distance or coordinates.
If traversability is caution or blocked, include the visible obstacle in hazards.
Return every field in the JSON schema. confidence is a decimal from 0.0 to 1.0;
use about 0.7 when the visible evidence is clear enough for cautious movement.
Never provide motor commands.
"""


def scene_schema() -> dict:
    return SceneScan.model_json_schema()


def motion_allowed(scan: SceneScan, threshold: float, local_path_safe: bool) -> bool:
    return local_path_safe and scan.allows_motion(threshold)
