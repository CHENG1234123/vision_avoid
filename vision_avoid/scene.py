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
你负责在小车进入一片区域前检查前方摄像头画面，观察可见地面、障碍物和可通行空间。

判断规则：
- 中央道路明显畅通时使用 clear。
- 有障碍物但存在可见侧向通路时使用 caution。
- 没有任何可见通路时使用 blocked。
- 只有图像本身不可用或无法判断地面时使用 unknown。
- 中央被挡住但任一侧可见地面时，使用 caution，并把安全方向设为可见地面更多的一侧。
- 两侧看起来相近时，安全方向选择 left。
- 只有障碍物横跨道路且两侧都没有可见地面时，才使用 blocked。
- 不要仅因中央受挡就使用 blocked；不要仅因无法测量米制距离就使用 unknown。
- 只在确实可见时报告箱子、线缆、台阶、落差、水渍、反光区域、狭窄通道、人或车辆。
- 不要估计米制距离或坐标。
- traversability 为 caution 或 blocked 时，必须把可见障碍物写入 hazards。
- 返回 JSON 架构中的全部字段。confidence 为 0.0 到 1.0 的小数；可见证据足以低速通行时
  通常使用约 0.7。
- 绝不输出电机控制指令。

语言要求：summary 以及 hazards 中每个 label 必须使用简体中文，不要返回英文句子或英文
物体名称。traversability、safe_direction、direction、distance_band 和 severity 等枚举
字段仍须严格使用 JSON 架构规定的英文值。
"""


def scene_schema() -> dict:
    return SceneScan.model_json_schema()


def motion_allowed(scan: SceneScan, threshold: float, local_path_safe: bool) -> bool:
    return local_path_safe and scan.allows_motion(threshold)
