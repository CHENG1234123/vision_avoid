from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    mount_height_m: float
    pitch_deg: float


@dataclass(frozen=True)
class RobotConfig:
    half_width_m: float
    safety_margin_m: float
    max_linear_mps: float
    caution_linear_mps: float
    max_angular_rps: float
    stop_distance_m: float
    ground_tolerance_m: float
    obstacle_max_height_m: float


@dataclass(frozen=True)
class PlannerConfig:
    grid_length_m: float
    grid_width_m: float
    resolution_m: float
    lookahead_m: float
    time_step_s: float
    turn_duration_s: float
    yaw_rates: tuple[float, ...]
    hysteresis_s: float


@dataclass(frozen=True)
class SceneChangeConfig:
    sample_period_s: float
    hamming_threshold: int
    hold_seconds: float
    cooldown_seconds: float
    min_brightness: float
    min_contrast: float
    min_sharpness: float


@dataclass(frozen=True)
class VlmConfig:
    provider: str
    local_timeout_s: float
    scan_prior_s: float
    api_enabled: bool
    api_model: str
    api_base_url: str
    api_key_env: str
    api_timeout_s: float
    structured_output_mode: str
    max_calls_per_hour: int
    confidence_threshold: float
    lmstudio_model: str = "qwen3-vl-4b-instruct-vision"
    lmstudio_url: str = "http://127.0.0.1:1234/v1"
    api_enable_thinking: bool = False


@dataclass(frozen=True)
class AppConfig:
    profile: str
    models_root: Path
    detection_model: str
    depth_model: str
    camera: CameraConfig
    robot: RobotConfig
    planner: PlannerConfig
    scene_change: SceneChangeConfig
    vlm: VlmConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    required = {
        "profile",
        "models",
        "camera",
        "robot",
        "planner",
        "scene_change",
        "vlm",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{config_path} missing sections: {sorted(missing)}")

    planner_data = dict(data["planner"])
    planner_data["yaw_rates"] = tuple(float(x) for x in planner_data["yaw_rates"])
    root_value = os.environ.get("VISION_MODELS_DIR", data["models"]["root"])
    models_root = Path(root_value).expanduser()
    if not models_root.is_absolute():
        models_root = config_path.parent / models_root
    models_root = models_root.resolve()

    config = AppConfig(
        profile=str(data["profile"]),
        models_root=models_root,
        detection_model=str(data["models"]["detection"]),
        depth_model=str(data["models"]["depth"]),
        camera=CameraConfig(**data["camera"]),
        robot=RobotConfig(**data["robot"]),
        planner=PlannerConfig(**planner_data),
        scene_change=SceneChangeConfig(**data["scene_change"]),
        vlm=VlmConfig(**data["vlm"]),
        raw=data,
    )
    vlm = replace(
        config.vlm,
        provider=os.environ.get("VLM_PROVIDER", config.vlm.provider),
        lmstudio_model=os.environ.get(
            "VLM_LMSTUDIO_MODEL", config.vlm.lmstudio_model
        ),
        lmstudio_url=os.environ.get(
            "VLM_LMSTUDIO_URL", config.vlm.lmstudio_url
        ),
        api_enabled=_env_bool("VLM_API_ENABLED", config.vlm.api_enabled),
        api_model=os.environ.get("VLM_API_MODEL", config.vlm.api_model),
        api_base_url=os.environ.get("VLM_API_BASE_URL", config.vlm.api_base_url),
        api_key_env=os.environ.get("VLM_API_KEY_ENV", config.vlm.api_key_env),
        api_timeout_s=_env_float("VLM_API_TIMEOUT_S", config.vlm.api_timeout_s),
        structured_output_mode=os.environ.get(
            "VLM_STRUCTURED_OUTPUT_MODE", config.vlm.structured_output_mode
        ),
        api_enable_thinking=_env_bool(
            "VLM_API_ENABLE_THINKING", config.vlm.api_enable_thinking
        ),
    )
    config = replace(config, vlm=vlm)
    _validate(config)
    return config


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _validate(config: AppConfig) -> None:
    if config.profile not in {"indoor", "outdoor"}:
        raise ValueError("profile must be indoor or outdoor")
    if config.vlm.provider not in {"local", "api"}:
        raise ValueError("vlm.provider must be local or api")
    if config.vlm.structured_output_mode not in {"json_schema", "json_object"}:
        raise ValueError("structured_output_mode must be json_schema or json_object")
    if config.robot.caution_linear_mps > config.robot.max_linear_mps:
        raise ValueError("caution speed cannot exceed max speed")
    positive = {
        "camera.fx": config.camera.fx,
        "camera.fy": config.camera.fy,
        "robot.half_width_m": config.robot.half_width_m,
        "robot.max_linear_mps": config.robot.max_linear_mps,
        "robot.max_angular_rps": config.robot.max_angular_rps,
        "robot.stop_distance_m": config.robot.stop_distance_m,
        "planner.lookahead_m": config.planner.lookahead_m,
        "planner.resolution_m": config.planner.resolution_m,
        "scene_change.sample_period_s": config.scene_change.sample_period_s,
        "vlm.confidence_threshold": config.vlm.confidence_threshold,
        "vlm.scan_prior_s": config.vlm.scan_prior_s,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError(f"values must be positive: {bad}")
    if not 0.0 <= config.vlm.confidence_threshold <= 1.0:
        raise ValueError("vlm.confidence_threshold must be in [0, 1]")
    if not 0 <= config.scene_change.hamming_threshold <= 64:
        raise ValueError("scene_change.hamming_threshold must be in [0, 64]")
    if not config.planner.yaw_rates or 0.0 not in config.planner.yaw_rates:
        raise ValueError("planner.yaw_rates must include zero")
    if any(abs(rate) > config.robot.max_angular_rps for rate in config.planner.yaw_rates):
        raise ValueError("planner yaw rate exceeds robot.max_angular_rps")
