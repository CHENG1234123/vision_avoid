import pytest
from pydantic import ValidationError

from vision_avoid.scene import SCENE_SCANNER_PROMPT, SceneScan, motion_allowed, scene_schema


def test_scene_prompt_requires_simplified_chinese_output():
    assert "简体中文" in SCENE_SCANNER_PROMPT
    assert "summary" in SCENE_SCANNER_PROMPT
    assert "hazards" in SCENE_SCANNER_PROMPT


def test_scene_scan_allows_safe_result():
    scan = SceneScan.model_validate(
        {
            "traversability": "caution",
            "safe_direction": "left",
            "summary": "Narrow but visible path",
            "hazards": [
                {
                    "label": "cable",
                    "direction": "center",
                    "distance_band": "near",
                    "severity": "medium",
                }
            ],
            "requires_operator": False,
            "confidence": 0.8,
        }
    )
    assert scan.allows_motion(0.6)
    assert scan.speed_cap(0.25, 0.15) == 0.15
    assert scene_schema()["required"]


def test_unknown_or_operator_required_blocks_motion():
    unknown = SceneScan(
        traversability="unknown",
        safe_direction="unknown",
        summary="Not enough detail",
        confidence=0.9,
    )
    operator = SceneScan(
        traversability="clear",
        safe_direction="center",
        summary="Needs confirmation",
        requires_operator=True,
        confidence=0.9,
    )
    assert not unknown.allows_motion(0.6)
    assert not operator.allows_motion(0.6)


def test_unknown_direction_is_only_a_local_fallback_for_caution():
    stop = SceneScan(
        traversability="clear",
        safe_direction="stop",
        summary="Contradictory result",
        confidence=0.9,
    )
    unknown = SceneScan(
        traversability="clear",
        safe_direction="unknown",
        summary="Direction not visible",
        confidence=0.9,
    )
    caution = SceneScan(
        traversability="caution",
        safe_direction="unknown",
        summary="Side paths are visible but not ranked",
        hazards=[
            {
                "label": "box",
                "direction": "center",
                "distance_band": "near",
                "severity": "medium",
            }
        ],
        confidence=0.9,
    )
    assert not stop.allows_motion(0.6)
    assert not unknown.allows_motion(0.6)
    assert caution.allows_motion(0.6)


def test_caution_without_a_listed_hazard_stays_closed():
    vague = SceneScan(
        traversability="caution",
        safe_direction="left",
        summary="Something may be wrong",
        confidence=0.9,
    )
    assert not vague.allows_motion(0.6)


def test_high_severity_hazard_requires_operator_review():
    scan = SceneScan(
        traversability="caution",
        safe_direction="left",
        summary="Glass hazard visible",
        hazards=[
            {
                "label": "glass",
                "direction": "center",
                "distance_band": "near",
                "severity": "high",
            }
        ],
        confidence=0.9,
    )
    assert not scan.allows_motion(0.6)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValidationError):
        SceneScan(
            traversability="clear",
            safe_direction="center",
            summary="bad",
            confidence=1.5,
        )


def test_local_safety_cannot_be_overridden_by_safe_vlm():
    scan = SceneScan(
        traversability="clear",
        safe_direction="center",
        summary="VLM sees a path",
        confidence=0.99,
    )
    assert scan.allows_motion(0.6)
    assert not motion_allowed(scan, 0.6, local_path_safe=False)
