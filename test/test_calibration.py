import numpy as np
import pytest

from vision_avoid.calibrate_camera import calibrate, detect_views
from vision_avoid.download_models import HF_MODELS


def test_calibration_rejects_unusable_views():
    blank = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(5)]
    with pytest.raises(ValueError, match="at least 5"):
        calibrate(blank, (9, 6), 0.025)


def test_model_revisions_are_pinned():
    assert HF_MODELS
    for repo_id, revision in HF_MODELS.values():
        assert repo_id
        assert len(revision) == 40


def test_detect_views_requires_an_image():
    with pytest.raises(ValueError, match="no images"):
        detect_views([], (9, 6))
