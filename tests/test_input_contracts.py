"""Public boundaries reject invalid images, board geometry and camera shapes."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.calibration import check_model
from src.corners import detect, object_points, order_corners
from src.dataset import read_gray


@pytest.mark.parametrize(
    "image",
    [
        None,
        np.zeros((0, 10), np.uint8),
        np.zeros((10, 10, 3), np.uint8),
        np.zeros((10, 10), np.float32),
    ],
)
def test_detector_rejects_invalid_grayscale(image: Any) -> None:
    with pytest.raises(ValueError, match="grayscale"):
        detect(image, (12, 9))


@pytest.mark.parametrize("spacing", [0.0, -1.0, float("nan"), float("inf")])
def test_board_spacing_is_positive_and_finite(spacing: float) -> None:
    with pytest.raises(ValueError, match="square_mm"):
        object_points(dict(columns=12, rows=9, square_mm=spacing))


def test_ordering_rejects_wrong_count_and_nonfinite_coordinates() -> None:
    with pytest.raises(ValueError, match="corners"):
        order_corners(np.zeros((107, 1, 2), np.float32), (12, 9))
    with pytest.raises(ValueError, match="corners"):
        order_corners(np.full((108, 1, 2), np.nan, np.float32), (12, 9))


def test_empty_image_error_contains_path(tmp_path: Path) -> None:
    path = tmp_path / "empty.bmp"
    path.touch()
    with pytest.raises(ValueError, match="empty.bmp"):
        read_gray(path)


def test_camera_shape_error_names_parameter() -> None:
    model = dict(
        K_left=np.eye(2),
        K_right=np.eye(3),
        D_left=np.zeros(5),
        D_right=np.zeros(5),
        R_RL=np.eye(3),
        t_RL=np.array([1.0, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="K_left.*shape"):
        check_model(model)
