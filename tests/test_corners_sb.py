"""Minimal executable checks for the explicit OpenCV SB detector path."""

import numpy as np
import pytest

from src.corners import detect


def make_checkerboard(
    inner_columns: int = 12,
    inner_rows: int = 9,
    square_px: int = 36,
    margin_px: int = 48,
) -> np.ndarray:
    """Create an ideal board with one more square than inner corners per axis."""
    square_columns = inner_columns + 1
    square_rows = inner_rows + 1
    height = square_rows * square_px + 2 * margin_px
    width = square_columns * square_px + 2 * margin_px
    image = np.full((height, width), 255, dtype=np.uint8)
    for row in range(square_rows):
        for column in range(square_columns):
            if (row + column) % 2 == 0:
                y_start = margin_px + row * square_px
                x_start = margin_px + column * square_px
                image[y_start : y_start + square_px, x_start : x_start + square_px] = 0
    return image


def test_sb_detects_complete_synthetic_board() -> None:
    image = make_checkerboard()

    corners, method = detect(image, (12, 9), detector="sb")

    grid = corners.reshape(9, 12, 2)
    assert method == "SB"
    assert corners.shape == (108, 1, 2)
    assert corners.dtype == np.float32
    assert np.all(np.diff(grid[0, :, 0]) > 0)
    assert np.all(np.diff(grid[:, 0, 1]) > 0)


def test_sb_rejects_image_without_checkerboard() -> None:
    blank = np.full((456, 564), 127, dtype=np.uint8)

    with pytest.raises(ValueError, match="corners_not_found"):
        detect(blank, (12, 9), detector="sb")
