"""Grid assembly must consume unordered points, never a traditional detector."""

import cv2
import numpy as np
import pytest

from src.deeplearning.grid import assemble_grid


@pytest.mark.parametrize("angle", [0, 30, 80, 140])
def test_grid_with_perspective_clutter_and_shuffled_indices(angle: int) -> None:
    rng = np.random.default_rng(15)
    grid = np.mgrid[:12, :9].T.reshape(-1, 1, 2).astype(np.float32)
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    h = np.array([[25 * c, -25 * s, 500], [25 * s, 25 * c, 350], [0.009, 0.005, 1.0]])
    expected = cv2.perspectiveTransform(grid, h).reshape(-1, 2)
    clutter = rng.uniform([10, 10], [100, 150], (20, 2))
    candidates = np.r_[expected, clutter].astype(np.float32)
    permutation = rng.permutation(len(candidates))
    result = assemble_grid(candidates[permutation], (12, 9))
    selected = permutation[result["indices"]]
    assert set(selected) == set(range(108))
    assert result["homography_rms_px"] < 0.01


def test_incomplete_board_rejected() -> None:
    points = np.mgrid[:12, :9].T.reshape(-1, 2).astype(np.float32) * 20 + 50
    with pytest.raises(ValueError):
        assemble_grid(points[:-1], (12, 9))


def test_unstructured_background_rejected() -> None:
    points = np.random.default_rng(91).uniform(0, 1000, (150, 2))
    with pytest.raises(ValueError):
        assemble_grid(points, (12, 9))


def test_nonfinite_candidates_rejected() -> None:
    with pytest.raises(ValueError):
        assemble_grid(np.full((108, 2), np.nan), (12, 9))


def test_missing_corner_not_replaced_with_background_point() -> None:
    points = np.mgrid[:12, :9].T.reshape(-1, 2).astype(np.float32) * 20 + 200
    points = np.r_[np.delete(points, 54, axis=0), [[10, 10], [20, 30], [15, 60]]]
    with pytest.raises(ValueError):
        assemble_grid(points, (12, 9))


def test_distorted_grid_preserves_original_pixel_coordinates() -> None:
    grid = np.zeros((108, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[:12, :9].T.reshape(-1, 2) * 19
    camera = dict(
        K=np.array([[770.0, 0, 630], [0, 770, 360], [0, 0, 1]]),
        D=np.array([-0.36, 0.2, 0.001, 0.002, -0.08]),
    )
    points = cv2.projectPoints(
        grid, np.array([0.2, 0.1, 0.3]), np.array([-10.0, -50, 600]), camera["K"], camera["D"]
    )[0][:, 0]
    result = assemble_grid(points[::-1], (12, 9), camera)
    assert np.array_equal(result["corners"], points[::-1][result["indices"]])
    assert result["homography_rms_px"] < 0.01
