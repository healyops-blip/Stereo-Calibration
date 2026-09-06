"""Independent checks for synthetic coordinates, labels and dataset isolation."""

import cv2
import numpy as np
import pytest

from src.deeplearning.synthetic import heatmap, project_board, render_crop, scene_seed


def test_frontal_projection_matches_closed_form() -> None:
    camera = dict(K=[[500, 0, 320], [0, 500, 240], [0, 0, 1]], D=[0] * 5)
    board = dict(columns=12, rows=9, square_mm=19.0)
    points = project_board(camera, board, np.eye(3), np.array([0, 0, 500]))
    assert np.allclose(points[0], [320, 240])
    assert np.allclose(points[-1], [529, 392])


def test_rendered_intersection_matches_noninteger_label() -> None:
    camera = dict(K=[[500, 0, 50.25], [0, 500, 50.35], [0, 0, 1]], D=[0] * 5)
    board = dict(columns=4, rows=3, square_mm=19.0)
    image = render_crop(camera, board, np.eye(3), np.array([0, 0, 500]), (0, 0), 120, 4)
    initial = np.array([[[50.0, 50.0]]], dtype=np.float32)
    refined = cv2.cornerSubPix(
        image,
        initial,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-5),
    )
    assert np.linalg.norm(refined[0, 0] - [50.25, 50.35]) < 0.2


def test_heatmap_preserves_phase_and_no_corner_image() -> None:
    target = heatmap(np.array([[20.25, 18.1]]), 40)
    assert target[18, 20] > target[18, 21] > target[18, 19]
    assert np.count_nonzero(heatmap(np.empty((0, 2)), 40)) == 0


def test_split_seeds_do_not_overlap() -> None:
    sets = [{scene_seed(split, i) for i in range(2000)} for split in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    with pytest.raises(ValueError):
        scene_seed("unknown", 0)
