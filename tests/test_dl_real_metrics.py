"""Reference agreement is not ground-truth accuracy."""

import numpy as np

from src.deeplearning.real_metrics import match_reference, symmetric_epipolar


def test_matching_is_one_to_one_and_gated() -> None:
    reference = np.array([[10, 10], [20, 20], [30, 30]])
    predicted = np.array([[10.1, 10], [10.2, 10], [20.8, 20], [34, 30]])
    indices, errors = match_reference(reference, predicted)
    assert indices.tolist() == [0, 2, -1]
    assert np.allclose(errors, [0.1, 0.8])


def test_matching_handles_empty_candidates() -> None:
    indices, errors = match_reference(np.zeros((2, 2)), np.empty((0, 2)))
    assert indices.tolist() == [-1, -1]
    assert not len(errors)


def test_rectified_epipolar_error_matches_vertical_offset() -> None:
    model = dict(
        K_left=np.eye(3),
        K_right=np.eye(3),
        D_left=np.zeros(5),
        D_right=np.zeros(5),
        F=np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
    )
    left = np.array([[1.0, 2.0], [3.0, 4.0]])
    right = left + [3, 0.5]
    assert np.allclose(symmetric_epipolar(left, right, model), 0.5)
