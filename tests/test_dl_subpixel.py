import cv2
import numpy as np
import pytest

from src.deeplearning.subpixel import refine


def test_log_quadratic_recovers_fractional_gaussian_peak() -> None:
    y, x = np.mgrid[:40, :40]
    response = np.exp(-((x - 20.25) ** 2 + (y - 18.1) ** 2) / 4.5)
    raw = np.array([[20.0, 18.0]], dtype=np.float32)
    result = refine(np.zeros((40, 40), np.uint8), raw, response, "log3")
    assert np.allclose(result, [[20.25, 18.1]], atol=1e-5)
    assert np.array_equal(raw, [[20, 18]])


@pytest.mark.parametrize("method", ["win3", "win5", "win7", "log3"])
def test_empty_points_supported(method: str) -> None:
    assert refine(
        np.zeros((40, 40), np.uint8), np.empty((0, 2)), np.zeros((40, 40)), method
    ).shape == (0, 2)


def test_flat_response_retains_original_point() -> None:
    points = np.array([[20.0, 20.0]], dtype=np.float32)
    assert np.array_equal(
        refine(np.zeros((40, 40), np.uint8), points, np.ones((40, 40)), "log3"), points
    )


@pytest.mark.parametrize("method", ["win3", "win5", "win7"])
def test_cv_refinement_array_layout_and_no_input_mutation(method: str) -> None:
    image = np.zeros((64, 64), np.uint8)
    image[32:, 32:] = 255
    points = np.array([[31.0, 31.0], [32.0, 32.0]], dtype=np.float32)
    before = points.copy()
    refined = refine(image, points, np.zeros_like(image), method)
    assert np.isfinite(refined).all()
    assert np.array_equal(points, before)


def test_win3_matches_legacy_q0_for_noncontiguous_points() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[32:, 32:] = 255
    predicted = np.array([[31, 30], [31, 30]], dtype=np.float32).T
    original = predicted.copy()
    assert not predicted.flags.c_contiguous
    expected = cv2.cornerSubPix(
        image,
        np.array(predicted.reshape(-1, 1, 2), dtype=np.float32, order="C", copy=True),
        (3, 3),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
    )[:, 0].astype(np.float32)
    actual = refine(image, predicted, None, "win3")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(predicted, original)
    assert np.isfinite(actual).all()
    assert not np.shares_memory(actual, predicted)
