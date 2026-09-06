"""Integration test from neural response peaks to an ordered checkerboard grid."""

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.deeplearning.infer_grid import detect_ordered  # noqa: E402
from src.deeplearning.synthetic import heatmap  # noqa: E402


def test_response_to_subpixel_ordered_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The neural path must return all 108 ordered points without a chessboard detector."""
    image_size = 240
    expected = np.mgrid[:12, :9].T.reshape(-1, 2).astype(np.float32) * 15 + [30, 40]
    response = heatmap(expected, image_size, sigma=1.2)

    def forbidden_detector(*args: object, **kwargs: object) -> None:
        raise AssertionError("Traditional checkerboard detectors must not be called")

    monkeypatch.setattr(cv2, "findChessboardCorners", forbidden_detector)
    monkeypatch.setattr(cv2, "findChessboardCornersSB", forbidden_detector)
    monkeypatch.setattr(
        "src.deeplearning.real_evaluate.dense_response",
        lambda network, image: response,
    )

    result = detect_ordered(
        network=None,
        image=np.zeros((image_size, image_size), dtype=np.uint8),
        shape=(12, 9),
        camera=None,
        method="log3",
    )

    assert result["corners"].shape == (108, 2)
    assert np.allclose(result["corners"], expected, atol=1e-5)
    assert result["reference_assisted"] is False
