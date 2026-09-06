"""Independent synthetic ground truth tests for geometry and pixel error definitions."""

from typing import Any

import cv2
import numpy as np
import pytest

from src.calibration import check_model
from src.corners import object_points
from src.pose import estimate, positive_depth, project, rms
from src.validation import geometry, grouped_folds, held_out_corners


def synthetic() -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    board = dict(columns=12, rows=9, square_mm=19.0)
    points = object_points(board).astype(np.float64)
    kl = np.array([[800.0, 0, 640], [0, 810, 360], [0, 0, 1]])
    kr = np.array([[805.0, 0, 635], [0, 815, 362], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    translation = np.array([[-120.0], [1.0], [2.0]])
    model = dict(
        K_left=kl,
        K_right=kr,
        D_left=np.zeros((1, 5)),
        D_right=np.zeros((1, 5)),
        R_RL=rotation,
        t_RL=translation,
        size=(1280, 720),
    )
    pose = np.array([0.18, -0.24, 0.04, -70, -60, 850.0])
    # Build ground truth directly from transformed 3-D points, independent of project().
    left_xyz = cv2.Rodrigues(pose[:3])[0] @ points.T + pose[3:, None]
    right_xyz = rotation @ left_xyz + translation
    left_h, right_h = kl @ left_xyz, kr @ right_xyz
    obs = dict(
        pair_id="000001",
        left=(left_h[:2] / left_h[2]).T.reshape(-1, 1, 2),
        right=(right_h[:2] / right_h[2]).T.reshape(-1, 1, 2),
        size=(1280, 720),
    )
    return points, model, pose, obs


def test_rms_counts_points_not_coordinates() -> None:
    assert rms(np.array([[3, 4], [0, 0]])) == pytest.approx(np.sqrt(12.5))


def test_projection_transform_and_inverse() -> None:
    points, model, pose, obs = synthetic()
    left, right = project(points, pose, model)
    np.testing.assert_allclose(left, obs["left"].reshape(-1, 2), atol=1e-9)
    np.testing.assert_allclose(right, obs["right"].reshape(-1, 2), atol=1e-9)
    x = np.array([[12.0], [20.0], [1000.0]])
    np.testing.assert_allclose(
        model["R_RL"].T @ (model["R_RL"] @ x + model["t_RL"] - model["t_RL"]), x
    )


def test_recover_known_stereo_pose() -> None:
    points, model, pose, obs = synthetic()
    fit = estimate(points, obs, model)
    np.testing.assert_allclose(fit["pose"], pose, atol=1e-5)
    assert fit["joint_rms_px"] < 1e-7
    assert fit["positive_depth"]


def test_held_out_corners_with_known_pose() -> None:
    points, model, _, obs = synthetic()
    assert held_out_corners(points, obs, model) < 1e-7


def test_noisy_stereo_refinement_reduces_joint_residual() -> None:
    points, model, _, obs = synthetic()
    rng = np.random.default_rng(42)
    for side in ("left", "right"):
        obs[side] += rng.normal(0, 0.2, obs[side].shape)
    fit = estimate(points, obs, model)
    assert fit["joint_rms_px"] < fit["p0_joint_rms_px"]
    assert fit["joint_rms_px"] < 0.4


def test_triangulated_edge_scale_and_epipolar_error() -> None:
    _, model, _, obs = synthetic()
    metrics = geometry(obs, model, dict(columns=12, rows=9, square_mm=19.0))
    assert metrics["edge_mean_mm"] == pytest.approx(19, abs=1e-8)
    assert metrics["epipolar_mean_px"] < 1e-8
    assert metrics["triangulation_positive_depth"]


def test_reject_improper_rotation_and_negative_depth() -> None:
    points, model, pose, _ = synthetic()
    check_model(model)
    pose[5] = -850
    assert not positive_depth(points, pose, model)
    model["R_RL"] = -np.eye(3)
    with pytest.raises(ValueError, match="determinant"):
        check_model(model)


def test_near_duplicates_do_not_leak_between_folds() -> None:
    observations = [
        dict(left=np.full((4, 1, 2), i * 10.0), right=np.full((4, 1, 2), i * 10.0))
        for i in range(10)
    ]
    observations.append(observations[0])
    folds = grouped_folds(observations, 5, 42, 1.0)
    assert sorted(i for fold in folds for i in fold) == list(range(11))
    assert next(f for f in folds if 0 in f) == next(f for f in folds if 10 in f)
