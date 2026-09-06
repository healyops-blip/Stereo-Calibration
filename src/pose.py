"""Left-only IPPE prediction and fixed-camera, six-DOF stereo refinement."""

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


def rms(residual: Any) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(residual).reshape(-1, 2) ** 2, axis=1))))


def project(points: Any, pose: Any, model: dict[str, Any]) -> tuple[Any, Any]:
    # Finite-difference optimization requires float64 projection precision.
    points = np.asarray(points, dtype=np.float64)
    rotation = cv2.Rodrigues(pose[:3])[0]
    translation = np.asarray(pose[3:]).reshape(3, 1)
    right_rotation = model["R_RL"] @ rotation
    right_translation = model["R_RL"] @ translation + model["t_RL"].reshape(3, 1)
    left = cv2.projectPoints(points, pose[:3], translation, model["K_left"], model["D_left"])[
        0
    ].reshape(-1, 2)
    right = cv2.projectPoints(
        points,
        cv2.Rodrigues(right_rotation)[0],
        right_translation,
        model["K_right"],
        model["D_right"],
    )[0].reshape(-1, 2)
    return left, right


def positive_depth(points: Any, pose: Any, model: dict[str, Any]) -> bool:
    left = cv2.Rodrigues(pose[:3])[0] @ points.T + np.asarray(pose[3:]).reshape(3, 1)
    right = model["R_RL"] @ left + model["t_RL"].reshape(3, 1)
    return bool(np.all(left[2] > 0) and np.all(right[2] > 0))


def estimate(points: Any, obs: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    found, rotations, translations, _ = cv2.solvePnPGeneric(
        points, obs["left"], model["K_left"], model["D_left"], flags=cv2.SOLVEPNP_IPPE
    )
    if not found:
        raise ValueError("IPPE_failed")

    def residual(pose: Any) -> Any:
        left, right = project(points, pose, model)
        return np.concatenate(
            (left - obs["left"].reshape(-1, 2), right - obs["right"].reshape(-1, 2))
        ).ravel()

    candidates = [np.r_[r.ravel(), t.ravel()] for r, t in zip(rotations, translations, strict=True)]
    candidates = [p for p in candidates if positive_depth(points, p, model)]
    if not candidates:
        raise ValueError("no_positive_depth_IPPE_solution")
    # P0 must be chosen with LEFT observations alone to keep right prediction independent.
    candidates.sort(key=lambda p: rms(project(points, p, model)[0] - obs["left"].reshape(-1, 2)))
    initial = candidates[0]
    initial_residual = residual(initial).reshape(2, -1, 2)
    solutions = []
    for candidate in candidates:
        fit = least_squares(
            residual,
            candidate,
            method="trf",
            loss="linear",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=300,
        )
        if fit.success and positive_depth(points, fit.x, model):
            solutions.append((rms(fit.fun), fit.x))
    if not solutions:
        raise ValueError("stereo_pose_optimization_failed")
    solutions.sort(key=lambda value: value[0])
    score, pose = solutions[0]
    distinct = [s for s in solutions[1:] if np.linalg.norm(s[1] - pose) > 1e-3]
    ambiguous = bool(distinct and distinct[0][0] - score < 0.05)
    error = residual(pose).reshape(2, -1, 2)
    left_scores = [
        rms(project(points, p, model)[0] - obs["left"].reshape(-1, 2)) for p in candidates
    ]
    return dict(
        pose=pose,
        initial_pose=initial,
        left_rms_px=rms(error[0]),
        right_rms_px=rms(error[1]),
        joint_rms_px=score,
        p0_left_rms_px=rms(initial_residual[0]),
        p0_right_prediction_rms_px=rms(initial_residual[1]),
        p0_joint_rms_px=rms(initial_residual),
        ambiguity_flag=ambiguous,
        p0_ambiguity_flag=len(left_scores) > 1 and left_scores[1] - left_scores[0] < 0.05,
        positive_depth=True,
    )
