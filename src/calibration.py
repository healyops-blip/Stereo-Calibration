"""Five-coefficient pinhole calibration, with optional joint intrinsic refinement."""

from typing import Any

import cv2
import numpy as np


def calibrate(
    observations: list[dict[str, Any]], points: Any, joint: bool = False
) -> dict[str, Any]:
    if len(observations) < 6:
        raise ValueError("At least six complete stereo observations are required")
    objects = [points] * len(observations)
    left = [o["left"] for o in observations]
    right = [o["right"] for o in observations]
    size = observations[0]["size"]
    lrms, kl, dl, _, _ = cv2.calibrateCamera(objects, left, size, None, None)
    rrms, kr, dr, _, _ = cv2.calibrateCamera(objects, right, size, None, None)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS if joint else cv2.CALIB_FIX_INTRINSIC
    rms, kl, dl, kr, dr, rotation, translation, _, fundamental = cv2.stereoCalibrate(
        objects,
        left,
        right,
        kl,
        dl,
        kr,
        dr,
        size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8),
    )
    model = dict(
        K_left=kl,
        D_left=dl,
        K_right=kr,
        D_right=dr,
        R_RL=rotation,
        t_RL=translation,
        F=fundamental,
        size=size,
        stereo_rms_px=rms,
        mono_left_rms_px=lrms,
        mono_right_rms_px=rrms,
        baseline_mm=float(np.linalg.norm(translation)),
        retained_ids=[o["pair_id"] for o in observations],
    )
    check_model(model)
    return model


def check_model(model: dict[str, Any]) -> None:
    for key in ("K_left", "D_left", "K_right", "D_right", "R_RL", "t_RL"):
        if not np.isfinite(model[key]).all():
            raise ValueError(f"Nonfinite camera parameter: {key}")
    for side in ("left", "right"):
        if model[f"K_{side}"][0, 0] <= 0 or model[f"K_{side}"][1, 1] <= 0:
            raise ValueError("Nonpositive focal length")
    rotation = model["R_RL"]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Invalid rotation orthogonality")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Invalid rotation determinant")
    if np.linalg.norm(model["t_RL"]) <= 1e-9:
        raise ValueError("Degenerate stereo baseline")
