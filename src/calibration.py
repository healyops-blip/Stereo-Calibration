"""Five-coefficient pinhole calibration, with optional joint intrinsic refinement."""

from typing import Any

import cv2
import numpy as np


def calibrate(
    observations: list[dict[str, Any]], points: Any, joint: bool = False
) -> dict[str, Any]:
    """先标定左右相机，再求解五参数针孔双目模型。

    Args:
        observations: 至少六对完整观测。每项含 ``pair_id``、``size``
            （宽、高，px）及 ``left``、``right`` 角点数组 (N, 1, 2)。
            调用方保证图像尺寸和角点编号一致。
        points: 棋盘坐标 (N, 3)，单位 mm；与图像角点一一对应。
        joint: False 为 C0（固定单目内参）；True 为 C2（联合优化内参）。

    Returns:
        包含左右 K、D、左到右的 R_RL (3, 3) 与 t_RL (3, 1)、F、
        图像尺寸、保留编号和 RMS 的字典。基线与平移单位 mm，RMS 单位 px。

    Raises:
        ValueError: 观测不足六对，或求得的模型未通过有效性检查。
        cv2.error: OpenCV 拒绝输入或求解失败。

    Note:
        输入不做近重复分组；训练 RMS 不是外部绝对精度。
    """
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
    """检查相机参数的有限性、焦距、旋转矩阵和非零基线。

    Args:
        model: 含 K_left、D_left、K_right、D_right、R_RL、t_RL 的模型。
            调用方提供符合约定形状的数值数组。

    Raises:
        ValueError: 参数非有限、焦距非正、旋转不合法或基线退化。
        KeyError: 缺少必需字段。

    Note:
        不提供完整的输入形状校验。
    """
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
