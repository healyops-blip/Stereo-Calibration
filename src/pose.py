"""Left-only IPPE prediction and fixed-camera, six-DOF stereo refinement."""

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


def rms(residual: Any) -> float:
    """计算二维点残差的均方根欧氏距离。

    Args:
        residual: 可重排为 (N, 2) 的非空残差数组，通常单位 px。

    Returns:
        sqrt(mean(dx**2 + dy**2))，单位与输入相同，不是分量级 RMS。

    Note:
        不主动拒绝空数组或非有限值；调用方须保证输入有效。
    """
    return float(np.sqrt(np.mean(np.sum(np.asarray(residual).reshape(-1, 2) ** 2, axis=1))))


def project(points: Any, pose: Any, model: dict[str, Any]) -> tuple[Any, Any]:
    """将棋盘三维点投影到原始左右相机图像。

    Args:
        points: 棋盘坐标 (N, 3)，单位 mm；投影时转为 float64。
        pose: 棋盘到左相机的六维位姿。前三项为 Rodrigues 旋转向量
            （rad），后三项为平移（mm），不是欧拉角。
        model: 含左右 K、D 和左到右 R_RL、t_RL 的相机模型。

    Returns:
        左、右投影坐标两个 (N, 2) 数组，单位 px，保留镜头畸变。

    Raises:
        cv2.error: 旋转或投影输入不符合 OpenCV 要求。
        KeyError: 缺少相机字段。

    Note:
        不进行正深度筛选。
    """
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
    """判断所有棋盘点是否同时位于左右相机前方。

    Args:
        points: 棋盘坐标数组 (N, 3)，单位 mm。
        pose: 棋盘到左相机的旋转向量（rad）和平移（mm），形状 (6,)。
        model: 含左到右 R_RL 和 t_RL 的模型。

    Returns:
        所有点在两相机中的 Z 都严格大于零时为 True。
        空点集也返回 True，因此调用方必须另行检查点数。
    """
    left = cv2.Rodrigues(pose[:3])[0] @ points.T + np.asarray(pose[3:]).reshape(3, 1)
    right = model["R_RL"] @ left + model["t_RL"].reshape(3, 1)
    return bool(np.all(left[2] > 0) and np.all(right[2] > 0))


def estimate(points: Any, obs: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """用左图 IPPE 初始化，再在固定相机参数下优化双目六自由度位姿。

    Args:
        points: 编号一致的平面棋盘坐标 (N, 3)，单位 mm，须满足 IPPE 要求。
        obs: left、right 图像角点数组 (N, 1, 2)，单位 px。
        model: 含左右内参、畸变和双目外参的已标定模型。

    Returns:
        含最终 pose、左图选择的 initial_pose（均为形状 (6,)）、
        左右及联合 RMS、P0 右图预测 RMS、歧义和正深度标记的字典。
        位姿前三项是旋转向量 rad，后三项是平移 mm；残差单位 px。

    Raises:
        ValueError: IPPE 未找到解、没有正深度候选，或双目优化失败。
        cv2.error: 输入不满足 OpenCV 求解或投影要求。

    Note:
        P0 候选排序仅使用左图观测；双目优化使用两图观测。
        像素残差不等同于真实位姿误差。
    """
    found, rotations, translations, _ = cv2.solvePnPGeneric(
        points, obs["left"], model["K_left"], model["D_left"], flags=cv2.SOLVEPNP_IPPE
    )
    if not found:
        raise ValueError("IPPE_failed")

    def residual(pose: Any) -> Any:
        """返回当前候选 pose 在左右图的连续二维投影残差展开向量，单位 px。"""
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
