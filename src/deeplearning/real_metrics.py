"""Paired reference agreement and distortion-aware geometric consistency."""

from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_reference(reference: Any, predicted: Any, radius: float = 1.0) -> tuple[Any, Any]:
    """用一对一匹配及距离门限比较预测点与参考点。

    Args:
        reference: 参考坐标 (N, 2)，px；不一定是物理真值。
        predicted: 无序预测坐标 (M, 2)，px。
        radius: 接受匹配的最大欧氏距离，px，应为正。

    Returns:
        每个参考点对应的预测索引（未匹配为 -1），以及仅成功匹配的距离数组。
        任一输入为空时距离数组为空。
    """
    reference = np.asarray(reference).reshape(-1, 2)
    predicted = np.asarray(predicted).reshape(-1, 2)
    indices = np.full(len(reference), -1, dtype=int)
    if not len(reference) or not len(predicted):
        return indices, np.empty(0)
    distances = np.linalg.norm(reference[:, None] - predicted[None], axis=2)
    penalty = (len(reference) + 1) * (radius + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= radius, distances, penalty))
    accepted = distances[rows, cols] <= radius
    indices[rows[accepted]] = cols[accepted]
    return indices, distances[rows[accepted], cols[accepted]]


def symmetric_epipolar(left: Any, right: Any, model: dict[str, Any]) -> Any:
    """去畸变后计算左右对称点到极线距离。

    Args:
        left: 左图对应坐标，可重排为 (N, 2)，px。
        right: 同序右图坐标，点数与 left 相同。
        model: 左右 K、D 以及与其对应的基础矩阵 F。

    Returns:
        每个对应点的左右点到线距离算术平均，形状 (N,)，单位 px。

    Note:
        不等同于立体校正后非视差轴误差；分母用下限防止除零。
    """
    # OpenCV 4 exposes Iter separately; OpenCV 5 accepts criteria on undistortPoints.
    undistort = getattr(cv2, "undistortPointsIter", cv2.undistortPoints)
    homogeneous = []
    for side, points in [("left", left), ("right", right)]:
        k = np.asarray(model[f"K_{side}"], dtype=np.float64)
        points = undistort(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
            k,
            np.asarray(model[f"D_{side}"], dtype=np.float64),
            R=None,
            P=k,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-10),
        ).reshape(-1, 2)
        homogeneous.append(np.c_[points, np.ones(len(points))])
    left_h, right_h = homogeneous
    f = np.asarray(model["F"])
    line_r, line_l = left_h @ f.T, right_h @ f
    residual = np.abs(np.sum(right_h * line_r, axis=1))
    return (
        0.5
        * residual
        * (
            1 / np.maximum(np.linalg.norm(line_r[:, :2], axis=1), 1e-12)
            + 1 / np.maximum(np.linalg.norm(line_l[:, :2], axis=1), 1e-12)
        )
    )
