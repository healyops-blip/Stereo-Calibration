"""Bounded subpixel refinements with immutable raw peak coordinates."""

from typing import Any

import cv2
import numpy as np

METHODS = ("win3", "win5", "win7", "log3")


def refine(image: Any, peaks: Any, response: Any, method: str) -> Any:
    """在不修改原始峰值的前提下执行窗口细化或对数二次峰值拟合。

    Args:
        image: 灰度图 (H, W)，窗口法需要 uint8 或 OpenCV 支持的浮点图像。
        peaks: 可重排为 (N, 2) 的候选坐标，px，可非连续。
        response: 网络概率热图 (H, W)；win 方法不使用，可传 None。
        method: win3、win5、win7（半窗口）或 log3（3×3 响应拟合）。

    Returns:
        独立的 float32 坐标数组 (N, 2)，px；空输入仍返回 (0, 2)。
        log3 的边界、非负定曲面或超限偏移保留原始候选，不删点。

    Raises:
        ValueError: 方法未知。
        cv2.error: 窗口法输入不满足 OpenCV 要求。

    Note:
        log3 只接受有限且每轴绝对值不超过 1 px 的偏移；不进行网格编号。
    """
    if method not in METHODS:
        raise ValueError("Unknown refinement method")
    points = np.array(peaks, dtype=np.float32, order="C", copy=True).reshape(-1, 2)
    if not len(points):
        return points
    if method.startswith("win"):
        radius = int(method[3:])
        return cv2.cornerSubPix(
            image,
            np.array(points.reshape(-1, 1, 2), dtype=np.float32, order="C", copy=True),
            (radius, radius),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
        )[:, 0]
    y, x = np.mgrid[-1:2, -1:2]
    design = np.c_[
        x.ravel() ** 2, y.ravel() ** 2, (x * y).ravel(), x.ravel(), y.ravel(), np.ones(9)
    ]
    inverse = np.linalg.pinv(design)
    for i, point in enumerate(points):
        x0, y0 = np.rint(point).astype(int)
        if not (1 <= x0 < response.shape[1] - 1 and 1 <= y0 < response.shape[0] - 1):
            continue
        values = np.log(np.maximum(response[y0 - 1 : y0 + 2, x0 - 1 : x0 + 2].ravel(), 1e-8))
        a, b, c, d, e, _ = inverse @ values
        hessian = np.array([[2 * a, c], [c, 2 * b]])
        if np.max(np.linalg.eigvalsh(hessian)) >= -1e-6:
            continue
        offset = -np.linalg.solve(hessian, [d, e])
        if np.isfinite(offset).all() and np.max(np.abs(offset)) <= 1:
            points[i] = [x0, y0] + offset
    return points
