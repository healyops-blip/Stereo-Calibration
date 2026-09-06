"""Point-only regular-grid assembly; no image detector or reference corners."""

from typing import Any

import cv2
import numpy as np


def ideal_pixels(points: Any, camera: dict[str, Any] | None) -> Any:
    """生成用于网格几何判断的去畸变像素坐标。

    Args:
        points: 原始像素点数组 (N, 2)。
        camera: K、D；为 None 时仅复制输入。

    Returns:
        使用原内参 K 重新映射的像素坐标，不是归一化相机坐标。
    """
    if camera is None:
        return points.copy()
    # OpenCV 4 exposes Iter separately; OpenCV 5 accepts criteria on undistortPoints.
    undistort = getattr(cv2, "undistortPointsIter", cv2.undistortPoints)
    k = np.asarray(camera["K"], dtype=np.float64)
    return (
        undistort(
            points.reshape(-1, 1, 2).astype(np.float64),
            k,
            np.asarray(camera["D"]),
            R=None,
            P=k,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-10),
        )
        .reshape(-1, 2)
        .astype(np.float32)
    )


def orient_indices(indices: Any, pixels: Any, shape: tuple[int, int]) -> Any:
    """根据图像方向翻转网格索引，不移动候选坐标。

    Args:
        indices: 可重排为棋盘网格的候选索引。
        pixels: 候选原图像素坐标 (M, 2)。
        shape: 内角点 (columns, rows)。

    Returns:
        按上方行、向右列约定排列的一维索引副本。

    Note:
        此编号不是跨帧可辨识的物理角点身份。
    """
    grid = indices.reshape(shape[1], shape[0])
    if pixels[grid[0], 1].mean() > pixels[grid[-1], 1].mean():
        grid = grid[::-1]
    if pixels[grid[:, 0], 0].mean() > pixels[grid[:, -1], 0].mean():
        grid = grid[:, ::-1]
    return grid.ravel().copy()


def assemble_grid(
    candidates: Any, shape: tuple[int, int], camera: dict[str, Any] | None = None
) -> dict[str, Any]:
    """从候选点独立筛出完整规则网格，不借助 SB 参考或补造角点。

    Args:
        candidates: 原始图像候选坐标，可重排为 (M, 2)，px。
        shape: 已知内角点 (columns, rows)，每轴至少三个。
        camera: 用于几何判断的 K、D；为 None 时假设无需去畸变。

    Returns:
        有序原始坐标 corners、对应候选 indices、单应残差、间距及拒绝候选数。
        输出坐标不被拟合网格替换。

    Raises:
        ValueError: 点数不足、非有限、无法组成完整唯一网格，或单应残差超限。
        cv2.error: 底层点集网格或单应求解错误。

    Note:
        RMS/最大残差门限为中位相邻间距的 5%/15%。不检查黑白格纹理。
    """
    points = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
    required = shape[0] * shape[1]
    if min(shape) < 3 or len(points) < required or not np.isfinite(points).all():
        raise ValueError("invalid_or_insufficient_candidates")
    ideal = ideal_pixels(points, camera)
    # Explicit parameters select the overload where a null detector means input points.
    # OpenCV's runtime supports None here; its type stub omits that documented case.
    point_only_detector: Any = None
    for clustering in (False, True):
        flags = cv2.CALIB_CB_SYMMETRIC_GRID
        if clustering:
            flags |= cv2.CALIB_CB_CLUSTERING
        found, centers = cv2.findCirclesGrid(
            np.ascontiguousarray(ideal[:, None]),
            shape,
            flags=flags,
            blobDetector=point_only_detector,
            parameters=cv2.CirclesGridFinderParameters(),
        )
        if found:
            break
    if not found or centers is None or len(centers) != required:
        raise ValueError("complete_grid_not_found")
    distances = np.linalg.norm(centers.reshape(-1, 2)[:, None] - ideal[None], axis=2)
    indices = distances.argmin(axis=1)
    if len(set(indices)) != required or np.max(distances[np.arange(required), indices]) > 1e-3:
        raise ValueError("grid_must_select_unique_existing_candidates")
    indices = orient_indices(indices, points, shape)
    ordered = ideal[indices]
    grid_xy = np.mgrid[: shape[0], : shape[1]].T.reshape(-1, 2).astype(np.float32)
    h, _ = cv2.findHomography(grid_xy, ordered, method=0)
    if h is None or not np.isfinite(h).all():
        raise ValueError("invalid_grid_homography")
    projected = cv2.perspectiveTransform(grid_xy[:, None], h).reshape(-1, 2)
    residual = np.linalg.norm(projected - ordered, axis=1)
    grid = ordered.reshape(shape[1], shape[0], 2)
    spacing = float(
        np.median(
            np.r_[
                np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
            ]
        )
    )
    rms = float(np.sqrt(np.mean(residual**2)))
    if spacing <= 0 or rms > 0.05 * spacing or np.max(residual) > 0.15 * spacing:
        raise ValueError("grid_geometry_inconsistent")
    return dict(
        indices=indices,
        corners=points[indices].copy(),
        homography_rms_px=rms,
        homography_max_px=float(np.max(residual)),
        median_spacing_px=spacing,
        rejected_candidates=len(points) - required,
        clustering_fallback=clustering,
        reference_assisted=False,
    )
