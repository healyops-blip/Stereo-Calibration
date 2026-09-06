"""Inverse-ray checkerboard rendering in original distorted camera coordinates."""

from typing import Any

import cv2
import numpy as np

from src.corners import object_points


def scene_seed(split: str, index: int) -> int:
    """为不同合成划分分配不重叠的确定性随机种子。

    Args:
        split: train、val 或 test。
        index: 场景索引，范围 [0, 10000000)。

    Returns:
        划分专属偏移加索引的整数种子。

    Raises:
        ValueError: 划分未知或索引越界。
    """
    offsets = {"train": 100000000, "val": 200000000, "test": 300000000}
    if split not in offsets or not 0 <= index < 10000000:
        raise ValueError("Invalid split or scene index")
    return offsets[split] + index


def project_board(
    camera: dict[str, Any], board: dict[str, Any], rotation: Any, translation: Any
) -> Any:
    """在单相机中投影理想棋盘，拒绝非正深度的物点。

    Args:
        camera: 含 K (3, 3) 与 D 畸变系数的相机配置。
        board: 行列与 square_mm 配置。
        rotation: 棋盘到当前相机旋转矩阵 (3, 3)。
        translation: 棋盘到当前相机平移数组，三元素，mm。

    Returns:
        有序的畸变像素坐标 (N, 2)。

    Raises:
        ValueError: 至少一个角点深度不大于零。
    """
    points = object_points(board).astype(np.float64)
    if np.any((points @ rotation.T + translation.reshape(1, 3))[:, 2] <= 0):
        raise ValueError("Nonpositive board depth")
    return cv2.projectPoints(
        points,
        cv2.Rodrigues(rotation)[0],
        translation.astype(np.float64),
        np.asarray(camera["K"], dtype=np.float64),
        np.asarray(camera["D"], dtype=np.float64),
    )[0].reshape(-1, 2)


def render_crop(
    camera: dict[str, Any],
    board: dict[str, Any],
    rotation: Any,
    translation: Any,
    origin: tuple[int, int],
    size: int,
    supersample: int = 2,
    colors: tuple[float, float, float] = (25.0, 225.0, 125.0),
) -> Any:
    """按像素面积积分的逆射线近似渲染原始畸变坐标下的棋盘裁块。

    Args:
        camera: 当前相机的 K 和 D。
        board: 内角点行列数及 square_mm。
        rotation: 棋盘到相机旋转 (3, 3)。
        translation: 棋盘到相机平移 (3,)，mm。
        origin: 裁块左上角的原图 (x, y)，px。
        size: 方形裁块边长，正整数 px。
        supersample: 每轴子采样数，须为正整数。
        colors: 黑格、白格与背景的灰度值，最终裁剪到 [0, 255]。

    Returns:
        uint8 灰度图 (size, size)，不包含额外模糊和噪声。

    Note:
        OpenCV 像素中心为整数坐标；场景真值仅相对于理想平面渲染模型。
    """
    grid = (np.arange(size * supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    u, v = np.meshgrid(grid + origin[0], grid + origin[1])
    pixels = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    # OpenCV 4 exposes Iter separately; OpenCV 5 accepts criteria on undistortPoints.
    undistort = getattr(cv2, "undistortPointsIter", cv2.undistortPoints)
    normalized = undistort(
        pixels,
        np.asarray(camera["K"], dtype=np.float64),
        np.asarray(camera["D"], dtype=np.float64),
        R=None,
        P=None,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-10),
    ).reshape(-1, 2)
    rays = np.c_[normalized, np.ones(len(normalized))]
    # P_camera = R P_board + t; plane normal is R[:, 2].
    normal = rotation[:, 2]
    denominator = rays @ normal
    valid = np.abs(denominator) > 1e-10
    depth = np.divide(normal @ translation, denominator, out=np.zeros(len(rays)), where=valid)
    board_xyz = (rays * depth[:, None] - translation) @ rotation
    x, y = board_xyz[:, 0] / board["square_mm"], board_xyz[:, 1] / board["square_mm"]
    inside = valid & (depth > 0) & (x >= -1) & (x < board["columns"])
    inside &= (y >= -1) & (y < board["rows"])
    dark = (np.floor(x).astype(int) + np.floor(y).astype(int)) % 2 == 0
    values = np.full(len(rays), colors[2])
    values[inside & dark], values[inside & ~dark] = colors[0], colors[1]
    # A one-square white border outside the black/white cells.
    border = valid & (depth > 0) & ~inside & (x >= -1.5) & (x <= board["columns"] + 0.5)
    border &= (y >= -1.5) & (y <= board["rows"] + 0.5)
    values[border] = colors[1]
    image = values.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def heatmap(corners: Any, size: int, sigma: float = 1.5) -> Any:
    """在浮点角点中心生成高斯标签，重叠处取最大响应。

    Args:
        corners: 裁块坐标 (N, 2)，px，允许点位于裁块外。
        size: 输出方形热图边长，px。
        sigma: 正高斯标准差，px；截取约四倍标准差的局部支持。

    Returns:
        float32 热图 (size, size)，空角点返回全零图。
    """
    target = np.zeros((size, size), dtype=np.float32)
    radius = int(np.ceil(4 * sigma))
    for x, y in corners:
        x0, x1 = max(0, int(x) - radius), min(size, int(x) + radius + 2)
        y0, y1 = max(0, int(y) - radius), min(size, int(y) + radius + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], gaussian)
    return target


def make_scene(model: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    """采样左右相机共享的物理棋盘姿态和连续投影标签。

    Args:
        model: A-only 相机模型，含 board、size、generation_depth_mm 和双目外参。
        split: train、val 或 test。
        index: 该划分中的合法场景索引。

    Returns:
        场景索引、种子及左右 R、t、corners；旋转矩阵无量纲，平移 mm，角点 px。

    Raises:
        ValueError: 种子参数无效或尝试 300 次仍未得到双目可见且间距足够的棋盘。

    Note:
        左右姿态由固定双目外参关联，不分别随机采样。
    """
    rng = np.random.default_rng(scene_seed(split, index))
    width, height = model["size"]
    depth_range = model["generation_depth_mm"]
    for _ in range(300):
        angles = np.deg2rad(rng.uniform([-40, -40, -180], [40, 40, 180]))
        rotations = [cv2.Rodrigues(np.eye(3)[i] * angles[i])[0] for i in range(3)]
        rotation = rotations[2] @ rotations[1] @ rotations[0]
        depth = rng.uniform(*depth_range)
        center = object_points(model["board"]).mean(axis=0)
        k = np.asarray(model["K_left"])
        uv = rng.uniform([width * 0.25, height * 0.25], [width * 0.75, height * 0.75])
        translation = (
            np.array(
                [(uv[0] - k[0, 2]) * depth / k[0, 0], (uv[1] - k[1, 2]) * depth / k[1, 1], depth]
            )
            - rotation @ center
        )
        poses = {"left": (rotation, translation)}
        rr = np.asarray(model["R_RL"])
        poses["right"] = (rr @ rotation, rr @ translation + np.asarray(model["t_RL"]).ravel())
        scene: dict[str, Any] = dict(index=index, split=split, seed=scene_seed(split, index))
        valid = True
        for side, (r, t) in poses.items():
            camera = dict(K=model[f"K_{side}"], D=model[f"D_{side}"])
            try:
                points = project_board(camera, model["board"], r, t)
            except ValueError:
                valid = False
                break
            grid = points.reshape(model["board"]["rows"], model["board"]["columns"], 2)
            spacing = min(np.linalg.norm(np.diff(grid, axis=d), axis=2).min() for d in (0, 1))
            valid &= bool(
                np.all(points >= 12) and np.all(points < [width - 12, height - 12]) and spacing >= 8
            )
            scene[side] = dict(R=r.tolist(), t=t.tolist(), corners=points.tolist())
        if valid:
            return scene
    raise ValueError("Could not sample a visible board from training-only camera model")
