"""数组约定：图像 H×W×C，尺寸 (W,H)，像素 (x,y)，彩色 BGR。

棋盘物点按行排列，长度 mm，旋转向量 rad，像素 px。
左相机为双目基准，P_right = R_RL @ P_left + t_RL；
五参数畸变顺序为 (k1, k2, p1, p2, k3)。
"""

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray


class CornerQuality(TypedDict):
    """棋盘包围框覆盖比例、中心 (x,y) px 和 Laplacian 方差。"""

    coverage: float
    center_x: float
    center_y: float
    sharpness: float


def require_gray(image: object, *, context: str) -> NDArray[np.uint8]:
    """验证非空 uint8 灰度 ndarray (H,W)，原样返回，不复制或修改输入。"""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 2
        or image.size == 0
        or image.dtype != np.uint8
    ):
        raise ValueError(
            f"{context}: expected nonempty uint8 grayscale (H, W), "
            f"got shape={getattr(image, 'shape', None)}, dtype={getattr(image, 'dtype', None)}"
        )
    return cast(NDArray[np.uint8], image)


def require_pattern(shape: tuple[int, int], *, minimum: int = 1) -> None:
    """检查内角点 (columns,rows) 为达到 minimum 的整数，拒绝布尔值。"""
    if len(shape) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
        for value in shape
    ):
        raise ValueError(f"Expected (columns, rows) integers >= {minimum}, got {shape}")
