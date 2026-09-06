"""Pair images by explicit identifiers; never depend on filesystem order."""

import re
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def pair_paths(folder: Path) -> dict[str, dict[str, Path]]:
    """按六位编号配对左右 BMP，不依赖文件系统遍历顺序。

    Args:
        folder: 含 left、right 子目录的数据集目录。

    Returns:
        按编号排序的映射；每项保存已找到的左右路径，缺侧不在此处拒绝。

    Raises:
        ValueError: BMP 文件名不符合 side_pair_000001.bmp 约定，或没有找到图像。
    """
    pairs: dict[str, dict[str, Path]] = {}
    for side in ("left", "right"):
        for path in sorted((folder / side).glob("*.bmp")):
            match = re.fullmatch(rf"{side}_pair_(\d{{6}})\.bmp", path.name)
            if match is None:
                raise ValueError(f"Unexpected filename: {path}")
            pair = pairs.setdefault(match[1], {})
            pair[side] = path
    if not pairs:
        raise ValueError(f"No BMP image pairs in {folder}")
    return dict(sorted(pairs.items()))


def read_gray(path: Path) -> NDArray[np.uint8]:
    """从路径解码单通道灰度图，支持非 ASCII 路径。

    Args:
        path: 原始图像路径。

    Returns:
        形状 (H, W) 的 uint8 图像。

    Raises:
        ValueError: 解码未返回有效图像。
        OSError: 文件无法读取。
        cv2.error: 解码器拒绝输入缓冲区。
    """
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cast(NDArray[np.uint8], image)
