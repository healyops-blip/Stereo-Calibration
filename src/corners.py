"""Detect complete boards and retain auditable per-frame corner numbering."""

import hashlib
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.contracts import CornerQuality, require_gray, require_pattern
from src.dataset import pair_paths, read_gray

SUBPIX_HALF_WINDOW_PX = 5
SUBPIX_MAX_ITERATIONS = 100
SUBPIX_EPSILON_PX = 1e-4


def object_points(board: dict[str, Any]) -> NDArray[np.float32]:
    """生成按行排列的理想平面棋盘物点，Z=0。

    Args:
        board: 内角点 rows、columns 与格距 square_mm。

    Returns:
        (rows * columns, 3) 的 float32 理想物点，单位 mm。

    Raises:
        ValueError: 行列数非正整数或 square_mm 非有限正数。
    """
    require_pattern((board["columns"], board["rows"]))
    spacing = float(board["square_mm"])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"Expected positive finite square_mm, got {spacing}")
    points = np.zeros((board["rows"] * board["columns"], 3), np.float32)
    points[:, :2] = np.mgrid[: board["columns"], : board["rows"]].T.reshape(-1, 2)
    return points * board["square_mm"]


def detect(
    image: NDArray[np.uint8], shape: tuple[int, int], detector: str = "auto"
) -> tuple[NDArray[np.float32], str]:
    """检测并排列完整棋盘的亚像素角点。

    Args:
        image: uint8 灰度图 (H, W)。
        shape: 内角点列数、行数，顺序不可交换。
        detector: auto 优先 SB、失败后回退 classic；显式指定方法不回退。

    Returns:
        np.ndarray (N, 1, 2)，float32，按 (x,y) 存储的角点 px，与实际检测器名称。

    Raises:
        ValueError: 图像非 uint8 灰度、规格小于 3×3、检测器未知或棋盘不完整。
        cv2.error: 图像或棋盘规格不满足 OpenCV 接口要求。
    """
    require_gray(image, context="detect")
    require_pattern(shape, minimum=3)
    if detector not in ("auto", "sb", "classic"):
        raise ValueError(f"Unknown detector: {detector}")
    ok, corners, method = False, None, "SB"

    # SB already returns subpixel corner locations, so no cornerSubPix pass is required.
    if detector in ("auto", "sb"):
        ok, corners = cv2.findChessboardCornersSB(image, shape, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not ok and detector in ("auto", "classic"):
        ok, corners = cv2.findChessboardCorners(
            image, shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        method = "classic_subpix"
        if ok:
            corners = cv2.cornerSubPix(
                image,
                corners,
                (SUBPIX_HALF_WINDOW_PX, SUBPIX_HALF_WINDOW_PX),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    SUBPIX_MAX_ITERATIONS,
                    SUBPIX_EPSILON_PX,
                ),
            )
    if not ok or corners is None:
        raise ValueError("corners_not_found")
    return order_corners(cast(NDArray[np.float32], corners), shape), method


def order_corners(
    corners: NDArray[np.float32] | NDArray[np.float64], shape: tuple[int, int]
) -> NDArray[np.float32]:
    """统一图像角点编号；对称棋盘的编号不代表跨帧固定的物理原点。

    Args:
        corners: 可重排为 (rows, columns, 2) 的已检测角点，px。
        shape: 内角点 (columns, rows)。

    Returns:
        C 连续的 float32 数组 (N, 1, 2)，按图像上方行及向右列编号。

    Raises:
        ValueError: 角点数量不匹配、坐标非有限或棋盘行列数不合法。
    """
    require_pattern(shape)
    if corners.size != shape[0] * shape[1] * 2 or not np.isfinite(corners).all():
        raise ValueError(
            f"Invalid corners: shape={corners.shape}, expected {shape[0] * shape[1]} xy pairs"
        )
    grid = corners.reshape(shape[1], shape[0], 2)
    if grid[0, :, 1].mean() > grid[-1, :, 1].mean():
        grid = grid[::-1]
    if grid[:, 0, 0].mean() > grid[:, -1, 0].mean():
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid.reshape(-1, 1, 2), dtype=np.float32)


def corner_quality(image: NDArray[np.uint8], corners: NDArray[np.float32]) -> CornerQuality:
    """计算棋盘覆盖率、中心位置和局部清晰度。

    Args:
        image: np.ndarray uint8 灰度图 (H,W)，须已通过读取/检测校验。
        corners: np.ndarray float32 有效角点 (N,1,2)，每点 (x,y)，px。

    Returns:
        包围框面积比、角点平均中心 px、包围框内 Laplacian 方差。
    """

    x, y, w, h = cv2.boundingRect(corners)
    return dict(
        coverage=w * h / image.size,
        center_x=float(corners[:, 0, 0].mean()),
        center_y=float(corners[:, 0, 1].mean()),
        sharpness=float(cv2.Laplacian(image[y : y + h, x : x + w], cv2.CV_64F).var()),
    )


def draw_corners(image: Any, corners: Any, shape: tuple[int, int], path: Path) -> None:
    """保存含检测网格与每个角点编号的原图叠加。

    Args:
        image: 只读灰度图 (H, W)。
        corners: 有序角点 (N, 1, 2)，px。
        shape: 内角点 (columns, rows)。
        path: 输出图路径；调用方创建父目录，同名图会覆盖。

    Raises:
        OSError: imwrite 报告写入失败。
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(canvas, shape, corners, True)
    for index, point in enumerate(corners[:, 0]):
        cv2.putText(
            canvas,
            str(index),
            tuple(point.astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (0, 0, 255),
            1,
        )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Cannot save corner overlay: {path}")


def collect(
    folder: Path, board: dict[str, Any], output: Path, excluded: list[str], detector: str = "auto"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """收集双侧完整、分辨率一致的棋盘观测，并记录每对图像的审计结果。

    Args:
        folder: 含 left/right BMP 的数据目录。
        board: 棋盘行列数和 square_mm。
        output: 角点叠加图目录，自动创建；同名输出覆盖。
        excluded: 预先排除的图像对编号。
        detector: auto、sb 或 classic。

    Returns:
        成功观测列表和含失败行的 manifest；角点单位 px，原图只读。

    Note:
        捕获逐对 ValueError/cv2.error 并记录拒绝原因；文件 IO 错误向外传播。
        被提前拒绝的行可能没有所有图像哈希和质量字段。
    """
    output.mkdir(parents=True, exist_ok=True)
    observations, manifest = [], []
    shape = (board["columns"], board["rows"])
    size = None
    seen: dict[str, str] = {}
    for pair_id, paths in pair_paths(folder).items():
        row: dict[str, Any] = {"dataset": folder.name, "pair_id": pair_id, "status": "ok"}
        obs: dict[str, Any] = {"pair_id": pair_id, "paths": paths}
        try:
            if set(paths) != {"left", "right"}:
                raise ValueError("missing_side")
            if pair_id in excluded:
                raise ValueError("suspected_image_discontinuity")
            for side, path in paths.items():
                image = read_gray(path)
                current_size = (image.shape[1], image.shape[0])
                if size is not None and current_size != size:
                    raise ValueError("inconsistent_image_size")
                size = current_size

                # 仅记录完全重复图像的来源，不在此处剔除。
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                row[f"{side}_sha256"] = digest
                row[f"{side}_duplicate_of"] = seen.get(digest, "")
                seen.setdefault(digest, f"{pair_id}/{side}")
                corners, method = detect(image, shape, detector)
                obs[side] = corners
                obs["size"] = size
                row[f"{side}_detector"] = method
                row[f"{side}_corner_count"] = len(corners)
                row.update(
                    {
                        f"{side}_{key}": value
                        for key, value in corner_quality(image, corners).items()
                    }
                )
                draw_corners(image, corners, shape, output / f"{pair_id}_{side}.png")
            observations.append(obs)
        except (ValueError, cv2.error) as error:
            row["status"] = "rejected"
            row["failure_reason"] = str(error)
        manifest.append(row)
        print(f"{folder.name}/{pair_id}: {row['status']}", flush=True)
    return observations, manifest
