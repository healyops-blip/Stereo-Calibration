"""Round-trippable numeric artifacts and original-camera visualizations."""

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.calibration import check_model
from src.dataset import read_gray

CAMERA_KEYS = ("K_left", "D_left", "K_right", "D_right", "R_RL", "t_RL", "F")
ORIGIN = "per_frame_upper_row_left_corner; physical_origin_not_globally_identifiable"


def json_default(value: Any) -> Any:
    """将 NumPy 数值、数组或路径转换成 JSON 可序列化对象。

    Args:
        value: JSON 编码器无法直接处理的对象。

    Returns:
        数组对应列表、NumPy 标量对应 Python 标量、路径对应字符串。

    Raises:
        TypeError: 输入类型不在上述支持范围。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value: Any) -> None:
    """以 UTF-8 写入可审计 JSON，拒绝 NaN 和 Infinity。

    Args:
        path: 输出文件；自动创建父目录，存在时覆盖。
        value: JSON 对象，可包含 json_default 支持的 NumPy 对象及路径。

    Raises:
        ValueError: 对象含非有限 JSON 浮点值。
        TypeError: 包含不支持的类型。
        OSError: 建目录或写文件失败。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_model(path: Path) -> dict[str, Any]:
    """读取模型 JSON，将相机矩阵恢复为 float64 数组。

    Args:
        path: 包含 CAMERA_KEYS 和 size 的 JSON 文件。

    Returns:
        相机字段为 NumPy 数组、size 为元组的模型；其他字段原样保留。

    Note:
        只恢复数据类型，不执行 check_model 的几何有效性检查。
    """
    model = json.loads(path.read_text(encoding="utf-8"))
    for key in CAMERA_KEYS:
        model[key] = np.asarray(model[key], dtype=np.float64)
    model["size"] = tuple(model["size"])
    return model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """以所有行字段的有序并集写出带 UTF-8 BOM 的 CSV。

    Args:
        path: 目标路径；创建父目录，覆盖同名文件。
        rows: 非空记录列表；某行缺少的字段输出为空单元格。

    Raises:
        ValueError: rows 为空，拒绝生成没有表头的数据文件。
        OSError: 目录或文件写入失败。
    """
    if not rows:
        raise ValueError(f"Refusing to export empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_model(path: Path, model: dict[str, Any], board: dict[str, Any]) -> None:
    """将相机矩阵按行列索引展开为带单位及坐标约定的 CSV。

    Args:
        path: 目标 CSV，写入时覆盖。
        model: 标定结果，包含相机矩阵、基线、RMS、size 和 retained_ids。
        board: 棋盘 columns、rows、square_mm 配置。

    Note:
        t_RL 和格距为 mm，焦距及主点为 px；R_RL 表示左相机到右相机。
        内参矩阵的第三行是无量纲齐次项。
    """
    rows = []
    parameters = {key: model[key] for key in CAMERA_KEYS}
    parameters.update(
        baseline_mm=model["baseline_mm"],
        stereo_rms_px=model["stereo_rms_px"],
        image_width=model["size"][0],
        image_height=model["size"][1],
        retained_pair_count=len(model["retained_ids"]),
        **board,
    )
    for key, value in parameters.items():
        unit = "dimensionless"
        if key.startswith("K_") or key in ("image_width", "image_height", "stereo_rms_px"):
            unit = "px"
        if key in ("t_RL", "baseline_mm", "square_mm"):
            unit = "mm"
        for index, number in np.ndenumerate(np.atleast_2d(value)):
            rows.append(
                dict(
                    group="camera",
                    parameter=key,
                    row=index[0],
                    col=index[1],
                    value=float(number),
                    unit="dimensionless" if key.startswith("K_") and index[0] == 2 else unit,
                    convention="P_R=R_RL@P_L+t_RL; pinhole; D=k1,k2,p1,p2,k3",
                )
            )
    write_csv(path, rows)


def empty_pose_row(pair_id: str, reason: str) -> dict[str, Any]:
    """构造保留完整 CSV 字段的失败姿态记录。

    Args:
        pair_id: 图像对编号。
        reason: 失败原因。

    Returns:
        status 为 failed、数值列为空字符串的记录。
    """
    row = dict(
        pair_id=pair_id,
        status="failed",
        reference_frame="original_left_camera",
        board_origin_convention=ORIGIN,
        failure_reason=reason,
    )
    for key in (
        "rx_rad",
        "ry_rad",
        "rz_rad",
        "tx_mm",
        "ty_mm",
        "tz_mm",
        "left_rms_px",
        "right_rms_px",
        "joint_rms_px",
        "ambiguity_flag",
    ):
        row[key] = ""
    row.update({f"r{i + 1}{j + 1}": "" for i in range(3) for j in range(3)})
    return row


def pose_row(pair_id: str, fit: dict[str, Any]) -> dict[str, Any]:
    """把成功位姿拟合转换为 CSV 行及显式旋转矩阵。

    Args:
        pair_id: 图像对编号。
        fit: estimate 返回值；pose 为旋转向量 rad 加平移 mm。

    Returns:
        状态为 ok 的字段映射，包含 r11 至 r33 以及拟合指标。
        不导出 initial_pose 数组。
    """
    pose = fit["pose"]
    rotation = cv2.Rodrigues(pose[:3])[0]
    row = empty_pose_row(pair_id, "")
    row["status"] = "ok"
    row.update(zip(("rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm"), pose, strict=True))
    row.update({f"r{i + 1}{j + 1}": rotation[i, j] for i in range(3) for j in range(3)})
    row.update({key: value for key, value in fit.items() if key not in ("pose", "initial_pose")})
    return row


def draw_pose(
    obs: dict[str, Any], fit: dict[str, Any], model: dict[str, Any], output: Path, square_mm: float
) -> None:
    """在原始双目图像上绘制棋盘坐标轴并保存 PNG。

    Args:
        obs: 含 pair_id 和左右原图 paths 的观测记录。
        fit: 含棋盘到左相机 pose 的估计结果，单位 rad、mm。
        model: 相机模型，右图位姿由固定 R_RL、t_RL 推得。
        output: 创建并写入坐标轴图的目录，同名图会覆盖。
        square_mm: 格距 mm；坐标轴长度设为其三倍。

    Raises:
        OSError: 任一侧图片写入失败；已经写入的另一侧图片不回滚。
    """
    output.mkdir(parents=True, exist_ok=True)
    rotation = cv2.Rodrigues(fit["pose"][:3])[0]
    translation = fit["pose"][3:].reshape(3, 1)
    for side in ("left", "right"):
        r, t = rotation, translation
        if side == "right":
            r, t = model["R_RL"] @ r, model["R_RL"] @ t + model["t_RL"]
        image = cv2.cvtColor(read_gray(obs["paths"][side]), cv2.COLOR_GRAY2BGR)
        cv2.drawFrameAxes(
            image, model[f"K_{side}"], model[f"D_{side}"], cv2.Rodrigues(r)[0], t, 3 * square_mm, 2
        )
        path = output / f"{obs['pair_id']}_{side}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Cannot save pose overlay: {path}")


def read_csv(path: Path) -> list[dict[str, Any]]:
    """读取带或不带 BOM 的 UTF-8 CSV，保留字符串字段。

    Args:
        path: CSV 文件路径。

    Returns:
        按文件顺序排列的行字典；数值转换由调用方负责。
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def camera_from_csv(path: Path) -> dict[str, Any]:
    """仅由 CSV 单元格索引恢复相机模型并检查其有效性。

    Args:
        path: export_model 格式的相机参数 CSV。

    Returns:
        含矩阵、图像 size 及 board 的模型，不读取 calibration.json。

    Raises:
        ValueError: 索引重复、矩阵缺项、非有限参数或模型检查失败。
        KeyError: 缺少必需的参数或列。

    Note:
        矩阵形状由最大行列索引推得，物理量沿用 CSV 的 mm/px 约定。
    """
    cells: dict[str, dict[tuple[int, int], float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            indexed = cells.setdefault(row["parameter"], {})
            index = (int(row["row"]), int(row["col"]))
            if index in indexed:
                raise ValueError(f"Duplicate CSV cell: {row['parameter']}/{index}")
            indexed[index] = float(row["value"])
    model: dict[str, Any] = {}
    for key, indexed in cells.items():
        shape = (max(i[0] for i in indexed) + 1, max(i[1] for i in indexed) + 1)
        array = np.full(shape, np.nan)
        for index, value in indexed.items():
            array[index] = value
        if not np.isfinite(array).all():
            raise ValueError(f"Incomplete or nonfinite CSV parameter: {key}")
        model[key] = array
    model["size"] = (int(model["image_width"].item()), int(model["image_height"].item()))
    model["board"] = dict(
        columns=int(model["columns"].item()),
        rows=int(model["rows"].item()),
        square_mm=model["square_mm"].item(),
    )
    check_model(model)
    return model
