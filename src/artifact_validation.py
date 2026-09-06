"""Read formal CSV artifacts back and verify their numerical consistency."""

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.calibration import check_model
from src.corners import object_points
from src.export import CAMERA_KEYS, read_csv, read_model, write_json
from src.pose import project, rms


def verify(output: Path, expected_ids: list[str]) -> dict[str, Any]:
    """回读正式 CSV，重算投影并核对冻结标定来源。

    Args:
        output: 含 calibration、pose 子目录的交付根目录。
        expected_ids: 位姿 CSV 应包含的完整有序编号列表。

    Returns:
        通过状态、成功/失败行数以及参数回读、重投影和哈希核对结果。

    Raises:
        ValueError: 编号、失败原因或冻结模型哈希不匹配。
        AssertionError: CSV 数值、旋转矩阵或回算残差与保存结果不一致。

    Note:
        成功时写入 quality/verification.json；不是只读检查，不提供外部精度。
    """
    model = read_model(output / "calibration/calibration.json")
    check_model(model)
    camera_rows = read_csv(output / "calibration/calibration_results.csv")
    reconstructed = dict(model)
    for key in CAMERA_KEYS:
        matrix = np.full_like(model[key], np.nan)
        for row in camera_rows:
            if row["parameter"] == key:
                matrix[int(row["row"]), int(row["col"])] = float(row["value"])
        np.testing.assert_allclose(matrix, model[key], rtol=1e-12, atol=1e-12)
        reconstructed[key] = matrix
    poses = read_csv(output / "pose/pose_results.csv")
    if [row["pair_id"] for row in poses] != expected_ids:
        raise ValueError("Pose CSV does not exactly match expected test IDs")
    saved = json.loads((output / "pose/pose_observations.json").read_text(encoding="utf-8"))
    observations = {o["pair_id"]: o for o in saved}
    checked = 0
    for row in poses:
        if row["status"] != "ok":
            if not row["failure_reason"]:
                raise ValueError("Failed pose row has no failure reason")
            continue
        pose = np.array(
            [float(row[key]) for key in ("rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm")]
        )
        rotation = np.array([[float(row[f"r{i + 1}{j + 1}"]) for j in range(3)] for i in range(3)])
        np.testing.assert_allclose(rotation, cv2.Rodrigues(pose[:3])[0], atol=1e-10)
        prediction = project(object_points(model["board"]), pose, reconstructed)
        for side, predicted in zip(("left", "right"), prediction, strict=True):
            observed = np.array(observations[row["pair_id"]][side]).reshape(-1, 2)
            np.testing.assert_allclose(
                rms(predicted - observed), float(row[f"{side}_rms_px"]), atol=1e-7
            )
        checked += 1
    provenance = json.loads((output / "pose/pose_provenance.json").read_text(encoding="utf-8"))
    if (
        provenance["calibration_sha256"]
        != hashlib.sha256((output / "calibration/calibration.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Pose artifacts refer to a different calibration")
    result = dict(
        status="passed",
        csv_rows=len(poses),
        successful_poses=checked,
        failed_poses=len(poses) - checked,
        camera_csv_roundtrip=True,
        pose_csv_reprojection=True,
        frozen_calibration_hash=True,
    )
    write_json(output / "quality/verification.json", result)
    return result
