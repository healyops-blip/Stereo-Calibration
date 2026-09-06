"""Coordinate-explicit pose export and failure-aware paired summaries."""

from typing import Any

import cv2
import numpy as np


def pose_transforms(pose: Any, model: Any) -> Any:
    """将六维位姿转换为棋盘到左右相机的齐次变换。

    Args:
        pose: 棋盘到左相机的旋转向量 rad 加平移 mm，形状 (6,)。
        model: 含固定左到右 R_RL、t_RL 的模型。

    Returns:
        T_board_to_left、T_board_to_right 两个 (4, 4) 数组及单位/原点约定。
    """
    left = np.eye(4)
    left[:3, :3] = cv2.Rodrigues(np.asarray(pose[:3], dtype=np.float64))[0]
    left[:3, 3] = pose[3:]
    right_from_left = np.eye(4)
    right_from_left[:3, :3] = model["R_RL"]
    right_from_left[:3, 3] = np.asarray(model["t_RL"]).ravel()
    return dict(
        T_board_to_left=left,
        T_board_to_right=right_from_left @ left,
        translation_unit="mm",
        rotation_vector_unit="radian",
        board_origin="per_frame_image_upper_row_left_corner_not_global_identity",
    )


def summarize_holdout(rows: list[dict[str, Any]]) -> Any:
    """在 log3 和 SB 共同成功的图像对上汇总位姿指标，保留失败信息。

    Args:
        rows: 每对含 methods/log3、methods/sb 结果的记录。

    Returns:
        总对数、共同编号、各方法成功/失败/歧义编号及逐对指标算术均值。
        没有可用指标时均值为 None；不是汇集全部角点残差后的 RMS。
    """
    methods = ("log3", "sb")
    common = [r["pair_id"] for r in rows if all("pose" in r["methods"][m] for m in methods)]
    summary: dict[str, Any] = dict(total_pairs=len(rows), common_pair_ids=common, methods={})
    for method in methods:
        valid = [r for r in rows if "pose" in r["methods"][method]]
        matched = [r for r in valid if r["pair_id"] in common]
        result: dict[str, Any] = dict(
            successful_pairs=len(valid),
            failed_pair_ids=[r["pair_id"] for r in rows if "pose" not in r["methods"][method]],
        )
        for key in ("joint_rms_px", "p0_right_prediction_rms_px", "epipolar_mean_px"):
            values = [
                r["methods"][method]["pose"][key]
                for r in matched
                if key in r["methods"][method]["pose"]
            ]
            result[f"common_{key.removesuffix('_px')}_mean_px"] = (
                float(np.mean(values)) if values else None
            )
        result["ambiguous_pairs"] = [
            r["pair_id"] for r in valid if r["methods"][method]["pose"].get("ambiguity_flag", False)
        ]
        summary["methods"][method] = result
    return summary
