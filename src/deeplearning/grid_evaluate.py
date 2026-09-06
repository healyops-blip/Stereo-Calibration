"""Evaluate point-only assembly on frozen cached real-B network predictions."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.corners import object_points
from src.dataset import pair_paths, read_gray
from src.deeplearning.grid import assemble_grid
from src.deeplearning.real_metrics import symmetric_epipolar
from src.export import read_model, write_json
from src.pose import estimate


def assemble_all(cache: Path, model: dict[str, Any], ids: list[str], output: Path) -> Any:
    """只用缓存的 refined 候选独立组装棋盘，并在评分前持久化预测。

    Args:
        cache: 含逐对 images/left/right/refined JSON 的缓存目录。
        model: 相机模型与 board 配置。
        ids: 要处理的有序图像对编号。
        output: 写 independent_predictions.json 的目录。

    Returns:
        每对每侧的网格预测或拒绝原因；不将 SB 坐标传入组装器。
    """
    predictions: dict[str, Any] = {}
    for pair_id in ids:
        cached = json.loads((cache / f"{pair_id}.json").read_text())
        predictions[pair_id] = {}
        for side in ("left", "right"):
            candidates = np.asarray(cached["images"][side]["refined"], dtype=np.float32)
            camera = dict(K=model[f"K_{side}"], D=model[f"D_{side}"])
            try:
                result = assemble_grid(
                    candidates, (model["board"]["columns"], model["board"]["rows"]), camera
                )
                result["status"] = "complete"
            except ValueError as error:
                result = dict(status="rejected", reason=str(error))
            predictions[pair_id][side] = result
    write_json(output / "independent_predictions.json", predictions)
    return predictions


def score_all(predictions: Any, cache: Path, model: Any, paths: Any, output: Path) -> Any:
    """对已经冻结的独立网格按编号评分并绘图。

    Args:
        predictions: assemble_all 返回的独立预测。
        cache: 含 SB 参考的逐对 JSON 目录。
        model: 固定相机模型与棋盘配置。
        paths: 图像对编号到左右原图路径的映射。
        output: 写入 _ordered.png 的目录，须存在。

    Returns:
        各图参考坐标差异与共同成功双目图的位姿指标列表。

    Raises:
        OSError: 叠加图写入失败。

    Note:
        不根据参考重新排列预测；参考缺失时保留未评分状态。
    """
    scores = []
    for pair_id, sides in predictions.items():
        cached = json.loads((cache / f"{pair_id}.json").read_text())
        row: dict[str, Any] = dict(pair_id=pair_id, images={})
        for side, prediction in sides.items():
            result: dict[str, Any] = dict(status=prediction["status"])
            if prediction["status"] == "complete":
                corners = prediction["corners"]
                reference = cached["images"][side]["sb"]
                if reference is not None:
                    error = np.linalg.norm(corners - np.asarray(reference), axis=1)
                    result.update(
                        ordered_agreement_1px=bool(np.all(error <= 1)),
                        ordered_disagreement_mean_px=float(error.mean()),
                    )
                result["discarded_candidates"] = prediction["rejected_candidates"]
                canvas = cv2.cvtColor(read_gray(paths[pair_id][side]), cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(
                    canvas,
                    (model["board"]["columns"], model["board"]["rows"]),
                    corners.reshape(-1, 1, 2),
                    True,
                )
                for index, point in enumerate(corners):
                    cv2.putText(
                        canvas,
                        str(index),
                        tuple(np.rint(point).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 255),
                        1,
                    )
                if not cv2.imwrite(str(output / f"{pair_id}_{side}_ordered.png"), canvas):
                    raise OSError("Could not write overlay")
            row["images"][side] = result
        if all(sides[s]["status"] == "complete" for s in ("left", "right")):
            obs = {s: sides[s]["corners"].reshape(-1, 1, 2) for s in ("left", "right")}
            try:
                fit = estimate(object_points(model["board"]), obs, model)
                fit["epipolar_mean_px"] = float(
                    np.mean(symmetric_epipolar(obs["left"], obs["right"], model))
                )
                row["independent_geometry"] = fit
            except (ValueError, cv2.error) as error:
                row["independent_geometry"] = dict(error=str(error))
        scores.append(row)
    return scores


def main() -> None:
    """核对冻结 B 缓存来源，依次执行独立排序和参考评分。

    CLI 指定 prepared、cache、images、output；创建新目录并写入
    independent_predictions、scores、summary JSON 及编号叠加图，无需 torch。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = read_model(args.prepared / "theta_A.json")
    split = json.loads((args.prepared / "split.json").read_text())
    protocol = json.loads((args.cache / "protocol.json").read_text())
    digest = hashlib.sha256((args.prepared / "theta_A.json").read_bytes()).hexdigest()
    if protocol["camera_sha256"] != digest or protocol["ids"] != split["B"]:
        raise ValueError("Frozen cache mismatch")
    paths = pair_paths(args.images)
    if not set(split["B"]).issubset(paths):
        raise ValueError("Missing B images")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260905)
    predictions = assemble_all(args.cache, model, split["B"], args.output)
    scores = score_all(predictions, args.cache, model, paths, args.output)
    images = [row["images"][s] for row in scores for s in ("left", "right")]
    summary = dict(
        images=len(images),
        complete_images=sum(r["status"] == "complete" for r in images),
        ordered_agreement_images=sum(r.get("ordered_agreement_1px", False) for r in images),
        discarded_candidates=sum(r.get("discarded_candidates", 0) for r in images),
        reference_used_for_assembly=False,
        opencv=cv2.__version__,
        camera_sha256=digest,
        frozen_detection_protocol=protocol,
        limitation=(
            "B is development validation; canonical image orientation is not physical identity."
        ),
    )
    write_json(args.output / "scores.json", scores)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
