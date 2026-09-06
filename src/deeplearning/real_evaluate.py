"""Frozen real-B evaluation, with SB-assisted correspondence explicitly separated."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.corners import detect, object_points
from src.dataset import pair_paths, read_gray
from src.deeplearning.real_metrics import match_reference, symmetric_epipolar
from src.deeplearning.subpixel import refine
from src.deeplearning.train import corner_network
from src.export import read_model, write_json
from src.pose import estimate


def dense_response(network: Any, image: Any) -> Any:
    """不缩放原图，用带上下文的分块 CUDA 推理生成全幅响应。

    Args:
        network: 已加载权重、位于 CUDA 且设为 eval 的角点网络。
        image: 灰度图 (H, W)，值域 [0, 255]。

    Returns:
        float32 概率响应 (H, W)。每块核心为 236 px，上下文为 10 px。

    Note:
        不接收棋盘参考位置，不改变模型训练模式；禁用梯度但需要 CUDA。
    """
    height, width = image.shape
    response = np.zeros((height, width), dtype=np.float32)
    halo, core = 10, 236
    with torch.inference_mode():
        for y in range(0, height, core):
            for x in range(0, width, core):
                y0, x0 = max(0, y - halo), max(0, x - halo)
                y1, x1 = min(height, y + core + halo), min(width, x + core + halo)
                crop = np.ascontiguousarray(image[y0:y1, x0:x1])
                logits = network(torch.from_numpy(crop[None, None]).cuda().float() / 255)
                patch = logits.sigmoid()[0, 0].cpu().numpy()
                out_y, out_x = min(y + core, height), min(x + core, width)
                response[y:out_y, x:out_x] = patch[y - y0 : out_y - y0, x - x0 : out_x - x0]
    return response


def raw_candidates(network: Any, image: Any) -> tuple[Any, Any]:
    """对全幅网络响应执行固定门限与 7×7 非极大抑制。

    Args:
        network: CUDA eval 网络。
        image: 灰度图 (H, W)。

    Returns:
        float32 整数峰值坐标 (N, 2) 与完整响应 (H, W)。阈值 0.5，
        忽略外缘 10 px；未执行亚像素细化或棋盘排序。
    """
    response = dense_response(network, image)
    peaks = (response == cv2.dilate(response, np.ones((7, 7), np.uint8))) & (response >= 0.5)
    peaks[:10] = peaks[-10:] = False
    peaks[:, :10] = peaks[:, -10:] = False
    y, x = np.nonzero(peaks)
    raw = np.ascontiguousarray(np.stack([x, y], axis=1), dtype=np.float32)
    return raw, response


def neural_candidates(network: Any, image: Any) -> tuple[Any, Any, Any]:
    """提取网络峰值并应用固定 win3/Q0 亚像素细化。

    Args:
        network: CUDA eval 网络。
        image: 原始灰度图 (H, W)。

    Returns:
        原始峰值、细化坐标（均 (N, 2)，px）与全幅响应。
        返回候选不保证都属于棋盘。
    """
    raw, response = raw_candidates(network, image)
    refined = refine(image, raw, response, "win3")
    return raw, refined, response


def image_result(network: Any, image: Any, shape: tuple[int, int], path: Path) -> Any:
    """比较单图网络候选、SB 与 classic 并写入全幅诊断图。

    Args:
        network: CUDA eval 网络。
        image: 原始灰度图 (H, W)。
        shape: 目标棋盘内角点 (columns, rows)。
        path: 叠加图路径；另写同目录的 _heatmap.png，父目录须存在。

    Returns:
        检测点、失败原因、候选数与 SB 一对一参考匹配结果的记录。

    Raises:
        OSError: 叠加图或响应图无法写入。

    Note:
        SB 是评分参考而非真值；捕获传统检测失败，但网络推理错误仍向外传播。
    """
    raw, refined, response = neural_candidates(network, image)
    record: dict[str, Any] = dict(network_candidates=len(raw), raw=raw, refined=refined)
    for method in ("sb", "classic"):
        try:
            record[method] = detect(image, shape, method)[0].reshape(-1, 2)
        except (ValueError, cv2.error) as error:
            record[method] = None
            record[f"{method}_error"] = str(error)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for point in refined:
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, (255, 0, 255), 1)
    if record["sb"] is not None:
        reference = record["sb"]
        indices, distances = match_reference(reference, refined)
        record.update(
            reference_indices=indices,
            matched=int(np.sum(indices >= 0)),
            disagreement_px=distances,
            complete_reference_coverage=bool(np.all(indices >= 0)),
        )
        for point, index in zip(reference, indices, strict=True):
            color = (0, 255, 0) if index >= 0 else (0, 0, 255)
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 5, color, 1)
        if record["classic"] is not None:
            record["classic_vs_sb_px"] = np.linalg.norm(record["classic"] - reference, axis=1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Cannot write {path}")
    if not cv2.imwrite(
        str(path.with_name(path.stem + "_heatmap.png")), np.rint(response * 255).astype(np.uint8)
    ):
        raise OSError("Cannot save heatmap")
    return record


def geometry(records: dict[str, Any], model: dict[str, Any]) -> Any:
    """在左右共同的 SB 参考对应点上比较四种检测结果的几何指标。

    Args:
        records: left、right 的 image_result 结果。
        model: 含 board 和相机内外参的固定模型。

    Returns:
        共同点数及各方法的姿态和极线指标；无参考、少于六点或失败时记录状态。

    Note:
        raw/refined 沿用细化点匹配的参考编号，不是独立网格识别评估。
    """
    if any(records[side]["sb"] is None for side in ("left", "right")):
        return dict(status="no_SB_reference")
    left, right = records["left"], records["right"]
    mask = (left["reference_indices"] >= 0) & (right["reference_indices"] >= 0)
    count = int(np.sum(mask))
    result: dict[str, Any] = dict(common_corners=count, correspondence="SB_assisted_1px_gate")
    if count < 6:
        result["status"] = "insufficient_common_corners"
        return result
    points = object_points(model["board"])[mask]
    for method in ("sb", "classic", "raw", "refined"):
        if any(records[s][method] is None for s in ("left", "right")):
            result[method] = dict(status="detector_failed")
            continue
        obs = {}
        for side in ("left", "right"):
            row = records[side]
            indices = row["reference_indices"][mask] if method in ("raw", "refined") else mask
            obs[side] = np.ascontiguousarray(
                row[method][indices].reshape(-1, 1, 2), dtype=np.float32
            )
        try:
            fit = estimate(points, obs, model)
            fit["epipolar_mean_px"] = float(
                np.mean(symmetric_epipolar(obs["left"], obs["right"], model))
            )
            result[method] = fit
        except (ValueError, cv2.error) as error:
            result[method] = dict(status="geometry_failed", error=str(error))
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总真实 B 检测覆盖及与 SB 的坐标差异。

    Args:
        records: 每对包含左右 images 字段的评估记录。

    Returns:
        成功图像数、候选数、匹配数与成功匹配差异的均值/P95，单位 px。
        没有匹配差异时对应统计为 None；不汇总位姿误差。
    """
    images = [pair["images"][side] for pair in records for side in ("left", "right")]
    valid = [row for row in images if row["sb"] is not None]
    differences = (
        np.concatenate([row["disagreement_px"] for row in valid]) if valid else np.empty(0)
    )
    summary = dict(
        pairs=len(records),
        images=len(images),
        sb_complete_images=len(valid),
        classic_complete_images=sum(row["classic"] is not None for row in images),
        network_complete_SB_coverage_images=sum(
            row.get("complete_reference_coverage", False) for row in images
        ),
        SB_reference_corners=sum(len(row["sb"]) for row in valid),
        network_matched_SB_corners=sum(row["matched"] for row in valid),
        network_candidates=sum(row["network_candidates"] for row in images),
        matched_disagreement_mean_px=float(np.mean(differences)) if len(differences) else None,
        matched_disagreement_p95_px=float(np.percentile(differences, 95))
        if len(differences)
        else None,
    )
    return summary


def main() -> None:
    """执行历史 SB 辅助编号的 B 集对照，不重新标定相机或训练网络。

    CLI 要求 prepared、run、images、output。核对冻结 B 编号及原图哈希，
    创建新输出目录，保存协议、逐对 JSON、叠加图和汇总；CUDA 为必需。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    split = json.loads((args.prepared / "split.json").read_text())
    model = read_model(args.prepared / "theta_A.json")
    paths = pair_paths(args.images)
    if set(paths) != set(split["B"]) or set(split["A"]) & set(split["B"]):
        raise ValueError("Only the frozen B dataset is allowed")
    reference_hashes = {row["pair_id"]: row for row in split["manifest"]}
    for pair_id, sides in paths.items():
        if set(sides) != {"left", "right"}:
            raise ValueError("Missing stereo side")
        for side, path in sides.items():
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != reference_hashes[pair_id][f"{side}_sha256"]
            ):
                raise ValueError("Real image hash mismatch")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260905)
    torch.set_num_threads(2)
    network = corner_network().cuda()
    weight = args.run / "best.pt"
    network.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True)["model"])
    network.eval()
    write_json(
        args.output / "protocol.json",
        dict(
            split="B",
            ids=split["B"],
            threshold=0.5,
            matching_radius_px=1,
            camera_sha256=hashlib.sha256((args.prepared / "theta_A.json").read_bytes()).hexdigest(),
            weights_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
            opencv=cv2.__version__,
            torch=str(torch.__version__),
            caveat=(
                "SB is a correspondence reference, not ground truth. "
                "This historical comparison does not use independent grid assembly."
            ),
        ),
    )
    results = []
    for pair_id, sides in paths.items():
        start = time.time()
        records = {}
        for side, path in sides.items():
            image = read_gray(path)
            if [image.shape[1], image.shape[0]] != list(model["size"]):
                raise ValueError("Unexpected resolution")
            records[side] = image_result(
                network,
                image,
                (model["board"]["columns"], model["board"]["rows"]),
                args.output / f"{pair_id}_{side}.png",
            )
        row = dict(
            pair_id=pair_id,
            images=records,
            geometry=geometry(records, model),
            elapsed_seconds=time.time() - start,
        )
        results.append(row)
        write_json(args.output / f"{pair_id}.json", row)
        print(f"{pair_id}: common={row['geometry'].get('common_corners', 0)}", flush=True)
    summary = summarize(results)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
