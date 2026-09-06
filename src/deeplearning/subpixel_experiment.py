"""Select using synthetic validation + A safety checks, then freeze before B."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.corners import object_points
from src.dataset import pair_paths, read_gray
from src.deeplearning.grid import assemble_grid
from src.deeplearning.real_evaluate import dense_response
from src.deeplearning.real_metrics import match_reference, symmetric_epipolar
from src.deeplearning.subpixel import METHODS, refine
from src.deeplearning.train import corner_network, render_sample
from src.export import read_model, write_json
from src.pose import estimate


def peaks_from_response(response: Any) -> Any:
    """以固定阈值 0.5 和 7×7 NMS 提取概率峰，忽略十像素边界。

    Args:
        response: 网络概率热图 (H, W)。

    Returns:
        float32 峰值坐标 (N, 2)，单位 px；不细化、不排序。
    """
    peaks = (response == cv2.dilate(response, np.ones((7, 7), np.uint8))) & (response >= 0.5)
    peaks[:10] = peaks[-10:] = False
    peaks[:, :10] = peaks[:, -10:] = False
    y, x = np.nonzero(peaks)
    return np.stack([x, y], axis=1).astype(np.float32)


def synthetic_scores(network: Any, prepared: Path, model: Any) -> Any:
    """在冻结合成验证集上比较四种亚像素方法。

    Args:
        network: CUDA eval 网络。
        prepared: 包含 val_scenes.json 的目录。
        model: 合成场景的 A-only 模型。

    Returns:
        每方法的真值/预测/匹配数、precision/recall、匹配误差和含漏检惩罚的选择分数。

    Note:
        固定 256 px 裁块与 1 px 匹配门限；均值仅对匹配点计算，无匹配时为零，
        因而不能脱离召回率和 selection_cost 单独解释。
    """
    scenes = json.loads((prepared / "val_scenes.json").read_text())["scenes"]
    totals: dict[str, Any] = {
        m: dict(truth=0, predicted=0, matched=0, error_sum=0.0, errors=[]) for m in METHODS
    }
    for i, scene in enumerate(scenes):
        image, _, metadata = render_sample((model, scene, 256))
        response = dense_response(network, image)
        raw = peaks_from_response(response)
        truth = np.asarray(metadata["corners"]).reshape(-1, 2)
        truth = truth[np.all((truth >= 10) & (truth < 246), axis=1)]
        for method in METHODS:
            predicted = refine(image, raw, response, method)
            _, errors = match_reference(truth, predicted)
            total = totals[method]
            total["truth"] += len(truth)
            total["predicted"] += len(predicted)
            total["matched"] += len(errors)
            total["error_sum"] += float(np.sum(errors))
            total["errors"].extend(errors.tolist())
        if i % 50 == 0:
            print(f"synthetic validation {i + 1}/{len(scenes)}", flush=True)
    for total in totals.values():
        total["recall"] = total["matched"] / max(1, total["truth"])
        total["precision"] = total["matched"] / max(1, total["predicted"])
        total["matched_mean_px"] = total["error_sum"] / max(1, total["matched"])
        total["matched_p95_px"] = (
            float(np.percentile(total.pop("errors"), 95)) if total["matched"] else None
        )
        total["selection_cost"] = (
            total["error_sum"]
            + total["truth"]
            - total["matched"]
            + total["predicted"]
            - total["matched"]
        ) / max(1, total["truth"])
    return totals


def real_scores(network: Any, model: Any, paths: Any, methods: Any, output: Path) -> Any:
    """在指定真实图像对上独立筛选网格并比较细化方法的几何一致性。

    Args:
        network: 固定 CUDA eval 网络。
        model: 含 board、size 的冻结相机模型。
        paths: 图像对编号到完整左右路径的映射。
        methods: 要比较的 METHODS 子集。
        output: 保存逐对 JSON 的目录，同名结果覆盖。

    Returns:
        每方法完整图像数和成功图像对的几何记录；失败保存在逐对 JSON 中。

    Raises:
        ValueError: 图像分辨率与相机模型不匹配。

    Note:
        不调用 SB 建立编号；A/B 划分与哈希检查由 CLI 承担。
    """
    rows = []
    for pair_id, sides in paths.items():
        predictions: dict[str, Any] = {m: {} for m in methods}
        for side, path in sides.items():
            image = read_gray(path)
            if (image.shape[1], image.shape[0]) != tuple(model["size"]):
                raise ValueError("Wrong image resolution")
            response = dense_response(network, image)
            raw = peaks_from_response(response)
            for method in methods:
                points = refine(image, raw, response, method)
                try:
                    grid = assemble_grid(
                        points,
                        (model["board"]["columns"], model["board"]["rows"]),
                        dict(K=model[f"K_{side}"], D=model[f"D_{side}"]),
                    )
                    predictions[method][side] = grid
                except (ValueError, cv2.error) as error:
                    predictions[method][side] = dict(error=str(error))
        row: dict[str, Any] = dict(pair_id=pair_id, methods={})
        for method in methods:
            result = dict(images=predictions[method])
            if all("corners" in predictions[method][s] for s in ("left", "right")):
                obs = {
                    s: predictions[method][s]["corners"].reshape(-1, 1, 2)
                    for s in ("left", "right")
                }
                try:
                    fit = estimate(object_points(model["board"]), obs, model)
                    fit["epipolar_mean_px"] = float(
                        np.mean(symmetric_epipolar(obs["left"], obs["right"], model))
                    )
                    result["geometry"] = fit
                except (ValueError, cv2.error) as error:
                    result["error"] = str(error)
            row["methods"][method] = result
        rows.append(row)
        write_json(output / f"{pair_id}.json", row)
        print(f"real: {pair_id}", flush=True)
    summary = {}
    for method in methods:
        valid = [row for row in rows if "geometry" in row["methods"][method]]
        summary[method] = dict(
            complete_images=sum(
                "corners" in row["methods"][method]["images"][s]
                for row in rows
                for s in ("left", "right")
            ),
            geometry_by_pair={row["pair_id"]: row["methods"][method]["geometry"] for row in valid},
        )
    return summary


def select_method(synthetic: Any, real: Any) -> Any:
    """通过 A 安全门槛后，按合成验证含漏检惩罚的分数选择细化方法。

    Args:
        synthetic: 四方法的 recall、selection_cost 记录。
        real: A 集完整图像数和 geometry_by_pair，必须含 win3 基线。

    Returns:
        selected、各方法 eligibility 及 B_used_for_selection=False。
        无合格候选时回退 win3。

    Note:
        要求基线图像对无丢失，覆盖不减少，合成召回下降不超过 0.005，
        A 共同对上的联合/极线均值不超过基线 110%；调用方保证输入确为 A。
    """
    baseline = real["win3"]["geometry_by_pair"]
    eligibility = {}
    for method in METHODS:
        candidate = real[method]["geometry_by_pair"]
        ok = bool(baseline) and set(baseline).issubset(candidate)
        ok &= real[method]["complete_images"] >= real["win3"]["complete_images"]
        ok &= synthetic[method]["recall"] >= synthetic["win3"]["recall"] - 0.005
        if ok:
            for key in ("joint_rms_px", "epipolar_mean_px"):
                ok &= np.mean([candidate[i][key] for i in baseline]) <= 1.1 * np.mean(
                    [baseline[i][key] for i in baseline]
                )
        eligibility[method] = bool(ok)
    eligible = [m for m in METHODS if eligibility[m]]
    selected = min(eligible, key=lambda m: synthetic[m]["selection_cost"]) if eligible else "win3"
    return dict(selected=selected, eligibility=eligibility, B_used_for_selection=False)


def main() -> None:
    """编排 A+合成验证选型，或使用冻结 selection 执行 B 复核。

    不传 --selection 时只比较 A 与合成验证，再写 selection.json；
    传入时核对模型哈希，只在 B 比较 win3 和已选方案。
    两条路径均校验图像编号/哈希、要求新输出目录与 CUDA。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    split = json.loads((args.prepared / "split.json").read_text())
    group = "B" if args.selection else "A"
    paths = pair_paths(args.images)
    if set(paths) != set(split[group]):
        raise ValueError("Dataset must match the frozen split exactly")
    hashes = {row["pair_id"]: row for row in split["manifest"]}
    for pair_id, sides in paths.items():
        if set(sides) != {"left", "right"}:
            raise ValueError("Incomplete stereo pair")
        for side, path in sides.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[pair_id][f"{side}_sha256"]:
                raise ValueError("Image hash mismatch")
    model = read_model(args.prepared / "theta_A.json")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260905)
    torch.set_num_threads(2)
    network = corner_network().cuda().eval()
    network.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)["model"]
    )
    provenance = dict(
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        camera_sha256=hashlib.sha256((args.prepared / "theta_A.json").read_bytes()).hexdigest(),
    )
    if args.selection:
        selection = json.loads(args.selection.read_text())
        if any(selection[k] != v for k, v in provenance.items()):
            raise ValueError("Frozen selection model mismatch")
        methods = list(dict.fromkeys(["win3", selection["selected"]]))
        scores = real_scores(network, model, paths, methods, args.output)
        write_json(args.output / "B_scores.json", scores)
        write_json(args.output / "frozen_selection.json", selection)
    else:
        write_json(
            args.output / "protocol.json",
            dict(
                methods=METHODS,
                split="A_and_synthetic_val",
                **provenance,
                rule=(
                    "Min capped localization cost plus unmatched-candidate penalty; "
                    "A coverage not worse; A mean joint/epipolar <=110% baseline; "
                    "synthetic recall drop <=0.005."
                ),
            ),
        )
        synthetic = synthetic_scores(network, args.prepared, model)
        write_json(args.output / "synthetic_scores.json", synthetic)
        real = real_scores(network, model, paths, METHODS, args.output)
        write_json(args.output / "A_scores.json", real)
        selection = dict(**select_method(synthetic, real), **provenance)
        write_json(args.output / "selection.json", selection)
        print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
