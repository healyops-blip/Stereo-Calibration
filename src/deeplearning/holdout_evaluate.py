"""One frozen C evaluation: independent neural ordering versus explicit SB."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.corners import detect, object_points
from src.dataset import pair_paths, read_gray
from src.deeplearning.holdout_support import pose_transforms, summarize_holdout
from src.deeplearning.infer_grid import detect_ordered
from src.deeplearning.real_metrics import symmetric_epipolar
from src.deeplearning.train import corner_network
from src.export import read_model, write_csv, write_json
from src.pose import estimate


def save_overlay(
    image: Any, corners: Any, transform: Any, model: Any, side: str, path: Path
) -> None:
    """在一侧原图上绘制有序棋盘和三格长的坐标轴。

    Args:
        image: 灰度原图 (H, W)，只读。
        corners: 当前方法的有序角点 (N, 1, 2)，px。
        transform: 棋盘到当前相机的齐次变换 (4, 4)，平移 mm。
        model: 当前模型的 K、D 和 board。
        side: left 或 right，用于选择内参。
        path: 输出图路径，父目录须存在，同名文件覆盖。

    Raises:
        OSError: 图片写入失败。
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(
        canvas, (model["board"]["columns"], model["board"]["rows"]), corners, True
    )
    cv2.drawFrameAxes(
        canvas,
        model[f"K_{side}"],
        model[f"D_{side}"],
        cv2.Rodrigues(transform[:3, :3])[0],
        transform[:3, 3],
        3 * model["board"]["square_mm"],
        2,
    )
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Cannot save {path}")


def evaluate_pair(pair_id: str, paths: Any, network: Any, model: Any, output: Path) -> Any:
    """在一对 C 图上独立比较 log3 网络与 SB 的位姿。

    Args:
        pair_id: 冻结测试图像对编号。
        paths: 完整 left/right 原图路径映射。
        network: CUDA eval 网络。
        model: 固定 A-only 相机模型及 board。
        output: 写入各方法角点/坐标轴 PNG 的目录。

    Returns:
        两方法的角点、检测失败、位姿及齐次变换记录。

    Raises:
        ValueError: 原图分辨率不匹配。
        OSError: 原图读取或叠加图保存失败。

    Note:
        网络检测排序先执行且不接收 SB 点；方法内检测/几何失败记录后继续。
    """
    shape = (model["board"]["columns"], model["board"]["rows"])
    images = {side: read_gray(path) for side, path in paths.items()}
    if any((im.shape[1], im.shape[0]) != tuple(model["size"]) for im in images.values()):
        raise ValueError("C image resolution mismatch")
    row: dict[str, Any] = dict(pair_id=pair_id, methods={})
    for method in ("log3", "sb"):
        observations, failures = {}, {}
        for side, image in images.items():
            try:
                if method == "log3":
                    grid = detect_ordered(
                        network,
                        image,
                        shape,
                        dict(K=model[f"K_{side}"], D=model[f"D_{side}"]),
                        "log3",
                    )
                    observations[side] = grid["corners"].reshape(-1, 1, 2)
                else:
                    observations[side] = detect(image, shape, "sb")[0]
            except (ValueError, cv2.error) as error:
                failures[side] = str(error)
        result: dict[str, Any] = dict(corners=observations, detection_failures=failures)
        if failures:
            result["error"] = "incomplete_stereo_detection"
        else:
            try:
                fit = estimate(object_points(model["board"]), observations, model)
                fit["epipolar_mean_px"] = float(
                    np.mean(symmetric_epipolar(observations["left"], observations["right"], model))
                )
                transforms = pose_transforms(fit["pose"], model)
                result.update(pose=fit, transforms=transforms)
                for side in images:
                    save_overlay(
                        images[side],
                        observations[side],
                        transforms[f"T_board_to_{side}"],
                        model,
                        side,
                        output / f"{pair_id}_{side}_{method}.png",
                    )
            except (ValueError, cv2.error) as error:
                result["error"] = str(error)
        row["methods"][method] = result
    return row


def main() -> None:
    """核对 C 编号、原图/相机/权重哈希，执行固定 log3 与 SB 比较。

    要求 --prepared、--weights、--selection、--images、--manifest、--output。
    必须使用冻结 log3 选择及新输出目录；写入协议、逐对 JSON、poses.csv、
    汇总和叠加图，不在 C 上调阈值、选择模型或重新标定相机。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "weights", "selection", "images", "manifest", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = read_model(args.prepared / "theta_A.json")
    split = json.loads((args.prepared / "split.json").read_text())
    selection = json.loads(args.selection.read_text())
    manifest = json.loads(args.manifest.read_text())
    paths = pair_paths(args.images)
    if set(paths) != set(split["C"]) or set(paths) != set(manifest["images"]):
        raise ValueError("Frozen C identifiers mismatch")
    for pair_id, sides in paths.items():
        if set(sides) != {"left", "right"}:
            raise ValueError("Missing stereo side")
        for side, path in sides.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["images"][pair_id][side]:
                raise ValueError("C image hash mismatch")
    provenance = dict(
        camera_sha256=hashlib.sha256((args.prepared / "theta_A.json").read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
    )
    if selection["selected"] != "log3" or any(selection[k] != v for k, v in provenance.items()):
        raise ValueError("Unexpected frozen selection")
    protocol = dict(
        **provenance,
        selection=selection,
        C_manifest=manifest,
        split_status=split["C_status"],
        opencv=cv2.__version__,
        torch=str(torch.__version__),
        code_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        policy="No threshold changes, camera refit, SB-assisted ordering or C tuning.",
    )
    write_json(args.output / "protocol.json", protocol)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260905)
    torch.set_num_threads(2)
    network = corner_network().cuda().eval()
    network.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)["model"]
    )
    rows, table = [], []
    for pair_id, sides in paths.items():
        row = evaluate_pair(pair_id, sides, network, model, args.output)
        rows.append(row)
        write_json(args.output / f"{pair_id}.json", row)
        for method, result in row["methods"].items():
            record = dict(
                pair_id=pair_id, method=method, status="pose" if "pose" in result else "failed"
            )
            if "pose" in result:
                fit = result["pose"]
                record.update(
                    {
                        k: fit[k]
                        for k in (
                            "joint_rms_px",
                            "p0_right_prediction_rms_px",
                            "epipolar_mean_px",
                            "ambiguity_flag",
                            "positive_depth",
                        )
                    }
                )
                record.update(
                    {
                        k: float(v)
                        for k, v in zip(
                            ["rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm"],
                            fit["pose"],
                            strict=True,
                        )
                    }
                )
            table.append(record)
        print(f"C {pair_id}: {[r['status'] for r in table[-2:]]}", flush=True)
    summary = summarize_holdout(rows)
    write_json(args.output / "summary.json", summary)
    write_csv(args.output / "poses.csv", table)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
