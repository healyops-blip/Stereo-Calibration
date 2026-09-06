"""Standalone neural candidates + geometric ordering; never uses an SB reference."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import torch

from src.dataset import pair_paths, read_gray
from src.deeplearning.grid import assemble_grid
from src.deeplearning.real_evaluate import neural_candidates, raw_candidates
from src.deeplearning.subpixel import METHODS, refine
from src.deeplearning.train import corner_network
from src.export import read_model, write_json


def detect_ordered(
    network: Any, image: Any, shape: tuple[int, int], camera: Any, method: str = "win3"
) -> Any:
    """对单图执行网络候选、指定细化和独立棋盘网格排序。

    Args:
        network: CUDA eval 网络。
        image: 原图灰度数组 (H, W)。
        shape: 内角点 (columns, rows)。
        camera: 网格几何判断所需的 K、D；None 表示不去畸变。
        method: win3、win5、win7 或 log3；默认保留旧 win3。

    Returns:
        assemble_grid 的有序原图角点与筛选指标，不读取 SB 参考。

    Raises:
        ValueError: 方法未知或候选无法组成合法完整棋盘。
    """
    if method not in METHODS:
        raise ValueError("Unknown refinement method")
    if method == "win3":
        _, refined, _ = neural_candidates(network, image)
    else:
        raw, response = raw_candidates(network, image)
        refined = refine(image, raw, response, method)
    return assemble_grid(refined, shape, camera)


def main() -> None:
    """从原始双目目录执行独立推理排序，保存每张图的成功或拒绝记录。

    读取 --model、--weights、--images，创建新 --output。
    可选 --selection 绑定相机/权重哈希并选择细化方法；不传时保持 win3。
    需要 CUDA，逐图检查分辨率；输出 ordered_corners.json。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260905)
    torch.set_num_threads(2)
    camera_model = read_model(args.model)
    method = "win3"
    if args.selection:
        selection = json.loads(args.selection.read_text())
        expected = dict(
            camera_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
            weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        )
        if any(selection[key] != value for key, value in expected.items()):
            raise ValueError("Selection does not match camera and network")
        method = selection["selected"]
        if method not in METHODS:
            raise ValueError("Invalid selected method")
    network = corner_network().cuda().eval()
    network.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)["model"]
    )
    shape = (camera_model["board"]["columns"], camera_model["board"]["rows"])
    records = []
    for pair_id, paths in pair_paths(args.images).items():
        for side, path in paths.items():
            row: dict[str, Any] = dict(pair_id=pair_id, side=side, refinement=method)
            try:
                image = read_gray(path)
                if (image.shape[1], image.shape[0]) != tuple(camera_model["size"]):
                    raise ValueError("Image resolution differs from calibration")
                result = detect_ordered(
                    network,
                    image,
                    shape,
                    dict(K=camera_model[f"K_{side}"], D=camera_model[f"D_{side}"]),
                    method,
                )
                row.update(result, status="complete")
            except (ValueError, cv2.error) as error:
                row.update(status="rejected", reason=str(error))
            records.append(row)
    write_json(args.output / "ordered_corners.json", records)
    completed = sum(row["status"] == "complete" for row in records)
    print(f"Independent grid inference: {completed}/{len(records)} images", flush=True)


if __name__ == "__main__":
    main()
