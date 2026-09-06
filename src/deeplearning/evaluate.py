"""Fixed-threshold synthetic validation only; not a real-world accuracy claim."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.deeplearning.subpixel import refine
from src.deeplearning.train import corner_network, render_sample
from src.export import write_json


def main() -> None:
    """用冻结权重评估合成验证集的网络加 win3 角点定位。

    读取 --prepared、--run，不使用真实 B/C。阈值 0.5、NMS 7×7、
    匹配半径 1 px，写 synthetic_validation.json；同名文件覆盖。
    需要 CUDA；匹配平均误差不包含漏检，必须结合 precision/recall 阅读。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((args.prepared / "theta_A.json").read_text())
    scenes = json.loads((args.prepared / "val_scenes.json").read_text())["scenes"]
    config = json.loads((args.run / "config.json").read_text())
    network = corner_network().cuda()
    # Only load weights we generated locally, never an untrusted checkpoint.
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(state["model"])
    network.eval()
    true_count = predicted_count = matches = 0
    errors = []
    with torch.inference_mode():
        for scene in scenes:
            image, _, metadata = render_sample((model, scene, config["size"]))
            response = network(torch.from_numpy(image[None, None]).cuda().float() / 255)
            response = response.sigmoid()[0, 0].cpu().numpy()
            peaks = response == cv2.dilate(response, np.ones((7, 7), np.uint8))
            peaks &= response >= 0.5
            peaks[:10] = peaks[-10:] = False
            peaks[:, :10] = peaks[:, -10:] = False
            y, x = np.nonzero(peaks)
            predicted = np.stack([x, y], axis=1).astype(np.float32)
            predicted = refine(image, predicted, response, "win3")
            truth = np.asarray(metadata["corners"]).reshape(-1, 2)
            truth = truth[np.all((truth >= 10) & (truth < config["size"] - 10), axis=1)]
            true_count += len(truth)
            predicted_count += len(predicted)
            if len(truth) and len(predicted):
                distances = np.linalg.norm(truth[:, None] - predicted[None], axis=2)
                cost = np.where(distances <= 1, distances, 1e6)
                rows, cols = linear_sum_assignment(cost)
                accepted = distances[rows, cols]
                accepted = accepted[accepted <= 1]
                matches += len(accepted)
                errors.extend(accepted.tolist())
    result = dict(
        split="synthetic_val",
        threshold=0.5,
        matching_radius_px=1.0,
        refinement="Q0_cornerSubPix",
        true_count=true_count,
        predicted_count=predicted_count,
        matches=matches,
        precision=matches / max(1, predicted_count),
        recall=matches / max(1, true_count),
        matched_mean_error_px=float(np.mean(errors)) if errors else None,
        caveat="Matched error excludes misses; no real B/C accuracy measured.",
    )
    write_json(args.run / "synthetic_validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
