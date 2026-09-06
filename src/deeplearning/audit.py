"""Check renderer alignment and sampling convergence on training scenes only."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.deeplearning.synthetic import render_crop
from src.export import write_json


def main() -> None:
    """在前两百个训练场景比较 4×4 与 8×8 采样及亚像素位置。

    读取 --prepared 中的模型与场景，写 renderer_audit.json。
    只验证渲染采样的一致性，cornerSubPix 不是独立物理真值。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((args.prepared / "theta_A.json").read_text())
    scenes = json.loads((args.prepared / "train_scenes.json").read_text())["scenes"][:200]
    errors, shifts = [], []
    cv2.setNumThreads(1)
    for scene in scenes:
        side = "left" if scene["index"] % 2 == 0 else "right"
        pose = scene[side]
        corner = np.asarray(pose["corners"])[54]
        origin = np.rint(corner - 32).astype(int)
        label = corner - origin
        estimates = []
        for samples in (4, 8):
            image = render_crop(
                {"K": model[f"K_{side}"], "D": model[f"D_{side}"]},
                model["board"],
                np.asarray(pose["R"]),
                np.asarray(pose["t"]),
                tuple(origin),
                64,
                samples,
            )
            point = np.array([[label]], dtype=np.float32)
            estimates.append(
                cv2.cornerSubPix(
                    image,
                    point,
                    (3, 3),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-5),
                )[0, 0]
            )
        errors.append(float(np.linalg.norm(estimates[1] - label)))
        shifts.append(float(np.linalg.norm(estimates[0] - estimates[1])))
    result = dict(
        count=len(scenes),
        label_vs_subpix_8x_mean_px=float(np.mean(errors)),
        label_vs_subpix_8x_p95_px=float(np.percentile(errors, 95)),
        subpix_4x_vs_8x_p95_px=float(np.percentile(shifts, 95)),
        note="Diagnostic only: cornerSubPix is not independent physical ground truth.",
    )
    write_json(args.prepared / "renderer_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
