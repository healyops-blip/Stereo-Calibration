"""Freeze grouped real splits and recalibrate using A only."""

import argparse
import hashlib
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.calibration import calibrate
from src.corners import collect, object_points
from src.deeplearning.synthetic import make_scene
from src.export import write_json
from src.validation import grouped_folds


def prepare(root: Path, output: Path, count: int) -> None:
    """冻结分组 A/B/C 编号，以 A-only 标定参数生成合成场景描述。

    Args:
        root: 含 config.yaml 的项目根目录。
        output: 尚不存在的准备目录。
        count: 训练场景数量；验证/测试各取 max(20, count // 10)。

    Raises:
        FileExistsError: 输出目录已存在。
        ValueError: 分组、标定、训练位姿或可见场景采样不成功。

    Note:
        写入 split、theta_A、场景 JSON 及角点图；不渲染训练裁块。
        B 固定为首个验证折，C 只记录编号及先前查看状态。
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared experiment: {output}")
    config = yaml.safe_load((root / "config.yaml").read_text())
    cv2.setNumThreads(1)
    cv2.setRNGSeed(config["seed"])
    observations, manifest = collect(
        root / config["calibration_dir"],
        config["board"],
        output / "corners",
        config["excluded_ids"],
    )
    folds = grouped_folds(observations, 5, config["seed"], config["near_duplicate_rms_px"])
    held_out = set(folds[0])
    training = [o for i, o in enumerate(observations) if i not in held_out]
    validation = [o for i, o in enumerate(observations) if i in held_out]
    model = calibrate(training, object_points(config["board"]), joint=True)
    depths = []
    for obs in training:
        ok, _, t = cv2.solvePnP(
            object_points(config["board"]), obs["left"], model["K_left"], model["D_left"]
        )
        if not ok or t[2, 0] <= 0:
            raise ValueError("Training pose invalid")
        depths.append(float(t[2, 0]))
    model.update(
        board=config["board"],
        generation_depth_mm=[min(depths) * 0.9, max(depths) * 1.1],
        model="pinhole_five_coefficients",
        convention="P_R=R_RL@P_L+t_RL; mm",
        calibration_policy="C2_fixed_in_advance_A_only",
        seed=config["seed"],
    )
    split: dict[str, Any] = dict(
        A=[o["pair_id"] for o in training],
        B=[o["pair_id"] for o in validation],
        C=config["expected_test_ids"],
        C_dataset=config["pose_dir"],
        C_status="previously_inspected_holdout_not_blind",
        excluded=config["excluded_ids"],
        policy="first_grouped_fold_is_B_no_residual_selection",
        manifest=manifest,
    )
    write_json(output / "split.json", split)
    write_json(output / "theta_A.json", model)
    digest = hashlib.sha256((output / "theta_A.json").read_bytes()).hexdigest()
    for name, n in [
        ("train", count),
        ("val", max(20, count // 10)),
        ("test", max(20, count // 10)),
    ]:
        scenes = [make_scene(model, name, i) for i in range(n)]
        write_json(output / f"{name}_scenes.json", dict(camera_sha256=digest, scenes=scenes))
    print(
        f"A={len(training)}, B={len(validation)}; baseline={model['baseline_mm']:.4f} mm",
        flush=True,
    )


def main() -> None:
    """解析 --root、--output、--scenes，要求至少二十个训练场景后准备数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=2000)
    args = parser.parse_args()
    if args.scenes < 20:
        parser.error("At least 20 scenes required")
    prepare(args.root.resolve(), args.output.resolve(), args.scenes)


if __name__ == "__main__":
    main()
