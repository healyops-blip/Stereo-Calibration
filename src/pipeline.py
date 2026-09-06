"""Run the three delivery stages independently or as one reproducible pipeline."""

import argparse
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.artifact_validation import verify
from src.calibration import calibrate
from src.corners import collect, object_points
from src.export import (
    ORIGIN,
    draw_pose,
    empty_pose_row,
    export_model,
    pose_row,
    read_model,
    write_csv,
    write_json,
)
from src.pose import estimate
from src.reporting import report
from src.validation import bootstrap, cross_validate, geometry, outlier_hints, summarize_cv


def stage_calibration(config: dict[str, Any], root: Path, output: Path) -> None:
    observations, manifest = collect(
        root / config["calibration_dir"],
        config["board"],
        output / "calibration/corners",
        config["excluded_ids"],
    )
    write_csv(output / "calibration/pair_manifest.csv", manifest)
    write_json(output / "calibration/calibration_observations.json", observations)
    points = object_points(config["board"])
    cv_rows, folds = cross_validate(observations, points, config)
    write_csv(output / "calibration/cross_validation.csv", cv_rows)
    summaries = summarize_cv(cv_rows)
    # Camera model selection uses calibration-only held-out RIGHT prediction RMS.
    selected = min(summaries, key=lambda key: summaries[key]["right_prediction_rms_px"])
    models = {name: calibrate(observations, points, joint=name == "C2") for name in ("C0", "C2")}
    model = models[selected]
    quality = []
    for obs in observations:
        fit = estimate(points, obs, model)
        quality.append(
            dict(
                pair_id=obs["pair_id"],
                **{key: value for key, value in fit.items() if key not in ("pose", "initial_pose")},
                **geometry(obs, model, config["board"]),
            )
        )
    hints = outlier_hints(quality)
    stability = bootstrap(observations, points, config, joint=selected == "C2")
    for name, candidate in models.items():
        write_json(output / "calibration" / f"{name}_calibration.json", candidate)
    model.update(
        board=config["board"],
        selected_candidate=selected,
        convention="P_R = R_RL @ P_L + t_RL; mm; original camera frames",
        board_origin_convention=ORIGIN,
        model="pinhole_five_coefficients",
        distortion_order=["k1", "k2", "p1", "p2", "k3"],
        config=config,
        python=platform.python_version(),
        dependencies={
            name: version(name) for name in ("numpy", "scipy", "opencv-python-headless", "PyYAML")
        },
    )
    write_json(output / "calibration/calibration.json", model)
    export_model(output / "calibration/calibration_results.csv", model, config["board"])
    write_csv(output / "calibration/calibration_quality.csv", quality)
    write_json(
        output / "calibration/calibration_validation.json",
        dict(
            inventory=len(manifest),
            preexcluded=sum(r["pair_id"] in config["excluded_ids"] for r in manifest),
            detected=len(observations),
            retained=len(observations),
            folds=[[observations[i]["pair_id"] for i in fold] for fold in folds],
            cv=summaries,
            selected=selected,
            outlier_review=hints,
            bootstrap=stability,
            c1_status="not_applied_no_confirmed_additional_exclusions; C2 uses C0 retained pairs",
        ),
    )
    print(f"Calibration frozen: {selected}, RMS={model['stereo_rms_px']:.4f} px", flush=True)


def stage_pose(config: dict[str, Any], root: Path, output: Path) -> None:
    model_path = output / "calibration/calibration.json"
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = read_model(model_path)
    if config["board"] != model["board"]:
        raise ValueError("Board configuration differs from frozen calibration")
    observations, manifest = collect(
        root / config["pose_dir"], model["board"], output / "pose/corners", []
    )
    write_csv(output / "pose/pose_manifest.csv", manifest)
    write_json(output / "pose/pose_observations.json", observations)
    points = object_points(model["board"])
    indexed = {obs["pair_id"]: obs for obs in observations}
    failures = {r["pair_id"]: r.get("failure_reason", "") for r in manifest}
    expected = config["expected_test_ids"]
    if any(r["pair_id"] not in expected for r in manifest):
        raise ValueError("Unexpected test IDs; update expected_test_ids explicitly")
    rows = []
    for pair_id in expected:
        try:
            if pair_id not in indexed:
                raise ValueError(failures.get(pair_id, "missing_pair"))
            obs = indexed[pair_id]
            if tuple(obs["size"]) != tuple(model["size"]):
                raise ValueError("Test image size differs from frozen calibration")
            fit = estimate(points, obs, model)
            row = pose_row(pair_id, fit)
            row.update(geometry(obs, model, model["board"]))
            draw_pose(obs, fit, model, output / "pose/axes", model["board"]["square_mm"])
        except (ValueError, cv2.error) as error:
            row = empty_pose_row(pair_id, str(error))
        rows.append(row)
    write_csv(output / "pose/pose_results.csv", rows)
    after = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("Frozen camera parameters changed during pose estimation")
    write_json(
        output / "pose/pose_provenance.json",
        dict(
            calibration_sha256=before,
            expected_ids=expected,
            success_count=sum(r["status"] == "ok" for r in rows),
        ),
    )
    print(f"Pose rows exported: {len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "calibrate", "pose", "validate"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml"
    )
    args = parser.parse_args()
    root = args.config.resolve().parent
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(config["seed"])
    if args.stage in ("all", "calibrate"):
        stage_calibration(config, root, output)
    if args.stage in ("all", "pose"):
        stage_pose(config, root, output)
    if args.stage in ("all", "validate"):
        print(verify(output, config["expected_test_ids"]), flush=True)
    report(output)


if __name__ == "__main__":
    main()
