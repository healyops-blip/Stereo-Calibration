"""Grouped calibration-only validation and geometric consistency diagnostics."""

from typing import Any

import cv2
import numpy as np

from src.calibration import calibrate
from src.pose import estimate, project, rms


def geometry(obs: dict[str, Any], model: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    kl, dl, kr, dr = (model[key] for key in ("K_left", "D_left", "K_right", "D_right"))
    rotation, translation = model["R_RL"], model["t_RL"]
    rl, rr, pl, pr, _, _, _ = cv2.stereoRectify(
        kl, dl, kr, dr, model["size"], rotation, translation
    )
    left = cv2.undistortPoints(obs["left"], kl, dl, R=rl, P=pl).reshape(-1, 2)
    right = cv2.undistortPoints(obs["right"], kr, dr, R=rr, P=pr).reshape(-1, 2)
    # OpenCV can choose vertical rectification for a predominantly vertical baseline.
    axis = 1 if abs(pr[0, 3]) >= abs(pr[1, 3]) else 0
    epipolar = np.abs(left[:, axis] - right[:, axis])
    ul = cv2.undistortPoints(obs["left"], kl, dl).reshape(-1, 2)
    ur = cv2.undistortPoints(obs["right"], kr, dr).reshape(-1, 2)
    homogeneous = cv2.triangulatePoints(
        np.c_[np.eye(3), np.zeros(3)], np.c_[rotation, translation], ul.T, ur.T
    )
    if np.any(np.abs(homogeneous[3]) < 1e-12):
        raise ValueError("Triangulated point at infinity")
    xyz = (homogeneous[:3] / homogeneous[3]).T
    grid = xyz.reshape(board["rows"], board["columns"], 3)
    lengths = np.r_[
        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
    ]
    error = lengths - board["square_mm"]
    return dict(
        epipolar_mean_px=float(epipolar.mean()),
        epipolar_median_px=float(np.median(epipolar)),
        epipolar_p95_px=float(np.percentile(epipolar, 95)),
        rectification_epipolar_axis="y" if axis == 1 else "x",
        edge_mean_mm=float(lengths.mean()),
        edge_bias_mm=float(error.mean()),
        edge_rmse_mm=float(np.sqrt(np.mean(error**2))),
        edge_abs_p95_mm=float(np.percentile(np.abs(error), 95)),
        triangulation_positive_depth=bool(
            np.all(xyz[:, 2] > 0) and np.all((rotation @ xyz.T + translation.reshape(3, 1))[2] > 0)
        ),
    )


def grouped_folds(
    observations: list[dict[str, Any]], folds: int, seed: int, threshold: float
) -> list[list[int]]:
    """Connected near-duplicate observations stay together in every candidate model."""
    groups: list[set[int]] = []
    for index, obs in enumerate(observations):
        joined = {index}
        others = []
        for group in groups:
            if any(
                max(rms(obs[side] - observations[j][side]) for side in ("left", "right"))
                < threshold
                for j in group
            ):
                joined |= group
            else:
                others.append(group)
        groups = [*others, joined]
    if len(groups) < folds:
        raise ValueError(f"Only {len(groups)} independent groups for {folds} folds")
    np.random.default_rng(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    result: list[list[int]] = [[] for _ in range(folds)]
    for group in groups:
        min(result, key=len).extend(sorted(group))
    return result


def cross_validate(
    observations: list[dict[str, Any]], points: Any, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    folds = grouped_folds(
        observations, config["folds"], config["seed"], config["near_duplicate_rms_px"]
    )
    rows = []
    for name, joint in (("C0", False), ("C2", True)):
        for fold, held_out in enumerate(folds):
            training = [o for index, o in enumerate(observations) if index not in held_out]
            print(f"{name}: fold {fold + 1}/{len(folds)}", flush=True)
            model = calibrate(training, points, joint)
            for index in held_out:
                obs = observations[index]
                fit = estimate(points, obs, model)
                rows.append(
                    dict(
                        candidate=name,
                        fold=fold + 1,
                        pair_id=obs["pair_id"],
                        train_count=len(training),
                        train_rms_px=model["stereo_rms_px"],
                        right_prediction_rms_px=fit["p0_right_prediction_rms_px"],
                        joint_rms_px=fit["joint_rms_px"],
                        **geometry(obs, model, config["board"]),
                    )
                )
    return rows, folds


def summarize_cv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in ("C0", "C2"):
        selected = [row for row in rows if row["candidate"] == name]
        result[name] = dict(
            held_out_pairs=len(selected),
            right_prediction_rms_px=float(
                np.sqrt(np.mean([r["right_prediction_rms_px"] ** 2 for r in selected]))
            ),
            mean_epipolar_px=float(np.mean([r["epipolar_mean_px"] for r in selected])),
            mean_pair_epipolar_p95_px=float(np.mean([r["epipolar_p95_px"] for r in selected])),
        )
    return result


def bootstrap(
    observations: list[dict[str, Any]], points: Any, config: dict[str, Any], joint: bool
) -> dict[str, Any]:
    rng = np.random.default_rng(config["seed"] + 1)
    values, failures = [], []
    for iteration in range(config["bootstrap_samples"]):
        indices = rng.integers(0, len(observations), len(observations))
        try:
            if len(set(indices)) < 6:
                raise ValueError("Fewer than six distinct sampled poses")
            model = calibrate([observations[i] for i in indices], points, joint)
            values.append([model["baseline_mm"], model["K_left"][0, 0], model["K_right"][0, 0]])
        except (ValueError, cv2.error) as error:
            failures.append(dict(iteration=iteration, reason=str(error)))
        print(f"Bootstrap {iteration + 1}/{config['bootstrap_samples']}", flush=True)
    return dict(
        requested=config["bootstrap_samples"],
        succeeded=len(values),
        failed=len(failures),
        failures=failures,
        unit="stereo_pair",
        replacement=True,
        seed=config["seed"] + 1,
        columns=["baseline_mm", "left_fx_px", "right_fx_px"],
        samples=values,
        standard_deviation=np.std(values, axis=0, ddof=1).tolist() if len(values) > 1 else None,
        interpretation="empirical_stability_not_absolute_accuracy_or_confidence_interval",
    )


def outlier_hints(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.array([row["joint_rms_px"] for row in rows])
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = median + 3 * 1.4826 * mad
    return dict(
        median_px=median,
        mad_px=mad,
        threshold_px=threshold,
        degenerate_mad=mad < 1e-9,
        flagged_ids=[row["pair_id"] for row in rows if row["joint_rms_px"] > threshold]
        if mad >= 1e-9
        else [],
        action="review_only_no_automatic_removal",
    )


def held_out_corners(points: Any, obs: dict[str, Any], model: dict[str, Any]) -> float:
    """Each corner is predicted once from a pose fitted without that stereo corner."""
    errors = []
    for fold in range(5):
        held_out = np.arange(len(points)) % 5 == fold
        training = {side: obs[side][~held_out] for side in ("left", "right")}
        fit = estimate(points[~held_out], training, model)
        predictions = project(points[held_out], fit["pose"], model)
        for side, predicted in zip(("left", "right"), predictions, strict=True):
            errors.append(predicted - obs[side][held_out].reshape(-1, 2))
    return rms(np.concatenate(errors))
