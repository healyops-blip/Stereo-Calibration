"""Grouped calibration-only validation and geometric consistency diagnostics."""

from typing import Any

import cv2
import numpy as np

from src.calibration import calibrate
from src.pose import estimate, project, rms


def geometry(obs: dict[str, Any], model: dict[str, Any], board: dict[str, Any]) -> dict[str, Any]:
    """计算校正极线与三角化棋盘边长的内部一致性指标。

    Args:
        obs: 编号一致的左右角点，形状 (N, 1, 2)，单位 px。
        model: 五参数双目模型；平移单位 mm。
        board: rows、columns、square_mm，点数必须等于行列乘积。

    Returns:
        极线均值/中位数/P95（px）、相邻边长偏差/RMSE（mm）、
        非视差轴名称及三角化正深度标记。

    Raises:
        ValueError: 三角化得到无穷远点，或数组无法按棋盘重排。

    Note:
        这些指标不是独立尺度真值；横向双目统计 y 误差，纵向双目统计 x。
    """
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


def near_duplicate_groups(observations: list[dict[str, Any]], threshold: float) -> list[list[int]]:
    """由原始双目角点距离构造近重复样本的连通分组。

    Args:
        observations: 已对齐编号的完整双目观测列表。
        threshold: 左右角点 RMS 距离最大值的严格上界，单位 px。

    Returns:
        各组的观测索引列表；连通关系具有传递性，非逐组直径约束。

    Raises:
        ValueError: threshold 不大于零。
    """
    if threshold <= 0:
        raise ValueError("Near-duplicate threshold must be positive")
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
    return [sorted(group) for group in groups]


def split_groups(groups: list[list[int]], folds: int, seed: int) -> list[list[int]]:
    """按组分配验证折，使近重复样本不跨折。

    Args:
        groups: 互不重叠的样本索引分组；调用方保证分组完整。
        folds: 至少两折，且不多于组数。
        seed: 随机打乱同规模组次序的种子。

    Returns:
        各折的留出样本索引。大组优先分配到当前样本数最少的折。

    Raises:
        ValueError: 折数不足两折或独立组数不足。
    """
    if folds < 2 or len(groups) < folds:
        raise ValueError(f"Only {len(groups)} independent groups for {folds} folds")
    groups = [list(group) for group in groups]
    np.random.default_rng(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    result: list[list[int]] = [[] for _ in range(folds)]
    for group in groups:
        min(result, key=len).extend(sorted(group))
    return result


def grouped_folds(
    observations: list[dict[str, Any]], folds: int, seed: int, threshold: float
) -> list[list[int]]:
    """从原始角点完成近重复分组和验证折划分。

    Args:
        observations: 完整双目观测列表。
        folds: 验证折数，至少二且不超过组数。
        seed: 折划分随机种子。
        threshold: 近重复角点 RMS 门限，px。

    Returns:
        各折留出索引；不依赖拟合后的重投影残差。
    """
    return split_groups(near_duplicate_groups(observations, threshold), folds, seed)


def cross_validate(
    observations: list[dict[str, Any]], points: Any, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """在固定分组划分上比较 C0 与 C2 的留出图像预测。

    Args:
        observations: 完整双目标定观测。
        points: 棋盘物点 (N, 3)，mm。
        config: 含 folds、seed、near_duplicate_rms_px 与 board 的配置。

    Returns:
        逐候选、逐折、逐对指标列表，以及使用的留出折索引。

    Note:
        每折重新标定；右图预测采用左图选择的 P0，不用右图选择初始姿态。
        求解失败向调用方传播。
    """
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
    """按 C0/C2 汇总逐对交叉验证记录。

    Args:
        rows: 含 candidate、右图预测 RMS 与极线统计的非空记录。

    Returns:
        两候选的留出数量、逐对 RMS 的平方均值根和极线算术均值，单位 px。

    Note:
        不使用组等权汇总，也不进行显著性检验。
    """
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
    """对图像对有放回重采样，报告固定模型策略的参数稳定性。

    Args:
        observations: 参与重采样的完整双目观测。
        points: 对应棋盘物点 (N, 3)，mm。
        config: 含 seed 与 bootstrap_samples 的配置。
        joint: 是否联合优化内参，传给 calibrate。

    Returns:
        每次基线 mm、左右 fx px、样本标准差和失败记录。
        成功不足两次时 standard_deviation 为 None。

    Note:
        单次有效样本不足六对或求解失败会记录并继续；不表示绝对误差或置信区间。
    """
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
    """利用联合 RMS 的中位数和 MAD 标记需要人工复核的样本。

    Args:
        rows: 含 pair_id、joint_rms_px 的非空记录。

    Returns:
        阈值（px）、MAD 退化标记和建议复核编号，不自动删除任何数据。
    """
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
    """按角点索引模五留出，检验固定相机下的位姿预测。

    Args:
        points: 棋盘物点 (N, 3)，mm，剩余点须满足位姿求解条件。
        obs: 左右对应角点 (N, 1, 2)，px。
        model: 冻结的相机内外参。

    Returns:
        每个角点被留出一次后，左右全部预测残差的总体 RMS，px。

    Note:
        左右同时留出相同编号。只重估棋盘位姿，不重标定相机；无外部真值。
    """
    errors = []
    for fold in range(5):
        held_out = np.arange(len(points)) % 5 == fold
        training = {side: obs[side][~held_out] for side in ("left", "right")}
        fit = estimate(points[~held_out], training, model)
        predictions = project(points[held_out], fit["pose"], model)
        for side, predicted in zip(("left", "right"), predictions, strict=True):
            errors.append(predicted - obs[side][held_out].reshape(-1, 2))
    return rms(np.concatenate(errors))
