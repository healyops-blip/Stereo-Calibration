"""Audit frozen test poses, including held-out corners and CSV-based overlays."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.artifact_validation import verify
from src.corners import object_points
from src.dataset import read_gray
from src.export import camera_from_csv, read_csv, write_csv, write_json
from src.pose import positive_depth, project, rms
from src.validation import geometry, held_out_corners
from src.visualization import annotate


def main() -> None:
    """核验五对测试位姿，执行角点留出并导出图文报告。

    CLI 参数 --output-dir 指定交付根目录。先回读验证 CSV 和标定来源，
    核对原图哈希，随后写入 pose/validation 的指标、叠加图和 HTML。
    任一测试姿态失败会拒绝继续；无外部真值时绝对旋转和平移误差保持 null。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs"
    )
    output = parser.parse_args().output_dir.resolve()
    destination = output / "pose/validation"
    destination.mkdir(parents=True, exist_ok=True)
    ids = [f"{i:06d}" for i in range(1, 6)]
    verify(output, ids)
    model = camera_from_csv(output / "calibration/calibration_results.csv")
    points = object_points(model["board"]).astype(np.float64)
    poses = read_csv(output / "pose/pose_results.csv")
    saved = json.loads((output / "pose/pose_observations.json").read_text(encoding="utf-8"))
    observations = {o["pair_id"]: o for o in saved}
    manifest = {r["pair_id"]: r for r in read_csv(output / "pose/pose_manifest.csv")}
    rows, sections = [], []
    for row in poses:
        pair_id = row["pair_id"]
        if row["status"] != "ok":
            raise ValueError(f"Test pose failed: {pair_id}: {row['failure_reason']}")
        obs = observations[pair_id]
        for side in ("left", "right"):
            obs[side] = np.array(obs[side], dtype=np.float64)
            path = Path(obs["paths"][side])
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[pair_id][f"{side}_sha256"]:
                raise ValueError(f"Test image changed: {path}")
        pose = np.array(
            [float(row[key]) for key in ("rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm")]
        )
        predictions = project(points, pose, model)
        residuals = [
            p - obs[s].reshape(-1, 2) for s, p in zip(("left", "right"), predictions, strict=True)
        ]
        metrics = dict(
            pair_id=pair_id,
            left_rms_px=rms(residuals[0]),
            right_rms_px=rms(residuals[1]),
            joint_rms_px=rms(np.concatenate(residuals)),
            held_out_corner_rms_px=held_out_corners(points, obs, model),
            p0_right_prediction_rms_px=float(row["p0_right_prediction_rms_px"]),
            positive_depth=positive_depth(points, pose, model),
            ambiguity_flag=row["ambiguity_flag"],
            **geometry(obs, model, model["board"]),
        )
        np.testing.assert_allclose(metrics["joint_rms_px"], float(row["joint_rms_px"]), atol=1e-7)
        rows.append(metrics)
        figures = []
        for side, predicted in zip(("left", "right"), predictions, strict=True):
            image = read_gray(Path(obs["paths"][side]))
            observed = obs[side].reshape(-1, 2)
            title = f"TEST {pair_id} {side} | CSV pose RMS {metrics[f'{side}_rms_px']:.4f} px"
            name = f"{pair_id}_{side}"
            cv2.imwrite(
                str(destination / f"{name}.png"), annotate(image, observed, predicted, title)
            )
            lower = np.maximum(np.floor(observed.min(axis=0)).astype(int) - 20, 0)
            upper = np.minimum(np.ceil(observed.max(axis=0)).astype(int) + 20, model["size"])
            zoom = annotate(
                image[lower[1] : upper[1], lower[0] : upper[0]],
                observed,
                predicted,
                title,
                4,
                (int(lower[0]), int(lower[1])),
            )
            cv2.imwrite(str(destination / f"{name}_zoom.png"), zoom)
            figures.append(
                f'<figure><a href="{name}.png"><img src="{name}.png"></a>'
                f'<figcaption>{side} · <a href="{name}_zoom.png">4 倍放大</a>'
                f' · <a href="../axes/{name}.png">姿态坐标轴</a></figcaption></figure>'
            )
        sections.append(
            f'<h2>测试对 {pair_id}</h2><div class="pair">' + "".join(figures) + "</div>"
        )
        print(
            f"{pair_id}: fit={metrics['joint_rms_px']:.4f}, "
            f"held-out={metrics['held_out_corner_rms_px']:.4f} px",
            flush=True,
        )
    write_csv(destination / "metrics.csv", rows)
    summary = dict(
        test_count=len(rows),
        pooled_joint_rms_px=float(np.sqrt(np.mean([r["joint_rms_px"] ** 2 for r in rows]))),
        pooled_held_out_corner_rms_px=float(
            np.sqrt(np.mean([r["held_out_corner_rms_px"] ** 2 for r in rows]))
        ),
        absolute_translation_error_mm=None,
        absolute_rotation_error_deg=None,
        limitation="no_external_pose_ground_truth; fixed_camera_corner_holdout_only",
        source_hashes={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("calibration/calibration_results.csv", "pose/pose_results.csv")
        },
    )
    write_json(destination / "summary.json", summary)
    lines = [
        "# 测试数据姿态精度验证",
        "",
        "相机与正式姿态均从 CSV 回读，不重新标定相机。原始测试图 SHA-256 已核验。",
        "",
        "| 测试对 | 左拟合/px | 右拟合/px | 双目拟合/px | 留出角点/px | 边长 RMSE/mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pair_id']} | {r['left_rms_px']:.4f} | {r['right_rms_px']:.4f} | "
            f"{r['joint_rms_px']:.4f} | {r['held_out_corner_rms_px']:.4f} | "
            f"{r['edge_rmse_mm']:.4f} |"
        )
    lines += [
        "",
        f"合并双目拟合 RMS：{summary['pooled_joint_rms_px']:.4f} px；"
        f"合并留出角点 RMS：{summary['pooled_held_out_corner_rms_px']:.4f} px。",
        "",
        "留出验证：按角点索引模 5 分组，每轮用约 80% 点拟合位姿，"
        "预测其余点；左右同一角点共同留出。相机参数冻结。"
        "这不是重新执行相机标定交叉验证，也不排除同图角点的相关误差。",
        "",
        "边长误差是三角化相邻点与 19 mm 的几何一致性，不是位姿平移误差。"
        "没有外部位姿真值，绝对平移误差（mm）和旋转误差（度）均无法确定。",
        "",
        "点击 index.html 查看检测点（绿圈）与 CSV 重投影点（红十字），"
        "以及原有姿态坐标轴。各项细节保存在 metrics.csv。",
    ]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>测试姿态验证</title>
<style>body{font:16px/1.8 system-ui;max-width:1500px;margin:30px auto;padding:20px;
background:#f5f7fa}.pair{display:flex;gap:15px}figure{flex:1;margin:0;min-width:0}
img{width:100%}a{color:#1454b8}@media(max-width:850px){.pair{display:block}}</style>
<h1>测试姿态：CSV 回读与原图重投影</h1><p>绿圈：检测点；红十字：CSV 中相机参数和位姿的重投影。
点击图片查看原分辨率。没有外部真值，本页展示拟合与几何一致性。</p>
<p><a href="report.md">精度报告</a> · <a href="metrics.csv">逐对指标</a></p>"""
    (destination / "index.html").write_text(page + "".join(sections) + "</html>", encoding="utf-8")


if __name__ == "__main__":
    main()
