"""Visualize training-image reprojection using camera parameters read ONLY from CSV."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from src.corners import object_points
from src.dataset import read_gray
from src.export import camera_from_csv, pose_row, write_csv, write_json
from src.pose import estimate, project, rms
from src.visualization import annotate


def main() -> None:
    """从交付相机 CSV 回读参数，生成标定原图叠加与放大 HTML。

    CLI 参数 --output-dir 指定交付根目录。核验原图哈希及分辨率，
    再用该帧角点重新求位姿；写入 calibration/visualization 下的图像与统计。
    不改原图或相机参数；同名可视化文件会覆盖，像素拟合不代表外部精度。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs"
    )
    args = parser.parse_args()
    source = args.output_dir.resolve()
    destination = source / "calibration/visualization"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = source / "calibration/calibration_results.csv"
    model = camera_from_csv(csv_path)
    points = object_points(model["board"])
    observations = json.loads(
        (source / "calibration/calibration_observations.json").read_text("utf-8")
    )
    with (source / "calibration/pair_manifest.csv").open(encoding="utf-8-sig") as handle:
        manifest = {r["pair_id"]: r for r in csv.DictReader(handle)}
    rows, poses, corner_rows, sections, previews = [], [], [], [], []
    for obs in observations:
        pair_id = obs["pair_id"]
        for side in ("left", "right"):
            obs[side] = np.array(obs[side], dtype=np.float64)
            path = Path(obs["paths"][side])
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[pair_id][f"{side}_sha256"]:
                raise ValueError(f"Original image changed: {path}")
        fit = estimate(points, obs, model)
        poses.append(pose_row(pair_id, fit))
        predictions = project(points, fit["pose"], model)
        images = []
        figures = []
        for side, predicted in zip(("left", "right"), predictions, strict=True):
            observed = obs[side].reshape(-1, 2)
            errors = np.linalg.norm(predicted - observed, axis=1)
            image = read_gray(Path(obs["paths"][side]))
            if (image.shape[1], image.shape[0]) != model["size"]:
                raise ValueError("Image size differs from CSV")
            metric = dict(
                pair_id=pair_id,
                side=side,
                rms_px=rms(predicted - observed),
                p95_px=float(np.percentile(errors, 95)),
                max_px=float(errors.max()),
            )
            rows.append(metric)
            title = (
                f"{pair_id} {side} | RMS {metric['rms_px']:.4f} px | max {metric['max_px']:.3f} px"
            )
            overlay = annotate(image, observed, predicted, title)
            name = f"{pair_id}_{side}"
            cv2.imwrite(str(destination / f"{name}.png"), overlay)
            lower = np.maximum(np.floor(observed.min(axis=0)).astype(int) - 20, 0)
            upper = np.minimum(np.ceil(observed.max(axis=0)).astype(int) + 20, model["size"])
            crop = image[lower[1] : upper[1], lower[0] : upper[0]]
            zoom = annotate(
                crop, observed, predicted, title, scale=4, origin=(int(lower[0]), int(lower[1]))
            )
            cv2.imwrite(str(destination / f"{name}_zoom.png"), zoom)
            images.append(cv2.resize(overlay, (640, 390)))
            figures.append(
                f'<figure><a href="{name}.png"><img loading="lazy" '
                f'src="{name}.png"></a><figcaption>{side} · '
                f"RMS {metric['rms_px']:.4f} px · "
                f'<a href="{name}_zoom.png">4 倍局部放大</a></figcaption></figure>'
            )
            for index, (actual, expected) in enumerate(zip(observed, predicted, strict=True)):
                corner_rows.append(
                    dict(
                        pair_id=pair_id,
                        side=side,
                        corner_id=index,
                        observed_x_px=actual[0],
                        observed_y_px=actual[1],
                        projected_x_px=expected[0],
                        projected_y_px=expected[1],
                        dx_px=expected[0] - actual[0],
                        dy_px=expected[1] - actual[1],
                        error_px=errors[index],
                    )
                )
        combined = np.hstack(images)
        cv2.imwrite(str(destination / f"{pair_id}_pair.jpg"), combined)
        previews.append(combined)
        sections.append(
            f'<section id="p{pair_id}"><h2>标定对 {pair_id}</h2><div class="pair">'
            + "".join(figures)
            + "</div></section>"
        )
        print(f"CSV reprojection: {pair_id}", flush=True)
    if not rows:
        raise ValueError("No training observations available")
    write_csv(destination / "image_errors.csv", rows)
    write_csv(destination / "reprojected_corners.csv", corner_rows)
    write_csv(destination / "training_board_poses.csv", poses)
    write_json(
        destination / "provenance.json",
        dict(
            camera_source=str(csv_path),
            camera_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            observation_source="calibration/calibration_observations.json",
            pairs=len(observations),
            purpose="training_fit_visual_inspection_not_held_out_validation",
            pose_method="fixed_CSV_camera_parameters_stereo_six_DOF_fit",
            original_image_hashes_verified=True,
        ),
    )
    cv2.imwrite(
        str(destination / "overview.jpg"),
        np.vstack([cv2.resize(preview, (768, 234)) for preview in previews]),
    )
    navigation = " · ".join(f'<a href="#p{o["pair_id"]}">{o["pair_id"]}</a>' for o in observations)
    excluded = ", ".join(k for k, v in manifest.items() if v["status"] != "ok")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>阶段一 · 标定 CSV 原图重投影检查</title>
<style>body{font:16px/1.7 system-ui;margin:30px auto;padding:0 20px;max-width:1500px;
background:#f5f7fa;color:#202936}h1{font-size:28px}section{background:white;padding:20px;
margin:24px 0;border-radius:12px}.pair{display:flex;gap:16px}figure{margin:0;flex:1;min-width:0}
img{width:100%}a{color:#1454b8}nav{line-height:2.2}.legend{padding:18px;background:#e5edf7}
@media(max-width:850px){.pair{display:block}}</style>
<h1>阶段一：标定 CSV → 原图重投影</h1>
<div class="legend">绿色圆圈：实际检测角点；红色十字：CSV 参数重投影角点。
点击图片查看原分辨率，点击“4 倍局部放大”检查棋盘区域与角点编号。
放大图的黄色线连接检测点与预测点，没有人为放大残差。</div>
<p>全部相机矩阵、畸变、图像尺寸及棋盘规格均从 calibration_results.csv 读取。
每张图的棋盘位姿在固定这些参数后求解，并另存 training_board_poses.csv。
这里检查标定训练集拟合，不是独立留出精度，也没有使用测试集。</p>
"""
    page += f"<p>有效标定对 {len(observations)} 组；排除：{html.escape(excluded)}。</p>"
    page += '<p><a href="image_errors.csv">逐图误差 CSV</a> · '
    page += '<a href="reprojected_corners.csv">逐角点坐标与残差 CSV</a></p>'
    page += f"<nav>{navigation}</nav>" + "".join(sections) + "</html>"
    (destination / "index.html").write_text(page, encoding="utf-8")
    print(f"Open {destination / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
