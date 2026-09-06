"""Generate the three-stage report from recorded measurement artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.export import read_csv, read_model


def write_result_index(output: Path) -> None:
    """生成交付输出目录的 Markdown 与 HTML 导航。

    Args:
        output: 已存在的输出根目录，README.md 和 index.html 将被覆盖。

    Note:
        扫描已有实验 report.md；固定阶段链接可能尚无对应产物。
    """
    links = [
        ("标定参数 CSV", "calibration/calibration_results.csv"),
        ("标定原图可视化", "calibration/visualization/index.html"),
        ("测试姿态 CSV", "pose/pose_results.csv"),
        ("测试姿态可视化", "pose/validation/index.html"),
        ("测试姿态精度报告", "pose/validation/report.md"),
        ("三阶段质量报告", "quality/quality_report.md"),
        ("代码验收证据", "quality/quality_checks.json"),
    ]
    for path in sorted((output / "experiments").glob("*/report.md")):
        links.append((f"标定改进实验：{path.parent.name}", str(path.relative_to(output))))
    description = (
        "calibration/：标定参数、观测与可视化；pose/：测试姿态、坐标轴与验证；"
        "quality/：验收证据；experiments/：独立改进实验。"
    )
    markdown = ["# 结果导航", "", description, "", "运行对应阶段后，以下文件才会生成。", ""]
    markdown += [f"- [{label}]({path})" for label, path in links]
    (output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    items = "".join(f'<li><a href="{path}">{label}</a></li>' for label, path in links)
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>结果导航</title>'
        "<style>body{font:18px/2 system-ui;max-width:950px;margin:50px auto;"
        "padding:20px;background:#f5f7fa}a{color:#1454b8}</style>"
        f"<h1>双目标定与姿态估计 · 结果导航</h1><p>{description}</p>"
        f"<p>运行相应阶段后生成对应结果。</p><ul>{items}</ul></html>"
    )
    (output / "index.html").write_text(page, encoding="utf-8")


def report(output: Path) -> None:
    """从保存的标定、位姿及验证数据生成三阶段质量报告。

    Args:
        output: 含正式标定产物的交付根目录。

    Note:
        只有位姿来源哈希匹配当前相机模型时才纳入姿态指标。
        写 quality/quality_report.md 并更新目录导航，不重算标定或训练模型。
    """
    (output / "quality").mkdir(parents=True, exist_ok=True)
    model = read_model(output / "calibration/calibration.json")
    validation = json.loads(
        (output / "calibration/calibration_validation.json").read_text(encoding="utf-8")
    )
    lines = [
        "# 三阶段实测质量报告",
        "",
        "## 一、标定的准确程度",
        "",
        f"原始 {validation['inventory']} 对 → 预排除 {validation['preexcluded']} 对 → "
        f"检测成功 {validation['detected']} 对 → 最终保留 {validation['retained']} 对。",
        "",
        "已知损坏帧按技术方案预排除，未自动删除其他高误差帧。C1 暂未执行额外筛选；"
        "C2 与 C0 使用相同数据。模型仅依照标定集 5 折留出右图预测 RMS 选择。",
        "",
        "| 模型 | 留出右图预测 RMS/px | 留出平均极线误差/px |",
        "|---|---:|---:|",
    ]
    for name, metrics in validation["cv"].items():
        lines.append(
            f"| {name} | {metrics['right_prediction_rms_px']:.4f} | "
            f"{metrics['mean_epipolar_px']:.4f} |"
        )
    lines += [
        "",
        f"冻结模型：{model['selected_candidate']}；最终双目拟合 RMS "
        f"{model['stereo_rms_px']:.4f} px；基线 {model['baseline_mm']:.4f} mm。",
        "",
        f"MAD 复核提示：{validation['outlier_review']['flagged_ids']}；仅提示，不自动删除。",
        "",
        f"Bootstrap：{validation['bootstrap']['succeeded']} 次成功，"
        f"{validation['bootstrap']['failed']} 次失败；整对有放回抽样。",
        "",
        "## 二、测试数据姿态估计的准确程度",
        "",
    ]
    provenance_path = output / "pose/pose_provenance.json"
    current_pose = (
        provenance_path.exists()
        and json.loads(provenance_path.read_text(encoding="utf-8"))["calibration_sha256"]
        == hashlib.sha256((output / "calibration/calibration.json").read_bytes()).hexdigest()
    )
    if current_pose and (output / "pose/pose_results.csv").exists():
        lines += [
            "| ID | 状态 | 左图拟合/px | 右图拟合/px | 双目拟合/px | 左目位姿右图预测/px |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in read_csv(output / "pose/pose_results.csv"):
            metrics = [
                f"{float(row[key]):.4f}" if row.get(key) else "—"
                for key in (
                    "left_rms_px",
                    "right_rms_px",
                    "joint_rms_px",
                    "p0_right_prediction_rms_px",
                )
            ]
            lines.append(f"| {row['pair_id']} | {row['status']} | " + " | ".join(metrics) + " |")
    else:
        lines += ["当前冻结模型的姿态阶段尚未执行；旧姿态结果不纳入本报告。"]
    lines += [
        "",
        "重投影与 19 mm 相邻边长用于评价拟合和几何一致性。没有外部位姿、基线真值，"
        "不能据此报告绝对毫米级位置精度；Bootstrap 也不是绝对误差或严格置信区间。",
        "",
        "棋盘原点按每帧图像上方行的左端点约定，不保证跨帧为同一物理角点。"
        "姿态位于原始左相机坐标系；右相机变换为 P_R = R_RL @ P_L + t_RL。",
        "",
        "## 三、代码和说明文档的规范程度",
        "",
        "运行 `python run_pipeline.py validate` 回读 CSV 并重算投影；"
        "运行 `python -m pytest` 执行单元测试。静态检查见 README。",
        "",
        "详细证据：cross_validation.csv、calibration_quality.csv、"
        "calibration_validation.json、verification.json；"
        "编号图位于 calibration/corners/ 与 pose/corners/，坐标轴图位于 pose/axes/。",
    ]
    quality = read_csv(output / "calibration/calibration_quality.csv")
    edge_rmse = float(np.sqrt(np.mean([float(r["edge_rmse_mm"]) ** 2 for r in quality])))
    lines += ["", f"最终标定对三角化相邻边长 RMSE：{edge_rmse:.4f} mm（内部几何一致性）。"]
    if current_pose and (output / "quality/verification.json").exists():
        verified = json.loads((output / "quality/verification.json").read_text(encoding="utf-8"))
        lines += [
            "",
            f"最近一次输出验证：{verified['status']}；"
            f"CSV 共 {verified['csv_rows']} 行，成功姿态 {verified['successful_poses']} 条。",
        ]
    (output / "quality/quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_result_index(output)
