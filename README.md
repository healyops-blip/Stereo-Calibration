# Stereo Camera Calibration

基于 OpenCV 的双目相机标定与棋盘格姿态估计项目。系统从左右相机棋盘图像中检测角点，求解相机内参、畸变参数和双目外参，并在冻结相机参数后估计测试棋盘的六自由度位姿。

棋盘规格为 **12 × 9 个内角点，格距 19 mm**。项目另包含一个轻量全卷积角点检测网络，用于与 OpenCV 角点检测方法进行扩展对比。

## 主要功能

- 按样本编号配对左右图像并检查异常、重复和缺失数据
- 使用 OpenCV SB/classic 检测棋盘角点
- 完成单目标定、双目标定及参数联合优化
- 通过分组交叉验证、极线误差和三角化结果验证标定参数
- 在固定相机参数下估计测试棋盘位姿
- 导出标定参数、姿态结果和可视化结果
- 训练和评估深度学习角点检测模型

## 结果摘要

| 指标 | 结果 | 含义 |
|---|---:|---|
| 双目标定重投影 RMS | 0.1948 px | 全部有效标定图上的像素拟合误差 |
| 五折留出右图预测 RMS | 0.3013 px | 未参与该折标定的右图角点预测误差 |
| 留出校正极线误差 | 0.1150 px | 双目校正后对应角点的垂直偏差 |
| 三角化相邻边长 RMSE | 0.2748 mm | 重建棋盘相邻角点距离相对 19 mm 的偏差 |

上述结果用于评价模型的内部一致性。由于没有外部高精度测量设备，不能将其解释为相机参数或棋盘位姿的绝对精度。

## 安装

推荐使用 Python 3.11：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-lock.txt
```

`requirements.txt` 包含运行依赖，`requirements-dev.txt` 包含测试和静态检查依赖。深度学习训练还需要安装与本机 CUDA 环境兼容的 PyTorch。

## 数据目录

原始数据不随仓库提交。使用时按以下结构放置：

```text
dataset/
├── calibration_images/
│   ├── left/left_pair_000001.bmp
│   └── right/right_pair_000001.bmp
└── pose_test_images/
    ├── left/left_pair_000001.bmp
    └── right/right_pair_000001.bmp
```

图像通过编号配对，棋盘规格、数据路径和排除样本在 `config.yaml` 中配置。

## 运行

```bash
# 双目标定
python3 run_pipeline.py calibrate --config config.yaml

# 测试棋盘姿态估计
python3 run_pipeline.py pose --config config.yaml

# 结果验证
python3 run_pipeline.py validate --config config.yaml

# 执行完整流程
python3 run_pipeline.py all --config config.yaml
```

生成结果默认写入 `outputs/`。

深度学习扩展实验：

```bash
python3 -m src.deeplearning.prepare \
  --root . \
  --output outputs/deeplearning/prepared \
  --scenes 2000

python3 -m src.deeplearning.train \
  --prepared outputs/deeplearning/prepared \
  --output outputs/deeplearning/run \
  --epochs 30
```

## 核心代码

| 文件 | 作用 |
|---|---|
| `run_pipeline.py` | 命令行入口与任务调度 |
| `src/dataset.py` | 双目图像配对与读取 |
| `src/corners.py` | 棋盘角点检测与质量统计 |
| `src/calibration.py` | 相机内参、畸变和双目外参求解 |
| `src/pose.py` | 棋盘位姿估计 |
| `src/validation.py` | 交叉验证与几何指标计算 |
| `src/export.py` | CSV 和 JSON 结果导出 |
| `src/deeplearning/` | 合成数据、网络训练、推理与评估 |

## 交付文件

| 文件 | 内容 |
|---|---|
| `delivery/calibration_results.csv` | 双目相机标定参数 |
| `delivery/pose_results.csv` | 5 对测试图像的棋盘位姿 |
| `delivery/technical_manual.pdf` | 技术说明与实验报告 |

## 测试

```bash
python3 -m pytest -q
python3 check_quality.py
```

`check_quality.py` 执行 Ruff、mypy、Bandit、pytest 和依赖一致性检查。更详细的开发规范见 `CONTRIBUTING.md`。
