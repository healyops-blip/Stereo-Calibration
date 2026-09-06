# Stereo Camera Calibration

双目相机标定与平面标定板位姿估计的 Python 实现。本仓库仅提供可运行源码，不包含笔试题目、原始数据、实验输出或内部技术文档。

## 安装

要求 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

## 数据目录

将本地数据放入以下目录，文件不会被 Git 跟踪：

```text
dataset/
├── calibration_images/
└── pose_test_images/
```

棋盘规格、数据路径和运行参数在 [config.yaml](config.yaml) 中配置。

## 运行

```bash
python run_pipeline.py calibrate  # 双目标定
python run_pipeline.py pose       # 位姿估计
python run_pipeline.py validate   # 结果检查
python run_pipeline.py all        # 完整流程
python -m pytest -q               # 单元测试
```

## 代码结构

| 文件 | 作用 |
|---|---|
| `run_pipeline.py` | 命令行入口 |
| `src/dataset.py` | 图像配对与读取 |
| `src/corners.py` | 棋盘角点检测与编号 |
| `src/calibration.py` | 相机内参与双目外参求解 |
| `src/pose.py` | 标定板位姿估计 |
| `src/validation.py` | 交叉验证与几何指标 |
| `src/export.py` | CSV 和 JSON 输出 |

运行结果默认写入 `outputs/`，该目录不上传仓库。

## 说明

- 程序不会修改原始图像。
- 相机参数在标定完成后冻结，测试数据不参与重新标定。
- 未使用外部真值设备时，输出指标仅表示模型内部一致性。
