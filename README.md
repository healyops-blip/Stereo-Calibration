# Stereo Camera Calibration

基于 OpenCV 的双目标定与棋盘格姿态估计。棋盘为 **12×9 内角点，格距 19 mm**。
使用冻结的相机参数估计 5 对测试图像的棋盘姿态。

## 交付清单

正式交付由仓库代码、依赖说明和 `delivery/` 中的四个文件组成。

| 要求 | 交付位置 | 内容 |
|---|---|---|
| 可运行代码与依赖说明 | `src/`、根目录 CLI、`config.yaml`、下述依赖文件 | 双目标定、姿态估计、验证与可视化 |
| 双目标定结果 CSV | [calibration_results.csv](delivery/calibration_results.csv) | 内参、畸变、双目外参及单位，共 57 行参数记录 |
| 5 个测试姿态放在一个 CSV | [pose_results.csv](delivery/pose_results.csv) | 编号 000001–000005，每对一行，共 5 行，均为 `ok` |
| 技术文档 HTML 版 | [technical_manual.html](delivery/technical_manual.html) | 浏览器阅读，含图表、验证方法和参考文献 |
| 技术文档 PDF 版 | [technical_manual.pdf](delivery/technical_manual.pdf) | 同一报告的固定版式版本。 |


## 目录与提交范围

```text
Stereo Calibration/
├── README.md                     # 交付与运行说明
├── config.yaml                   # 棋盘、数据路径和运行参数
├── requirements.txt              # 运行依赖
├── requirements-dev.txt          # 检查与回归测试依赖
├── requirements-lock.txt         # 已验证的完整版本锁定
├── run_pipeline.py               # 正式流程入口
├── visualize_calibration.py      # 标定结果可视化
├── validate_pose_accuracy.py     # 姿态验证
├── check_quality.py              # 正式交付范围质量检查
├── src/                          # 正式算法模块；传统对比实验被忽略
│   └── deeplearning/             # 训练、数据准备、推理与配套评估代码
├── tests/                        # 正式模块与深度学习模块的回归测试
├── scripts/check_repository.py   # 提交范围检查
├── delivery/                     # 两个 CSV + HTML/PDF 技术报告
├── .github/、.githooks/           # CI 与提交保护
├── dataset/                      # 仅本地：原始数据与题目 PDF
├── outputs/、output/、tmp/        # 仅本地：生成结果与临时文件
└── docs/、Stereo Calibration Reference/  # 仅本地：内部文档与参考论文
```



## 安装

要求 Python 3.11。锁定依赖曾在 macOS arm64 / Python 3.11.15 验证，跨平台仍需实际检验。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

仅运行算法可改用 `requirements.txt`；执行质量检查则安装锁定依赖。
传统标定和姿态流程不需要 PyTorch、CUDA、Sphinx 或 Matplotlib。

### 深度学习训练

训练入口为 `src/deeplearning/train.py`，网络定义、损失和权重保存/加载代码均提交，
但保存出来的权重文件不提交。额外依赖是与目标 GPU/驱动兼容的 CUDA 版 PyTorch；
`requirements-lock.txt` 不包含 PyTorch，也不是 GPU 训练环境锁定文件。
当前训练实现要求 NVIDIA CUDA，不支持直接在 macOS CPU/MPS 上运行训练。

配置好独立训练环境及本地数据后，在仓库根目录执行：

```bash
python -m src.deeplearning.prepare --root . --output outputs/deeplearning/prepared --scenes 2000
python -m src.deeplearning.train --prepared outputs/deeplearning/prepared --output outputs/deeplearning/run --epochs 30
```

两个输出目录均须尚不存在。准备阶段生成场景描述，训练阶段渲染样本并从头训练，
不需要随仓库提供预训练权重。训练日志、检查点和生成数据保存在忽略的 `outputs/` 中。
推理代码可提交，但运行推理需要自己训练或另行取得可信权重。
`check_quality.py` 包含深度学习代码检查；缺少 PyTorch 时相关运行测试会跳过，
检查通过不等同于 GPU 训练已验证。

CI 另设 PyTorch CPU 回归任务，固定 PyTorch 2.8.0，检查网络输出、损失反向传播、
权重保存/加载与失败保护；先强制导入 PyTorch，避免缺少依赖时整组静默跳过。
安装方法参照 [PyTorch 官方版本说明](https://docs.pytorch.org/get-started/previous-versions/)。
该版本用于回归测试，不声明与历史 GPU 训练环境完全相同。
CUDA 检查独立标记，可在具备环境的机器执行：

```bash
python -m pytest -q tests/test_dl_training.py -m 'not cuda'  # CPU 回归
python -m pytest -q tests/test_dl_training.py -m cuda        # CUDA 推理检查
```

## 数据目录

将本地数据放入以下目录，文件不会被 Git 跟踪：

```text
dataset/
├── calibration_images/
│   ├── left/left_pair_000001.bmp
│   └── right/right_pair_000001.bmp
└── pose_test_images/
    ├── left/left_pair_000001.bmp
    └── right/right_pair_000001.bmp
```

数据由接收方另行提供，其他图像按相同编号规则命名，左右按编号配对。
棋盘规格、数据路径、排除编号和运行参数在 [config.yaml](config.yaml) 中配置。

## 运行

```bash
python run_pipeline.py calibrate  # 双目标定
python run_pipeline.py pose       # 位姿估计
python run_pipeline.py validate   # 结果检查
python run_pipeline.py all        # 完整流程
python visualize_calibration.py --output-dir outputs
python validate_pose_accuracy.py --output-dir outputs
python check_quality.py          # 无原始数据也可运行的基础检查
python check_quality.py --with-results  # 另需完整 outputs/ 正式产物
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

运行结果默认写入 `outputs/`，不会自动覆盖 `delivery/` 中的交付快照。
独立运行 `validate` 需要完整流程生成的 JSON、观测记录和来源哈希，
不能只凭两个交付 CSV 执行全部审计。

## 评估口径

- **标定准确程度**：训练重投影 RMS、分组交叉验证右图预测 RMS、极线误差和三角化边长一致性。
- **姿态估计准确程度**：固定相机下的双目拟合、左图估姿后预测右图、留出角点预测。
- **代码和文档规范**：函数接口说明、基础回归测试、Ruff、mypy、Bandit、依赖检查及技术报告。

姿态为棋盘到原始左相机的变换；旋转向量单位 rad（不是欧拉角），平移单位 mm，残差单位 px。

- 程序不会修改原始图像。
- 相机参数在标定完成后冻结，测试数据不参与重新标定。
- 未使用外部真值设备时，输出指标仅表示模型内部一致性。

## 提交前检查

```bash
git config core.hooksPath .githooks
python scripts/check_repository.py
python check_quality.py
git status --short
```

`.gitignore` 防止普通误加入；钩子还会拒绝强制加入的数据、实验文件和非白名单交付文件。
钩子需执行上述配置命令才能启用，CI 同样检查提交范围。本文不表示已完成提交或推送。
历史检查继续禁止原始数据和生成产物；新增的本地实验/编辑器配置禁入规则只检查当前索引，
不追溯删除旧版本中的开发配置，也不改写 Git 历史。
