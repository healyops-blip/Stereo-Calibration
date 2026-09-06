# 双目相机标定与棋盘格姿态估计

代码仓库：[healyops-blip/Stereo-Calibration](https://github.com/healyops-blip/Stereo-Calibration)。
**dataset/ 不上传 GitHub**，原始数据需在本地放置；outputs/ 同样不跟踪。
新克隆后运行 `git config core.hooksPath .githooks` 启用提交与推送拦截。
仓库防误提交与 CI 配置见 [仓库规范](docs/repository.md)。

按三个阶段交付：**标定的准确程度 → 测试数据姿态估计的准确程度 → 代码与说明文档规范程度**。
采用 OpenCV 针孔五参数模型与 SciPy 双目位姿优化，无需训练网络。
入口为 `run_pipeline.py`；实测报告生成在 `outputs/quality/quality_report.md`。

交付文档入口：[架构与文档导航](docs/architecture.md)。
第三阶段验收标准与交付清单见 [交付验收](docs/acceptance.md)，
指标和 CSV 字段见 [数据约定](docs/data_contracts.md)，
已有测试与缺口见 [测试覆盖](docs/tests.md)。

## 环境与快速运行

已在本项目 `.venv` 配置并验证 Python **3.11.15**，macOS arm64。
进入项目根目录后：

```bash
source .venv/bin/activate
python run_pipeline.py all
python -m pytest -q
```

新机器重建环境：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_pipeline.py all
```

`requirements.txt` 锁定运行依赖，`requirements-dev.txt` 增加开发检查工具，
`requirements-lock.txt` 固定本次验证环境的全部依赖。
只安装 `opencv-python-headless`，不要同时安装其他 OpenCV Python 发行包。
流程直接生成图像文件，不依赖 `imshow` 或 GUI。
随机种子固定，OpenCV 使用单线程；不同平台仍可能有浮点末位差异。

## VS Code 运行配置

直接用 VS Code 打开本目录，安装推荐的 Python、Python Debugger、Ruff 扩展。
`.vscode/settings.json` 默认指向 `.venv/bin/python`；若工作区曾缓存其他解释器，
使用“Python: Select Interpreter”选择本目录虚拟环境。
`.vscode/launch.json` 中的每项也显式指定该解释器。

在“运行和调试”选择以下入口并按 F5：

| 配置 | 内容 | 前置条件 |
|---|---|---|
| 阶段一：标定与交叉验证 | 完整检测、C0/C2 五折验证、20 次重采样、冻结参数 | 原始标定数据 |
| 阶段二：五对测试姿态估计 | 固定相机，求解测试板位姿 | 先完成阶段一 |
| 阶段三：CSV 回读与结果验证 | 验证输出、重算投影、检查模型哈希 | 先完成阶段一、二 |
| 阶段三：单元测试 pytest | 合成真值与文件异常测试 | 无需运行实际数据 |
| 完整流程：三个阶段 | 顺序执行前三项 | 原始数据 |

完整质量检查通过“终端 → 运行任务 → 阶段三：完整质量检查”启动。
包含现有 CSV 的验收可在 F5 中选择“阶段三：完整交付验收”。
不要并行启动完整流程与单独阶段：它们写入同一个输出目录。

## 阶段一：标定与验证

### 查看 CSV 参数在训练原图上的重投影

```bash
python visualize_calibration.py
```

打开 `outputs/calibration/visualization/index.html`，逐对查看 23 对训练图。
也可在 VS Code 选择“阶段一：CSV 原图重投影可视化”按 F5。
绿色圆圈为检测角点，红色十字为预测角点；每侧提供原分辨率叠加图和 4 倍棋盘局部图。
放大图显示编号及黄色误差连线，不额外放大误差。
脚本只从 `calibration/calibration_results.csv` 还原相机参数与板规格，并核验原图哈希。
棋盘位姿由固定相机参数后的双目拟合获得，单独保存为 `training_board_poses.csv`。
逐角点观测、预测坐标和残差保存在 `reprojected_corners.csv`，逐图误差在 `image_errors.csv`。
这是训练集拟合可视化，不是交叉验证，也未使用测试集。

### 重新标定

```bash
python run_pipeline.py calibrate
```

板规格已用原图及完整编号叠加图确认：**12 列 × 9 行内角点，格距 19 mm**。
26 对标定图中，按原技术方案预排除 `000015`、`000016`、`000027`，保留原文件；
其余 23 对全部检测成功。SB 优先，传统检测与亚像素精化作为回退。
编号、失败原因、图像 SHA-256、重复内容、板覆盖率与清晰度写入清单。

C0 使用分别标定的内参并固定内参求双目外参；C2 从同一组单目初值联合优化。
同一图像对始终处于同一折，左右角点 RMS 距离均小于 5 px 的近重复图像连成组，
整组分折。分组只使用观测，不使用模型残差；不是基于采集时间的独立性证明。
每折仅在训练对上重新标定，并在留出对上用左图 IPPE 位姿预测右图角点。
选择聚合留出右图预测 RMS 较低的模型，随后用全部保留标定对重拟合。
本次 C2 胜出。不得用测试图结果选模型或更改筛选规则。

质量筛选遵循“提示后复核”，`median + 3×1.4826×MAD` 不自动触发删除。
本次无额外确定异常，C1 未实施额外删除；C2 与 C0 使用相同的 23 对。
这不是 C1 有效性实验，也不宣称复现 OutAw 或 Kalibr 的筛选算法。

20 次整对有放回重采样记录基线及左右焦距的波动与失败次数，
仅衡量经验稳定性，不作为外部绝对精度或严格置信区间。

## 阶段二：测试姿态与验证

专项验证命令：`python validate_pose_accuracy.py`（先完成姿态阶段）。
它从相机 CSV 与姿态 CSV 重算投影，核验测试原图哈希，并做固定相机的 5 折角点留出验证：
约 80% 角点求位姿，预测剩余 20%，左右同一角点共同留出。
结果位于 `outputs/pose/validation/report.md`；可视化入口为该目录 `index.html`，
含原图重投影、4 倍放大和姿态坐标轴链接。
VS Code 可选择“阶段二：姿态精度专项验证与可视化”。
留出角点误差不是外部真值位姿误差，同图观测仍可能存在相关系统误差。

```bash
python run_pipeline.py pose
```

只读取冻结的 `calibration/calibration.json`。平面 IPPE 提供候选解，P0 按左图误差选取，
因此 P0 的右图误差是跨相机预测指标。
P1 从候选出发，以双精度投影、固定相机参数，仅优化板的 6 个位姿参数。
使用未加权线性最小二乘，检查两相机正深度、收敛状态与候选歧义。
每角点 RMS 定义为 `sqrt(mean(dx² + dy²))`，不是每坐标 RMS。
优化后仍存在不同解且 RMS 差小于 0.05 px 时标记歧义；P0 的平面歧义另列。

测试 CSV 始终按配置的 5 个 ID 输出，单对失败保留状态和原因。
两阶段用模型 SHA-256 校验隔离：姿态求解不能修改相机参数。
相邻水平、垂直角点经过归一化去畸变及三角化后，与 19 mm 比较；不包含对角线。
极线误差在立体校正后计算，水平双目统计 y 差，垂直双目统计 x 差。

普通棋盘没有唯一可识别的物理原点。此实现采用每帧上方行左端角点为原点、
列向右和行向下的图像编号约定；适用本组图像，并通过低极线误差检查左右一致性。
大幅旋转或跨相机观察到不同面时应重新检查对应关系，不能无条件迁移该约定。
不保证不同帧使用同一物理角点为原点，不建议据此直接计算跨帧板运动轨迹。

## 阶段三：代码、格式与可复现性

```bash
python run_pipeline.py validate
python -m pytest -q
python check_quality.py --with-results
```

CSV 回读验证会还原相机矩阵，检查旋转向量与旋转矩阵一致，并重新计算实际观测上的投影误差。
统一检查覆盖全部 src、测试与四个命令入口，保存 outputs/quality/quality_checks.json；
任一工具失败时返回非零退出码。它是本地验收，不是已经配置的 CI 门禁。
测试覆盖已知合成位姿恢复、左右变换、RMS 分母、噪声下优化收益、三角化格距、
正深度和非法旋转、近重复分折、缺图与非连续编号、中文路径灰度图及失败行保留。
失败记录允许存在，但报告会明确列出；格式验证通过不等于所有图像姿态成功。

## 文件与坐标约定

结果总入口：[outputs/index.html](outputs/index.html)。目录按阶段整理：

```text
outputs/
  README.md / index.html    # 结果导航
  calibration/             # 标定 CSV、模型、清单、交叉验证
    corners/               # 标定角点编号图
    visualization/         # 原图重投影、放大图与浏览页面
  pose/                    # 测试姿态 CSV、观测与来源信息
    corners/               # 测试角点编号图
    axes/                  # 测试双侧姿态坐标轴
    validation/            # 姿态精度指标、报告与重投影页面
  quality/                 # 总报告、代码检查、回读及历史迁移证据
```

下面的路径均相对 outputs 根目录。--output-dir 仍传整个结果根目录，不传某个阶段子目录。

| 输出 | 用途 |
|---|---|
| `calibration/calibration_results.csv` | 正式标定长表：group, parameter, row, col, value, unit, convention |
| `calibration/calibration.json` | 冻结模型、规格、保留 ID、配置及依赖版本 |
| `pose/pose_results.csv` | 5 条姿态、旋转向量/矩阵、平移、P0/P1 误差、几何与歧义状态 |
| `calibration/pair_manifest.csv`, `pose/pose_manifest.csv` | 数据清单、预排除与检测失败原因 |
| `calibration/cross_validation.csv` | 每折每个留出对的误差，可核查无测试集泄漏 |
| `calibration/calibration_quality.csv` | 每对最终标定质量、极线与边长误差 |
| `calibration/calibration_validation.json` | 分折、对比、MAD 提示、20 次重采样原始结果 |
| `calibration/C0_calibration.json`, `calibration/C2_calibration.json` | 两种全数据拟合结果 |
| `*_observations.json` | 检测角点与原始路径，便于回读重算 |
| `pose/pose_provenance.json`, `quality/verification.json` | 冻结模型哈希与格式验证结果 |
| `calibration/corners/`, `pose/corners/`, `pose/axes/` | 全部成功检测的编号图与测试双侧坐标轴图 |
| `quality/quality_report.md` | 三阶段实测报告 |

左相机坐标：X 向右、Y 向下、Z 向前。
`P_R = R_RL @ P_L + t_RL`，`P_L = R_LB @ P_B + t_LB`。
平移与板尺寸为 mm，旋转向量为 rad，像素投影为 px。
K 的焦距与主点为 px，K 最后一行是无量纲齐次归一化项。
畸变顺序为 k1、k2、p1、p2、k3；不输出可能混淆旋转顺序的欧拉角。
坐标轴颜色由 OpenCV 约定：X 红、Y 绿、Z 蓝。

每次运行会更新 `config.yaml` 指定的输出目录，不修改原始 BMP。
希望保留多次实验时，复制配置文件（放在项目根目录）并修改 `output_dir`，然后使用
`python run_pipeline.py all --config other_config.yaml`。相对路径以配置文件所在目录解析。
`outputs/` 被 Git 忽略；交付时应另行打包该目录和代码，勿打包 `.venv`。

## 方法来源与边界

思想参考为原 `技术方案.md` 中的 Zhang 平面标定、Kalibr 多相机约束、OutAw 质量审计、
最优姿态引导的观测多样性与留出验证思想；具体来源沿用技术方案中的参考文献。
调用库为 OpenCV、NumPy、SciPy、PyYAML。本项目实现独立编写，没有移植第三方仓库源码。

当前证据支持亚像素重投影及几何一致性。没有外部真实位姿、基线测量或标定板计量报告，
**不能将像素误差或边长一致性解释为绝对位置精度达到某个毫米数值**。
算法和数值阈值对本组数据验证有效，换镜头、板规格或采集条件后应重新验证。
