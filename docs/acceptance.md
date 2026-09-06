# 交付验收

## 本次整理范围

按 clean-code 提取公共读写、绘图、输出验证及报告模块；按 lint-and-validate 运行代码检查；按 shipping-artifacts 整理架构、流程、配置和实际测试覆盖。三个原命令入口继续保留。

统一质量入口：`python check_quality.py --with-results`。VS Code 选择“阶段三：完整交付验收”，或运行对应终端任务。

| 检查 | 验收方式 | 证据位置 |
|---|---|---|
| 代码风格与格式 | Ruff check、format --check | outputs/quality/quality_checks.json |
| 类型检查 | MyPy（允许动态矩阵 Any） | 同上 |
| 安全静态扫描 | Bandit 中高风险门槛 -ll | 同上 |
| 几何和数据测试 | pytest，22 项 | 同上 |
| 依赖一致性 | pip check | 同上 |
| CSV 回读和模型冻结 | verify、重算原始投影 | outputs/quality/verification.json |
| 标定数据实测 | 23 对有效、五折、20 次重采样 | outputs/calibration/calibration_validation.json |
| 测试姿态实测 | 5 对成功、留出角点 | outputs/pose/validation/report.md |
| 重构前后数值 | 相机、姿态和测试精度指标比对 | outputs/quality/refactor_regression.json |

Bandit 的 -ll 门槛不表示零提示：质量与数据策略脚本使用 subprocess 启动本地检查，会有五个低风险提示；未使用 shell=True，参数不来自外部任意命令输入。原始扫描输出保存在验收 JSON 中。

参考实测：标定拟合 RMS 0.1948 px；五折右图预测 0.3013 px；测试双目合并拟合 0.1792 px，留出角点 0.1814 px。它们不代表绝对位姿真值误差。

## 交付文件清单

- 源码：根目录四个 Python 入口、src/、tests/。
- 运行配置：config.yaml、pyproject.toml、三个 requirements 文件、.vscode/。
- 说明：README.md、技术方案.md、docs/。
- 实测产物：outputs/ 中正式 CSV/JSON、质量报告、两阶段可视化及验收证据。
- 原始数据：dataset/；如交付包不包含原数据，需要说明获取方式并维持原目录结构。

不交付 .venv、缓存和 .DS_Store；在接收环境使用 Python 3.11 与 requirements-lock.txt 重建环境。
outputs 被 Git 忽略，打包时应明确包含，不能只交付源码仓库就声称已交付结果。

## 接收者验收步骤

1. 依据 README 建立虚拟环境并安装锁定依赖。
2. 运行完整流程与两项可视化；不要同时写同一输出目录。
3. 运行统一质量入口，检查总状态与每项退出码。
4. 打开两个 index.html，抽查角点、误差与坐标轴；查看 tests.md 中未验证范围。

此次验证平台为 macOS arm64、Python 3.11.15。已配置 Ubuntu CI，远程运行结果以 Actions 为准；VS Code 配置已检查 JSON 与路径，未声称每个 F5 入口都经过 GUI 人工测试。
