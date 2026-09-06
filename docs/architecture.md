# 架构与交付导航

本项目是本地 Python 命令行工具：从普通棋盘双目图像求相机参数，再在固定相机参数下求测试棋盘位姿。没有模型训练、在线服务或数据上传。

## 模块职责

| 层次 | 文件 | 职责 |
|---|---|---|
| 命令入口 | 根目录三个运行脚本 | 调用包内 main，不承载几何或读写实现 |
| 流程编排 | `src/pipeline.py` | 标定、模型选择、冻结、测试姿态与报告调度 |
| 数据 | `src/dataset.py`, `src/corners.py` | ID 配对、灰度读取、角点检测、清单与编号 |
| 几何 | `src/calibration.py`, `src/pose.py` | 单双目标定、IPPE、固定相机的位姿优化 |
| 算法验证 | `src/validation.py` | 五折验证、重采样、三角化、极线、留出角点 |
| 输出格式 | `src/export.py` | CSV/JSON、CSV 相机回读、位姿行、坐标轴 |
| 输出验收 | `src/artifact_validation.py` | CSV 矩阵回读、重投影重算、模型哈希一致性 |
| 报告 | `src/reporting.py` | 根据实测文件生成三阶段报告 |
| 可视化 | `src/visualization.py` | 共用角点标记与误差显示 |
| 专项流程 | `src/calibration_visualization.py`, `src/pose_accuracy.py` | 训练图和测试图可视化、测试精度专项报告 |
| 开发验收 | `check_quality.py` | 执行静态检查、测试、依赖检查并记录结果 |

运行依赖：Python 3.11、OpenCV、NumPy、SciPy、PyYAML；锁定版本见 `requirements-lock.txt`。公共函数置于 `src`，不从其他命令入口导入实现。

## 假设与已知限制

- `config.yaml` 规定 12×9 内角点、19 mm 格距。度量尺度依赖真实板尺寸，未验证标定板制造公差或平整度。
- `src/corners.py:detect` 使用每帧图像方向统一编号，不提供跨帧唯一物理原点保证。
- `src/pipeline.py:stage_calibration` 仅比较 C0/C2；没有已确认新增异常，因此未执行 C1 额外剔除。
- `src/validation.py:grouped_folds` 按角点距离近似分组，不保证采集时间上的完全独立。
- `src/artifact_validation.py:verify` 通过代表格式与数值一致，不等于外部物理精度达标；失败姿态行允许保留并计数。
- `Any` 仍用于 OpenCV/NumPy 的动态矩阵字典；MyPy 通过不等于严格的矩阵形状静态证明。
- 输出非事务性写入；运行中断可能产生部分新旧文件。不得并发运行写同一目录的命令；需要保留实验时使用不同输出目录。

无登录、会话、角色令牌或远程数据库；本地文件访问服从操作系统权限。输入边界为原图、配置、CSV/JSON，具体读写见 flows.md 和 permissions.md。

无邮件通知，无 emails.md；无定时任务，无 cron.md；无公开站点，无 seo.md；无嵌入式 AI 或外部自动化，无 automation.md。

## 相关文档

- [仓库规范与数据保护](repository.md)

- [操作与数据流](flows.md)
- [文件访问与写入范围](permissions.md)
- [配置与依赖](variables.md)
- [测试覆盖与缺口](tests.md)
- [CSV 与坐标约定](data_contracts.md)
- [代码维护规范](coding_standard.md)
- [交付验收](acceptance.md)
- [用户运行说明](../README.md)
