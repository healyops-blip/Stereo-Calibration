# 配置与依赖

| 名称 | 使用者 | 来源/范围 | 更改影响 |
|---|---|---|---|
| board.columns / rows | 检测、三维网格 | config.yaml，本地 | 改变完整板规格；应重新标定 |
| board.square_mm | 标定、位姿、边长检查 | config.yaml，mm | 决定尺度；尺寸错误会系统性影响平移 |
| calibration_dir | 阶段一 | 相对配置所在目录 | 标定图来源 |
| pose_dir | 阶段二 | 相对配置所在目录 | 测试图来源 |
| output_dir | 主流程 | 相对配置所在目录 | 新建或覆盖输出；与输入隔离 |
| excluded_ids | 阶段一 | config.yaml | 当前预排除 000015、000016、000027 |
| expected_test_ids | 阶段二与主验收 | config.yaml | 要求保留的测试 ID 与顺序 |
| seed | OpenCV、分折、重采样 | config.yaml | 控制随机性；bootstrap 使用 seed+1 |
| folds | 标定交叉验证 | config.yaml，默认 5 | 独立组不足时失败 |
| bootstrap_samples | 稳定性实验 | config.yaml，默认 20 | 成功和失败均记数 |
| near_duplicate_rms_px | 近重复分组 | config.yaml，默认 5 px | 控制图像对是否同组 |
| --output-dir | 两个可视化/专项命令 | 命令行，默认项目 outputs | 指向已有结果目录 |
| --with-results | check_quality.py | 命令行开关 | 增加默认配置对应的 CSV 验证 |

不使用环境变量存储密钥，也没有客户端/服务器密钥。密钥轮换和前端打包检查不适用。
姿态专项脚本当前按题目固定检查 000001–000005；它并非任意测试 ID 的通用批处理入口。

交付前检查：Python 3.11 环境可用；按 requirements-lock.txt 安装；pip check 通过；路径指向预期数据；相机和位姿属于同一次实验。自定义输出需在专项命令中传入相应 --output-dir。
