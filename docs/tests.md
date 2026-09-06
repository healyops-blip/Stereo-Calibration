# 测试覆盖与缺口

## 已有覆盖

以下均为仓库现有自动化测试，运行 `python -m pytest -q`。已配置 GitHub Actions 自动检查；分支保护是否强制执行另见仓库设置。

| 用例/规则 | 预期行为（含负例） | 证据 | 状态 |
|---|---|---|---|
| RMS 口径 | 3/4 px 残差按角点计算，不除以坐标数量 | test_geometry.py::test_rms_counts_points_not_coordinates；data_contracts.md | existing |
| 左右变换 | 独立三维投影一致，逆变换可恢复 | test_projection_transform_and_inverse | existing |
| 已知位姿 | 无噪声合成位姿恢复、误差接近零 | test_recover_known_stereo_pose | existing |
| 留出验证 | 已知几何上未参与拟合的点也正确预测 | test_held_out_corners_with_known_pose | existing |
| 优化收益 | 加噪后双目残差降低 | test_noisy_stereo_refinement_reduces_joint_residual | existing |
| 三角化 | 恢复 19 mm 边长与极线一致性 | test_triangulated_edge_scale_and_epipolar_error | existing |
| 物理合法性 | 非正深度与非法旋转被识别 | test_reject_improper_rotation_and_negative_depth | existing |
| 分折隔离 | 近重复观测不跨折 | test_near_duplicates_do_not_leak_between_folds | existing |
| 配对 | 非连续 ID 不漏行；缺右图记录原因 | test_dataset.py::test_noncontiguous_pair_ids_and_missing_side；flows.md | existing |
| 图像输入 | 中文路径灰度图正确读取 | test_gray_image_with_unicode_path | existing |
| 失败交付 | 五条失败记录保留，数值字段为空 | test_five_failure_rows_are_preserved | existing |
| CSV 合约 | 无 JSON 时还原相机；重复单元格被拒绝 | test_csv_visualization.py::test_csv_only_camera_roundtrip；data_contracts.md | existing |

实际图像检查由命令执行：`run_pipeline.py all`、两个可视化入口、`check_quality.py --with-results`。
它们验证本次数据运行与输出，不能替代未来环境中的自动回归。

## 建议增加的测试

| 用例 | 预期规则 | 类型 | 状态 |
|---|---|---|---|
| 配置边界 | 非正格距、非法折数、配置缺键有清楚报错 | 自动单元 | proposed |
| 输出中断 | 检测并拒绝混用不同批次产物 | 自动集成 | proposed |
| 标定板方向 | 大转角、翻转、左右编号不一致可识别 | 自动合成＋人工原图审查 | proposed |
| 文件写入失败 | 图像保存失败能显式反馈 | 自动单元 | proposed |

## 尚无充分验证的内容

| 优先级 | 缺口 | 影响 | 状态 |
|---|---|---|---|
| 高 | 外部姿态/基线真值与板计量 | 无法给出绝对 mm/度误差 | none |
| 中 | 中断时事务性输出和并发运行 | 可能存在部分新旧结果混合 | none |
| 中 | 所有平面/编号歧义的完整测试 | 换数据后方向约定可能失效 | none |
| 低 | Windows/Linux 与其他依赖版本 | 尚未证明跨平台可复现 | none |

上述缺口保留在交付说明，不把建议测试写成已经通过的测试。

## 仓库数据策略

`test_repository_policy.py` 的 10 个测试案例覆盖目录、BMP/压缩包拦截、源码放行，以及临时 Git 仓库中强制暂存数据后被拒绝。与原有 12 项合计 22 项。详见 repository.md。
