# 数据格式与几何约定

## 原图与观测

文件命名为 `{left|right}_pair_六位ID.bmp`，位于各数据集的 left/right 子目录。
标定与测试 ID 不跨数据集唯一，外部汇总需同时保留数据集名称。
原图尺寸本组为 1280×720，灰度；完整棋盘 12×9 内角点，行优先共 108 点。

## 标定 CSV

字段为 `group,parameter,row,col,value,unit,convention`，矩阵索引从 0 开始。

| 参数 | 形状或含义 | 单位 |
|---|---|---|
| K_left / K_right | 3×3，焦距与主点 | 前两行 px；最后一行无量纲 |
| D_left / D_right | 五参数，形状由 CSV 索引恢复 | 无量纲 |
| R_RL | 3×3，左相机坐标变换到右相机 | 无量纲 |
| t_RL | 3×1，同一变换的平移 | mm |
| F | 3×3 基础矩阵，归一化不唯一 | 仅几何辅助，不作长度解释 |
| baseline_mm | norm(t_RL) | mm |
| stereo_rms_px | 最终标定拟合 RMS | px |
| image_width / image_height | 原图尺寸 | px |
| columns / rows / retained_pair_count | 规格与数量 | 计数 |
| square_mm | 格距 | mm |

D 顺序为 k1、k2、p1、p2、k3。索引缺失、重复、非有限值不能用于 CSV 可视化回读。

## 姿态 CSV

一行一个预期测试 ID。`status=ok` 表示求解成功，失败保留 `failure_reason` 且必要数值留空。
`rx_rad,ry_rad,rz_rad` 是 Rodrigues 旋转向量，不是欧拉角；`r11` 至 `r33` 表示同一旋转矩阵。
`tx_mm,ty_mm,tz_mm` 为板原点在原始左相机系的位置。

```text
P_R = R_RL @ P_L + t_RL
P_L = R_LB @ P_B + t_LB
```

相机 X 右、Y 下、Z 前。板 X 沿列、Y 沿行、Z 依右手规则；原点按每帧编号确定，不保证跨帧物理唯一。

## 指标

- RMS：sqrt(mean(dx²+dy²))，按二维角点计数；左右合并含双方观测。
- P0 右图预测：只依据左图误差选择 IPPE 解后预测右图；不是双目联合拟合误差。
- P1 双目拟合：固定相机，只优化板的六自由度。
- 留出角点 RMS：每轮约 80% 点拟合位姿，预测余下 20%，左右同一点一同留出。
- edge_rmse_mm：三角化水平/垂直相邻边与 19 mm 的误差，不是平移真值误差。
- ambiguity_flag：不同候选优化解的拟合分数接近时提示；不保证发现所有对称性歧义。

CSV 使用 UTF-8 BOM，JSON 使用 UTF-8。浮点小数位数不代表有效测量精度。
