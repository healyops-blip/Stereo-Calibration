"""Shared original-image overlays; markers preserve actual residual scale."""

from typing import Any

import cv2
import numpy as np


def annotate(
    image: Any,
    observed: Any,
    predicted: Any,
    title: str,
    scale: int = 1,
    origin: tuple[int, int] = (0, 0),
) -> Any:
    """用真实残差尺度绘制观测、预测及局部放大标记。

    Args:
        image: 原图或裁块灰度图 (H, W)，不会原位修改。
        observed: 原图观测坐标 (N, 2)，px，绿色圆圈。
        predicted: 同序预测坐标 (N, 2)，px，红色十字。
        title: 绘制在画布上的标题。
        scale: 正整数放大倍数；大于一时增加连线和编号。
        origin: 裁块左上角在原图中的 (x, y) 坐标，px。

    Returns:
        BGR 叠加画布；坐标先减 origin 再乘 scale。

    Raises:
        ValueError: 观测与预测点数不一致。
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for index, (actual, expected) in enumerate(zip(observed, predicted, strict=True)):
        a = tuple(np.rint((actual - origin) * scale).astype(int))
        b = tuple(np.rint((expected - origin) * scale).astype(int))
        cv2.circle(canvas, a, 4 if scale == 1 else 6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas, b, (0, 0, 255), cv2.MARKER_CROSS, 6 if scale == 1 else 9, 1, cv2.LINE_AA
        )
        if scale > 1:
            cv2.line(canvas, a, b, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index),
                (a[0] + 7, a[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 220, 80),
                1,
                cv2.LINE_AA,
            )
    banner = np.zeros((60, canvas.shape[1], 3), np.uint8)
    cv2.putText(banner, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        banner,
        "Green circle: detected | Red cross: CSV reprojection",
        (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return np.vstack((banner, canvas))
