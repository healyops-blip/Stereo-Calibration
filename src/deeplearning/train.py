"""Bounded synthetic corner-detector pilot; no real holdout images are read."""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.deeplearning.synthetic import heatmap, render_crop
from src.export import write_json

LOSS_BORDER_PX = 10
POSITIVE_LABEL_THRESHOLD = 0.01


def corner_network() -> nn.Sequential:
    """构造保持空间分辨率的轻量全卷积角点网络。

    Returns:
        接收 (B, 1, H, W) 灰度张量、输出同形 logits 的 Sequential。
        卷积核依次为 13、3、3、3、1、1；隐藏通道 32，尚未移至 GPU。
    """
    layers: list[nn.Module] = []
    channels = 1
    for kernel in [13, 3, 3, 3, 1]:
        layers.extend([nn.Conv2d(channels, 32, kernel, padding=kernel // 2), nn.ReLU()])
        channels = 32
    layers.append(nn.Conv2d(32, 1, 1))
    return nn.Sequential(*layers)


def balanced_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """分别归一化正负像素的 BCE，忽略十像素边缘。

    Args:
        logits: 浮点 Tensor (B,1,H,W)，空间大小须大于 20。
        target: 同形、同设备浮点高斯标签 [0,1]；大于 0.01 的位置归为正区域。

    Returns:
        可反向传播的标量损失。

    Note:
        对空正区/负区用计数下限保护；不等价于角点定位误差。

    Raises:
        ValueError: 形状、通道、设备或浮点类型不匹配。
    """
    if (
        logits.ndim != 4
        or logits.shape != target.shape
        or logits.shape[1] != 1
        or logits.shape[0] == 0
        or min(logits.shape[-2:]) <= 2 * LOSS_BORDER_PX
        or not logits.is_floating_point()
        or not target.is_floating_point()
        or logits.device != target.device
    ):
        raise ValueError(
            "Expected matching floating (B,1,H,W) tensors with H,W > 20; "
            f"got {logits.shape}, {target.shape}"
        )
    logits = logits[..., LOSS_BORDER_PX:-LOSS_BORDER_PX, LOSS_BORDER_PX:-LOSS_BORDER_PX]
    target = target[..., LOSS_BORDER_PX:-LOSS_BORDER_PX, LOSS_BORDER_PX:-LOSS_BORDER_PX]
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = target > POSITIVE_LABEL_THRESHOLD
    negative = ~positive
    return (loss * positive).sum() / positive.sum().clamp_min(1) + (
        loss * negative
    ).sum() / negative.sum().clamp_min(1)


def render_sample(job: tuple[dict[str, Any], dict[str, Any], int]) -> tuple[Any, Any, Any]:
    """由冻结场景生成一个含增强的单目训练裁块及标签。

    Args:
        job: (相机模型, 场景记录, 裁块边长)；场景含 seed、index、左右姿态和角点。

    Returns:
        uint8 灰度图、float32 高斯热图（均为 size×size）以及包含原点、
        浮点角点、侧别和种子的元数据。按索引交替左右，10% 为无棋盘负样本。

    Note:
        固定 OpenCV 单线程，随机性来自场景种子。
    """
    model, scene, size = job
    cv2.setNumThreads(1)
    rng = np.random.default_rng(scene["seed"] + 7919)
    side = "left" if scene["index"] % 2 == 0 else "right"
    pose = scene[side]
    points = np.asarray(pose["corners"])
    center = points[rng.integers(len(points))].copy()
    category = scene["index"] % 4
    if category == 1:
        center += rng.uniform(-12, 12, 2)
    elif category == 2:
        center = (points[0] + points[1]) / 2
    elif category == 3:
        center = rng.uniform([size / 2, size / 2], np.array(model["size"]) - size / 2)
    origin = np.clip(np.rint(center - size / 2), 0, np.array(model["size"]) - size).astype(int)
    image = render_crop(
        {"K": model[f"K_{side}"], "D": model[f"D_{side}"]},
        model["board"],
        np.asarray(pose["R"]),
        np.asarray(pose["t"]),
        tuple(origin),
        size,
        4,
        (rng.uniform(10, 65), rng.uniform(175, 245), rng.uniform(70, 165)),
    )
    points = points - origin
    if scene["index"] % 10 == 9:
        image = np.full_like(image, rng.integers(25, 230))
        points = np.empty((0, 2))
    sigma = rng.uniform(0.2, 1.2)
    image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    image = np.clip(image + rng.normal(0, rng.uniform(0, 5), image.shape), 0, 255).astype(np.uint8)
    metadata = dict(
        scene_seed=scene["seed"],
        side=side,
        origin=origin.tolist(),
        corners=points.tolist(),
        category=category,
        geometry_valid=True,
    )
    return image, heatmap(points, size), metadata


def prepare_tensors(
    prepared: Path, output: Path, split: str, size: int, workers: int, limit: int | None = None
) -> Any:
    """用多进程渲染冻结场景，缓存为 CPU TensorDataset。

    Args:
        prepared: 含 theta_A.json 和划分场景 JSON 的目录。
        output: 写入 split_patches.json 元数据的目录。
        split: 要读取的场景划分名称。
        size: 裁块边长，px。
        workers: 渲染进程数，须为正整数。
        limit: 只使用前若干场景；None 表示全部。

    Returns:
        图像和标签形状为 (S, 1, size, size) 的 TensorDataset。
        图像尚未归一化；所有样本在 CPU 内存中，不是按需加载。
    """
    model = json.loads((prepared / "theta_A.json").read_text())
    records = json.loads((prepared / f"{split}_scenes.json").read_text())
    scenes = records["scenes"][:limit]
    images, labels, metadata = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (image, target, record) in enumerate(
            pool.map(render_sample, [(model, s, size) for s in scenes], chunksize=4)
        ):
            images.append(image)
            labels.append(target)
            metadata.append(record)
            if index % 100 == 0:
                print(f"render {split}: {index + 1}/{len(scenes)}", flush=True)
    write_json(output / f"{split}_patches.json", metadata)
    return TensorDataset(
        torch.from_numpy(np.stack(images)[:, None]),
        torch.from_numpy(np.stack(labels)[:, None]),
    )


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """先写同目录临时文件，再替换训练检查点。

    Args:
        path: 最终权重路径；父目录须存在，同名检查点会替换。
        payload: torch.save 支持的模型、优化器状态与元数据字典。

    Note:
        临时扩展名为 .tmp；仅加载可信来源的检查点。
    """
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args: Any) -> None:
    """执行有限轮次的 CUDA 训练并保存配置、日志及最佳/最后权重。

    Args:
        args: CLI 命名空间，含 prepared、output、size、batch、workers、epochs、overfit。

    Raises:
        FileExistsError: 输出目录已经存在，拒绝覆盖实验。
        RuntimeError: CUDA 不可用。
        FloatingPointError: 训练或验证损失非有限。

    Note:
        创建输出目录并设置随机种子/线程；overfit 模式复用少量训练样本验证，
        不可解释为泛化评估。正常模式只读取 train/val 合成场景。
    """
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", vars(args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU training")
    random.seed(20260905)
    np.random.seed(20260905)
    torch.manual_seed(20260905)
    torch.set_num_threads(2)
    write_json(
        args.output / "environment.json",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            opencv=cv2.__version__,
            gpu=torch.cuda.get_device_name(0),
            pid=os.getpid(),
        ),
    )
    write_json(args.output / "status.json", dict(state="rendering", pid=os.getpid()))
    limit = 16 if args.overfit else None
    train = prepare_tensors(args.prepared, args.output, "train", args.size, args.workers, limit)
    val = prepare_tensors(args.prepared, args.output, "val", args.size, args.workers, limit)
    if args.overfit:
        train = TensorDataset(*(t[:16] for t in train.tensors))
        val = train
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch)
    model = corner_network().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, stale = float("inf"), 0
    start = time.time()
    for epoch in range(args.epochs):
        lr = (
            1e-3
            * min(1, (epoch + 1) / 3)
            * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_loss(model(images.cuda().float() / 255), targets.cuda())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(images)
        model.eval()
        validation = 0.0
        with torch.inference_mode():
            for images, targets in val_loader:
                loss = balanced_loss(model(images.cuda().float() / 255), targets.cuda())
                validation += loss.item() * len(images)
        validation /= len(val)
        if not math.isfinite(validation):
            raise FloatingPointError("Non-finite validation loss")
        record = dict(
            epoch=epoch + 1,
            train_loss=total / len(train),
            val_loss=validation,
            lr=lr,
            elapsed_seconds=time.time() - start,
        )
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch + 1,
            metrics=record,
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        )
        checkpoint(args.output / "last.pt", payload)
        if validation < best:
            best, stale = validation, 0
            checkpoint(args.output / "best.pt", payload)
        else:
            stale += 1
        write_json(args.output / "status.json", dict(state="training", pid=os.getpid(), **record))
        print(json.dumps(record), flush=True)
        if stale >= 10:
            break
    write_json(
        args.output / "status.json",
        dict(
            state="completed",
            best_val_loss=best,
            epoch=epoch + 1,
            overfit=args.overfit,
            pid=os.getpid(),
        ),
    )


def main() -> None:
    """解析训练 CLI，校验正数计数和裁块大小，并记录异常状态。

    失败时尽可能写 status.json，
    不覆盖已有实验目录的状态，随后重新抛出异常。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()
    if min(args.batch, args.workers, args.epochs) <= 0 or args.size <= 20:
        parser.error("Positive counts and size >20 required")
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            write_json(args.output / "status.json", dict(state="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
