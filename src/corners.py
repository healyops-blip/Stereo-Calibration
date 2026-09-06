"""Detect complete boards and retain auditable per-frame corner numbering."""

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.dataset import pair_paths, read_gray


def object_points(board: dict[str, Any]) -> Any:
    points = np.zeros((board["rows"] * board["columns"], 3), np.float32)
    points[:, :2] = np.mgrid[: board["columns"], : board["rows"]].T.reshape(-1, 2)
    return points * board["square_mm"]


def detect(image: Any, shape: tuple[int, int]) -> tuple[Any, str]:
    ok, corners = cv2.findChessboardCornersSB(image, shape, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    method = "SB"
    if not ok:
        ok, corners = cv2.findChessboardCorners(
            image, shape, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        method = "classic_subpix"
        if ok:
            corners = cv2.cornerSubPix(
                image,
                corners,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
            )
    if not ok:
        raise ValueError("corners_not_found")
    grid = corners.reshape(shape[1], shape[0], 2)
    # Per-frame convention: row zero is the upper row; columns increase to the right.
    # This establishes image ordering, not a globally identifiable physical board origin.
    if grid[0, :, 1].mean() > grid[-1, :, 1].mean():
        grid = grid[::-1]
    if grid[:, 0, 0].mean() > grid[:, -1, 0].mean():
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid.reshape(-1, 1, 2), dtype=np.float32), method


def collect(
    folder: Path, board: dict[str, Any], output: Path, excluded: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output.mkdir(parents=True, exist_ok=True)
    observations, manifest = [], []
    shape = (board["columns"], board["rows"])
    size = None
    seen: dict[str, str] = {}
    for pair_id, paths in pair_paths(folder).items():
        row: dict[str, Any] = {"dataset": folder.name, "pair_id": pair_id, "status": "ok"}
        obs: dict[str, Any] = {"pair_id": pair_id, "paths": paths}
        try:
            if set(paths) != {"left", "right"}:
                raise ValueError("missing_side")
            if pair_id in excluded:
                raise ValueError("suspected_image_discontinuity")
            for side, path in paths.items():
                image = read_gray(path)
                current_size = (image.shape[1], image.shape[0])
                if size is not None and current_size != size:
                    raise ValueError("inconsistent_image_size")
                size = current_size
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                row[f"{side}_sha256"] = digest
                row[f"{side}_duplicate_of"] = seen.get(digest, "")
                seen.setdefault(digest, f"{pair_id}/{side}")
                corners, method = detect(image, shape)
                obs[side] = corners
                obs["size"] = size
                row[f"{side}_detector"] = method
                row[f"{side}_corner_count"] = len(corners)
                x, y, w, h = cv2.boundingRect(corners)
                row[f"{side}_coverage"] = w * h / image.size
                row[f"{side}_center_x"] = float(corners[:, 0, 0].mean())
                row[f"{side}_center_y"] = float(corners[:, 0, 1].mean())
                row[f"{side}_sharpness"] = float(
                    cv2.Laplacian(image[y : y + h, x : x + w], cv2.CV_64F).var()
                )
                canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(canvas, shape, corners, True)
                for index, point in enumerate(corners[:, 0]):
                    cv2.putText(
                        canvas,
                        str(index),
                        tuple(point.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (0, 0, 255),
                        1,
                    )
                cv2.imwrite(str(output / f"{pair_id}_{side}.png"), canvas)
            observations.append(obs)
        except (ValueError, cv2.error) as error:
            row["status"] = "rejected"
            row["failure_reason"] = str(error)
        manifest.append(row)
        print(f"{folder.name}/{pair_id}: {row['status']}", flush=True)
    return observations, manifest
