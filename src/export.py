"""Round-trippable numeric artifacts and original-camera visualizations."""

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.calibration import check_model
from src.dataset import read_gray

CAMERA_KEYS = ("K_left", "D_left", "K_right", "D_right", "R_RL", "t_RL", "F")
ORIGIN = "per_frame_upper_row_left_corner; physical_origin_not_globally_identifiable"


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    for key in CAMERA_KEYS:
        model[key] = np.asarray(model[key], dtype=np.float64)
    model["size"] = tuple(model["size"])
    return model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to export empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_model(path: Path, model: dict[str, Any], board: dict[str, Any]) -> None:
    rows = []
    parameters = {key: model[key] for key in CAMERA_KEYS}
    parameters.update(
        baseline_mm=model["baseline_mm"],
        stereo_rms_px=model["stereo_rms_px"],
        image_width=model["size"][0],
        image_height=model["size"][1],
        retained_pair_count=len(model["retained_ids"]),
        **board,
    )
    for key, value in parameters.items():
        unit = "dimensionless"
        if key.startswith("K_") or key in ("image_width", "image_height", "stereo_rms_px"):
            unit = "px"
        if key in ("t_RL", "baseline_mm", "square_mm"):
            unit = "mm"
        for index, number in np.ndenumerate(np.atleast_2d(value)):
            rows.append(
                dict(
                    group="camera",
                    parameter=key,
                    row=index[0],
                    col=index[1],
                    value=float(number),
                    unit="dimensionless" if key.startswith("K_") and index[0] == 2 else unit,
                    convention="P_R=R_RL@P_L+t_RL; pinhole; D=k1,k2,p1,p2,k3",
                )
            )
    write_csv(path, rows)


def empty_pose_row(pair_id: str, reason: str) -> dict[str, Any]:
    """Keep the required numeric columns even when every test image fails."""
    row = dict(
        pair_id=pair_id,
        status="failed",
        reference_frame="original_left_camera",
        board_origin_convention=ORIGIN,
        failure_reason=reason,
    )
    for key in (
        "rx_rad",
        "ry_rad",
        "rz_rad",
        "tx_mm",
        "ty_mm",
        "tz_mm",
        "left_rms_px",
        "right_rms_px",
        "joint_rms_px",
        "ambiguity_flag",
    ):
        row[key] = ""
    row.update({f"r{i + 1}{j + 1}": "" for i in range(3) for j in range(3)})
    return row


def pose_row(pair_id: str, fit: dict[str, Any]) -> dict[str, Any]:
    pose = fit["pose"]
    rotation = cv2.Rodrigues(pose[:3])[0]
    row = empty_pose_row(pair_id, "")
    row["status"] = "ok"
    row.update(zip(("rx_rad", "ry_rad", "rz_rad", "tx_mm", "ty_mm", "tz_mm"), pose, strict=True))
    row.update({f"r{i + 1}{j + 1}": rotation[i, j] for i in range(3) for j in range(3)})
    row.update({key: value for key, value in fit.items() if key not in ("pose", "initial_pose")})
    return row


def draw_pose(
    obs: dict[str, Any], fit: dict[str, Any], model: dict[str, Any], output: Path, square_mm: float
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rotation = cv2.Rodrigues(fit["pose"][:3])[0]
    translation = fit["pose"][3:].reshape(3, 1)
    for side in ("left", "right"):
        r, t = rotation, translation
        if side == "right":
            r, t = model["R_RL"] @ r, model["R_RL"] @ t + model["t_RL"]
        image = cv2.cvtColor(read_gray(obs["paths"][side]), cv2.COLOR_GRAY2BGR)
        cv2.drawFrameAxes(
            image, model[f"K_{side}"], model[f"D_{side}"], cv2.Rodrigues(r)[0], t, 3 * square_mm, 2
        )
        cv2.imwrite(str(output / f"{obs['pair_id']}_{side}.png"), image)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def camera_from_csv(path: Path) -> dict[str, Any]:
    """Reconstruct matrix shapes from indexed CSV cells, without calibration.json."""
    cells: dict[str, dict[tuple[int, int], float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            indexed = cells.setdefault(row["parameter"], {})
            index = (int(row["row"]), int(row["col"]))
            if index in indexed:
                raise ValueError(f"Duplicate CSV cell: {row['parameter']}/{index}")
            indexed[index] = float(row["value"])
    model: dict[str, Any] = {}
    for key, indexed in cells.items():
        shape = (max(i[0] for i in indexed) + 1, max(i[1] for i in indexed) + 1)
        array = np.full(shape, np.nan)
        for index, value in indexed.items():
            array[index] = value
        if not np.isfinite(array).all():
            raise ValueError(f"Incomplete or nonfinite CSV parameter: {key}")
        model[key] = array
    model["size"] = (int(model["image_width"].item()), int(model["image_height"].item()))
    model["board"] = dict(
        columns=int(model["columns"].item()),
        rows=int(model["rows"].item()),
        square_mm=model["square_mm"].item(),
    )
    check_model(model)
    return model
