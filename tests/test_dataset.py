"""File pairing and failure reporting must remain explicit and lossless."""

from pathlib import Path

import cv2
import numpy as np

from src.corners import collect
from src.dataset import pair_paths, read_gray
from src.export import write_json
from src.pipeline import stage_pose


def test_noncontiguous_pair_ids_and_missing_side(tmp_path: Path) -> None:
    for side in ("left", "right"):
        (tmp_path / side).mkdir()
    for side, number in (("left", "000001"), ("right", "000001"), ("left", "000003")):
        cv2.imwrite(
            str(tmp_path / side / f"{side}_pair_{number}.bmp"), np.zeros((20, 20), np.uint8)
        )
    pairs = pair_paths(tmp_path)
    assert list(pairs) == ["000001", "000003"]
    assert set(pairs["000003"]) == {"left"}
    observations, manifest = collect(
        tmp_path, dict(columns=12, rows=9), tmp_path / "corners", ["000001"]
    )
    assert not observations
    assert manifest[0]["failure_reason"] == "suspected_image_discontinuity"
    assert manifest[1]["failure_reason"] == "missing_side"


def test_gray_image_with_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "中文图像.bmp"
    image = np.arange(400, dtype=np.uint8).reshape(20, 20)
    cv2.imwrite(str(path), image)
    np.testing.assert_array_equal(read_gray(path), image)


def test_five_failure_rows_are_preserved(tmp_path: Path) -> None:
    import csv

    board = dict(columns=12, rows=9, square_mm=19.0)
    output = tmp_path / "output"
    model = {key: np.eye(3) for key in ("K_left", "K_right", "R_RL", "F")}
    model.update(D_left=np.zeros((1, 5)), D_right=np.zeros((1, 5)), t_RL=np.zeros((3, 1)))
    write_json(output / "calibration/calibration.json", dict(**model, board=board, size=[20, 20]))
    (tmp_path / "images/left").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/left/left_pair_000001.bmp"), np.zeros((20, 20), np.uint8))
    expected = [f"{i:06d}" for i in range(1, 6)]
    stage_pose(dict(board=board, pose_dir="images", expected_test_ids=expected), tmp_path, output)
    with (output / "pose/pose_results.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["pair_id"] for r in rows] == expected
    assert all(r["status"] == "failed" and r["failure_reason"] for r in rows)
    assert all(r["tx_mm"] == "" and r["joint_rms_px"] == "" for r in rows)
