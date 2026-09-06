"""Verify standalone CSV loading without access to calibration JSON."""

from pathlib import Path

import numpy as np
import pytest

from src.export import camera_from_csv, draw_pose, export_model


def test_csv_only_camera_roundtrip(tmp_path: Path) -> None:
    model = dict(
        K_left=np.diag([800.0, 810.0, 1.0]),
        K_right=np.diag([805.0, 815.0, 1.0]),
        D_left=np.zeros((1, 5)),
        D_right=np.zeros((1, 5)),
        R_RL=np.eye(3),
        t_RL=np.array([[-62.0], [0.0], [0.0]]),
        F=np.eye(3),
        size=(1280, 720),
        baseline_mm=62.0,
        stereo_rms_px=0.2,
        retained_ids=["000001"],
    )
    board = dict(columns=12, rows=9, square_mm=19.0)
    path = tmp_path / "camera.csv"
    export_model(path, model, board)
    actual = camera_from_csv(path)
    for key in ("K_left", "K_right", "D_left", "D_right", "R_RL", "t_RL"):
        np.testing.assert_array_equal(actual[key], model[key])
    assert actual["board"] == board
    assert actual["size"] == (1280, 720)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("camera,K_left,0,0,900,px,test\n")
    with pytest.raises(ValueError, match="Duplicate CSV cell"):
        camera_from_csv(path)


@pytest.mark.parametrize("fail_at", [1, 2, None])
def test_draw_pose_checks_each_image_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int | None
) -> None:
    written: list[str] = []

    def write(path: str, image: object) -> bool:
        written.append(path)
        return len(written) != fail_at

    monkeypatch.setattr("src.export.read_gray", lambda path: np.zeros((100, 100), np.uint8))
    monkeypatch.setattr("src.export.cv2.drawFrameAxes", lambda *args: None)
    monkeypatch.setattr("src.export.cv2.imwrite", write)
    obs = dict(pair_id="000001", paths=dict(left=Path("left.bmp"), right=Path("right.bmp")))
    fit = dict(pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 500.0]))
    model = dict(
        K_left=np.eye(3),
        K_right=np.eye(3),
        D_left=np.zeros(5),
        D_right=np.zeros(5),
        R_RL=np.eye(3),
        t_RL=np.array([[-60.0], [0.0], [0.0]]),
    )
    if fail_at is None:
        draw_pose(obs, fit, model, tmp_path, 19.0)
        assert len(written) == 2
    else:
        side = "left" if fail_at == 1 else "right"
        with pytest.raises(OSError, match=f"000001_{side}.png"):
            draw_pose(obs, fit, model, tmp_path, 19.0)
        assert len(written) == fail_at
