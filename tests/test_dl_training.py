"""PyTorch CPU regression tests and separately marked CUDA checks."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.deeplearning.infer_grid import detect_ordered  # noqa: E402
from src.deeplearning.real_evaluate import dense_response  # noqa: E402
from src.deeplearning.subpixel_experiment import select_method  # noqa: E402
from src.deeplearning.train import balanced_loss, checkpoint, corner_network  # noqa: E402


def test_network_shape_and_finite_background_gradients() -> None:
    torch.set_num_threads(1)
    model = corner_network()
    logits = model(torch.zeros(2, 1, 32, 32))
    assert tuple(logits.shape) == (2, 1, 32, 32)
    loss = balanced_loss(logits, torch.zeros_like(logits))
    loss.backward()
    assert np.isfinite(loss.item())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("shape", [(0, 1, 32, 32), (1, 2, 32, 32), (1, 1, 20, 32), (32, 32)])
def test_loss_rejects_invalid_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="floating"):
        balanced_loss(torch.zeros(shape), torch.zeros(shape))


def test_loss_rejects_mismatched_shape_and_integer_labels() -> None:
    logits = torch.zeros(1, 1, 32, 32)
    for target in (torch.zeros(1, 1, 33, 32), torch.zeros_like(logits, dtype=torch.int64)):
        with pytest.raises(ValueError, match="floating"):
            balanced_loss(logits, target)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA inference test")
def test_tile_halo_matches_full_frame_inference() -> None:
    image = np.random.default_rng(3).integers(0, 256, (300, 310), dtype=np.uint8)
    model = corner_network().cuda().eval()
    tiled = dense_response(model, image)
    with torch.inference_mode():
        full = model(torch.from_numpy(image[None, None]).cuda().float() / 255)
    assert np.allclose(tiled, full.sigmoid()[0, 0].cpu().numpy(), atol=1e-6)


def test_ordered_pipeline_never_calls_chessboard_detectors(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Traditional chessboard detector must not be called")

    points = np.mgrid[:12, :9].T.reshape(-1, 2).astype(np.float32) * 20 + 50
    monkeypatch.setattr(cv2, "findChessboardCorners", forbidden)
    monkeypatch.setattr(cv2, "findChessboardCornersSB", forbidden)
    monkeypatch.setattr(
        "src.deeplearning.infer_grid.neural_candidates",
        lambda network, image: (points, points, None),
    )
    result = detect_ordered(None, np.zeros((300, 400), np.uint8), (12, 9), None)
    assert len(result["corners"]) == 108


def test_checkpoint_roundtrip_and_replace(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    for epoch in (1, 2):
        checkpoint(path, {"epoch": epoch, "model": {"weight": torch.tensor([float(epoch)])}})
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["epoch"] == epoch
        assert loaded["model"]["weight"].item() == epoch
        assert not path.with_suffix(".tmp").exists()


def test_checkpoint_save_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "best.pt"
    checkpoint(path, {"epoch": 1})
    original = path.read_bytes()

    def fail_save(payload: object, target: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("src.deeplearning.train.torch.save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        checkpoint(path, {"epoch": 2})
    assert path.read_bytes() == original


def test_selection_rejects_geometry_or_coverage_regressions() -> None:
    methods = ["win3", "win5", "win7", "log3"]
    synthetic = {
        m: dict(recall=0.99, selection_cost=v)
        for m, v in zip(methods, [0.1, 0.05, 0.09, 0.02], strict=True)
    }
    real: dict[str, Any] = {
        m: dict(
            complete_images=2,
            geometry_by_pair={"one": dict(joint_rms_px=1.0, epipolar_mean_px=1.0)},
        )
        for m in methods
    }
    real["win5"]["geometry_by_pair"]["one"]["joint_rms_px"] = 2.0
    real["log3"]["complete_images"] = 0
    selected = select_method(synthetic, real)
    assert selected["selected"] == "win7"
    assert selected["B_used_for_selection"] is False


def test_selected_refinement_reaches_grid_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.mgrid[:12, :9].T.reshape(-1, 2).astype(np.float32) * 20 + 50
    used = []

    def refinement(image: object, raw: object, response: object, method: str) -> Any:
        used.append(method)
        return points

    monkeypatch.setattr(
        "src.deeplearning.infer_grid.raw_candidates",
        lambda network, image: (points, None),
    )
    monkeypatch.setattr("src.deeplearning.infer_grid.refine", refinement)
    result = detect_ordered(None, np.zeros((300, 400), np.uint8), (12, 9), None, "log3")
    assert used == ["log3"]
    assert len(result["corners"]) == 108
