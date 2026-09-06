"""Minimal CPU training integration test for the corner network."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.deeplearning.synthetic import heatmap  # noqa: E402
from src.deeplearning.train import balanced_loss, checkpoint, corner_network  # noqa: E402


def test_training_step_updates_and_restores_model(tmp_path: Path) -> None:
    """One optimizer step must update finite weights and survive a checkpoint round-trip."""
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    model = corner_network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    images = torch.rand(2, 1, 32, 32)
    target = heatmap(np.array([[16.25, 15.75]], dtype=np.float32), 32)
    targets = torch.from_numpy(np.repeat(target[None, None], len(images), axis=0))
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    loss = balanced_loss(model(images), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters(), strict=True)
    )

    checkpoint_path = tmp_path / "best.pt"
    checkpoint(checkpoint_path, {"epoch": 1, "model": model.state_dict()})
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = corner_network()
    restored.load_state_dict(payload["model"])
    with torch.inference_mode():
        assert torch.equal(model(images), restored(images))
