"""Data policy catches forced additions as well as ordinary ignored paths."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_repository import forbidden


@pytest.mark.parametrize(
    "path",
    [
        "dataset/a.txt",
        "kaoti/a.bmp",
        "outputs/a.csv",
        "nested/dataset/a.png",
        "frame.BMP",
        "backup.tar.gz",
        "dataset.rar",
        "output/pdf/internal.pdf",
        "tmp/scratch.py",
        "run_experiments.py",
        "src/experiment_design.py",
        "src/calibration_experiments.py",
        "scripts/build_manual.py",
        "scripts/plot_C_comparison.py",
        "tests/test_manual.py",
        "delivery/raw_image.png",
        "best.pt",
        "src/deeplearning/model.PTH",
        "last.ckpt",
        "model.safetensors",
        "model.onnx",
        "best.pt.tmp",
        "best.tmp",
        "nested/last.tmp",
        "weights/parameters.bin",
        "nested/checkpoints/state.bin",
    ],
)
def test_reject_data_paths(path: str) -> None:
    assert forbidden(path)


@pytest.mark.parametrize(
    "path",
    [
        "src/dataset.py",
        "config.yaml",
        "docs/data_contracts.md",
        "tests/test_geometry.py",
        "src/deeplearning/train.py",
        "src/deeplearning/prepare.py",
        "src/deeplearning/infer_grid.py",
        "tests/test_dl_training.py",
        "tests/test_dl_grid.py",
        "delivery/calibration_results.csv",
        "delivery/pose_results.csv",
        "delivery/technical_manual.html",
        "delivery/technical_manual.pdf",
    ],
)
def test_allow_source_and_documentation(path: str) -> None:
    assert not forbidden(path)


def test_history_keeps_data_protection_without_retroactive_local_rules() -> None:
    assert forbidden(".vscode/launch.json")
    assert not forbidden(".vscode/launch.json", historical=True)
    assert forbidden("src/experiment_design.py")
    assert not forbidden("src/deeplearning/train.py", historical=True)
    assert forbidden("dataset/sample.bmp", historical=True)
    assert forbidden("outputs/result.csv", historical=True)
    assert forbidden("best.pt", historical=True)


@pytest.mark.parametrize(
    "relative", ["dataset/sample.txt", "best.pt", "best.tmp", "weights/parameters.bin"]
)
def test_forced_add_is_blocked(tmp_path: Path, relative: str) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("synthetic test data")
    (tmp_path / ".gitignore").write_text("dataset/\nweights/\n*.pt\n")
    subprocess.run(["git", "add", "-f", relative], cwd=tmp_path, check=True)
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert relative in result.stderr
