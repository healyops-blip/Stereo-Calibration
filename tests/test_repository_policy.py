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
    ],
)
def test_reject_data_paths(path: str) -> None:
    assert forbidden(path)


@pytest.mark.parametrize("path", ["src/dataset.py", "config.yaml", "docs/data_contracts.md"])
def test_allow_source_and_documentation(path: str) -> None:
    assert not forbidden(path)


def test_forced_add_is_blocked(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset/sample.txt").write_text("synthetic test data")
    (tmp_path / ".gitignore").write_text("dataset/\n")
    subprocess.run(["git", "add", "-f", "dataset/sample.txt"], cwd=tmp_path, check=True)
    script = Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "dataset/sample.txt" in result.stderr
