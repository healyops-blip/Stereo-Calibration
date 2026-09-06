"""Keep default pytest discovery within the delivery boundary."""

from pathlib import Path

from scripts.check_repository import forbidden

ROOT = Path(__file__).resolve().parents[1]
collect_ignore = [
    path.name
    for path in Path(__file__).parent.glob("test_*.py")
    if forbidden(path.relative_to(ROOT).as_posix())
]
