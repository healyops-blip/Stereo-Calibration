"""Prevent missing source docstrings without importing optional ML dependencies."""

import ast
from pathlib import Path


def test_source_functions_and_classes_have_docstrings() -> None:
    """Check presence, not semantic correctness, across production Python files."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        set((root / "src").rglob("*.py"))
        | set((root / "scripts").rglob("*.py"))
        | set(root.glob("*.py"))
    )
    missing = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not (ast.get_docstring(node) or "").strip():
                    missing.append(f"{path.relative_to(root)}:{node.lineno}: {node.name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)
