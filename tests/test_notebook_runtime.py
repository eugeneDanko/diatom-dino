from pathlib import Path

import pytest

from core.notebook_runtime import find_project_root, format_command


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "notebooks" / "public"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "PROJECT_CONTEXT.md").write_text("context", encoding="utf-8")
    assert find_project_root(nested) == root


def test_find_project_root_rejects_unrelated_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)


def test_format_command_contains_every_argument() -> None:
    rendered = format_command(["python", "-m", "module", "path with spaces"])
    assert "python" in rendered
    assert "module" in rendered
    assert "path with spaces" in rendered
