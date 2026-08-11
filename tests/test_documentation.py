import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
NOTEBOOK_PATH = ROOT / "scMM_workflow.ipynb"


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda path: path.name)
def test_markdown_local_links_resolve(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]*\]\(([^)]+)\)", text)

    for target in targets:
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        local_target = unquote(target.split("#", 1)[0])
        assert (path.parent / local_target).exists(), f"Broken local link in {path}: {target}"


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda path: path.name)
def test_markdown_python_examples_parse(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for index, source in enumerate(re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL), 1):
        ast.parse(source, filename=f"{path}:python-block-{index}")


def test_workflow_notebook_is_clean_and_uses_project_kernel() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "scMM",
        "language": "python",
        "name": "scmm-dev",
    }
    assert "parameters" in code_cells[0].get("metadata", {}).get("tags", [])
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(not cell.get("outputs") for cell in code_cells)

    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{NOTEBOOK_PATH}:code-cell-{index}")
        assert "/home/" not in source
        assert not re.search(r"[A-Za-z]:[\\/]", source)
