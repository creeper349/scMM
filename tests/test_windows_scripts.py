import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows batch scripts")
ROOT = Path(__file__).parents[1]


def run_batch(path, args="", input=""):
    return subprocess.run(f'cmd.exe /d /s /c ""{path}" {args}"', input=input, capture_output=True,
                          encoding="utf-8", errors="replace", timeout=40)


def test_setup_single_file_read_only_check():
    if not (ROOT / ".scmm-env" / "python.exe").exists():
        pytest.skip("requires an existing local environment")
    result = run_batch(ROOT / "setup_scmm.cmd", "--check --no-pause")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no packages were installed" in result.stdout


@pytest.mark.parametrize("filename, detail", [
    ("setup_scmm.cmd", "pyproject.toml is missing"),
    ("start_scmm.cmd", "webapp.py is missing"),
])
def test_failed_batch_prints_error_and_waits_for_key(tmp_path, filename, detail):
    directory = tmp_path / "中文 & space"
    directory.mkdir()
    script = directory / filename
    shutil.copy2(ROOT / filename, script)
    result = run_batch(script, "--check" if filename == "setup_scmm.cmd" else "", input="\n")
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert detail in output
    assert "exit code" in output
    assert "Press any key" in output or "请按任意键" in output
