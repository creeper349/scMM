"""Build the source release and complete overlay patches from one payload."""
import argparse
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "SCMM-GUI-0.3.0-fix"
VERSION = "0.3.0.post1"
ROOT_FILES = [".gitignore", "pyproject.toml", "MANIFEST.in", "environment.yml", "README.md",
              "INSTALL_SCMM_GUI.md", "WINDOWS_SETUP_AND_USAGE_BILINGUAL.md", "ANNOTATION_GUIDE.md",
              "LIPID_NOMENCLATURE.md", "RELEASE_NOTES_0.3.0-fix.md", "UPGRADE_TO_0.3.0-fix.md",
              "setup_scmm.cmd", "setup_scmm.ps1", "setup_thermo_raw.ps1", "start_scmm.cmd", "start_scmm.ps1",
              "install_with_python.cmd", "install_with_python.ps1", "docs/languages.md"]


def release_files():
    files = {name: ROOT / name for name in ROOT_FILES}
    for path in (ROOT / "scMM").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".html", ".ps1"}:
            files[path.relative_to(ROOT).as_posix()] = path
    for path in (ROOT / "docs" / "images").glob("*.png"):
        files[path.relative_to(ROOT).as_posix()] = path
    return dict(sorted(files.items()))


def build(destination):
    destination.mkdir(parents=True, exist_ok=True)
    files = release_files()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
    payload_bytes = payload.getvalue()
    full = destination / f"{NAME}.zip"
    with zipfile.ZipFile(full, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, f"{NAME}/{name}")
    outputs = [full]
    for source in ("0.2.0", "0.3.0"):
        directory = f"scmm_patch_{source}"
        cmd = f'''@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0{directory}\\apply_update.ps1" -Target "%~dp0." %*
set "SCMM_PATCH_EXIT=%errorlevel%"
echo.
if not "%SCMM_PATCH_EXIT%"=="0" echo [SCMM] Update failed with exit code %SCMM_PATCH_EXIT%. Review the error above.
pause
exit /b %SCMM_PATCH_EXIT%
'''
        manifest = dict(source_version=source, target_version=VERSION, release=NAME, files=list(files))
        readme = f'''# SCMM {source} → {NAME}

1. 关闭 SCMM 网页程序及启动终端。
2. 将本补丁完整解压到原 SCMM 软件目录，使 `apply_patch_{source}.cmd` 与原来的 `start_scmm.cmd` 同级。
3. 双击 `apply_patch_{source}.cmd`，等待程序更新和依赖检查完成。首次补齐依赖需要联网。
4. 运行 `start_scmm.cmd`。

旧程序和配置自动备份到 `.scmm-update-backups`。已有环境、数据及结果保留。升级采用新版默认参数，原来的自定义参数和语言包可从备份中合并。

如环境配置失败，保留窗口中的错误，处理后重新运行 `setup_scmm.cmd`。可稍后选择安装可选的 Thermo RAW 组件。补丁可重复运行。

Close SCMM, extract the entire patch into the existing software folder, and run `apply_patch_{source}.cmd`. It backs up the previous program, installs the complete updated source, and checks missing/incompatible dependencies. Internet access is needed for dependency installation. Then run `start_scmm.cmd`. Previous settings are in `.scmm-update-backups`.

完整说明见更新后的 `UPGRADE_TO_0.3.0-fix.md`。
'''
        patch = destination / f"{NAME}-patch-from-{source}.zip"
        with zipfile.ZipFile(patch, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"apply_patch_{source}.cmd", cmd.replace("\n", "\r\n").encode("utf8"))
            archive.writestr(f"PATCH_README_{source}.md", readme.encode("utf-8-sig"))
            archive.writestr(f"{directory}/manifest.json", json.dumps(manifest, indent=2).encode())
            archive.writestr(f"{directory}/payload.zip", payload_bytes)
            archive.write(ROOT / "scripts" / "apply_update.ps1", f"{directory}/apply_update.ps1")
        outputs.append(patch)
    for path in outputs:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        print(f"{path} ({path.stat().st_size:,} bytes)")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "release")
    build(parser.parse_args().output)
