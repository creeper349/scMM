from __future__ import annotations

import subprocess
import sys
import base64
import os
from pathlib import Path


def _windows_dialog(kind, filetypes=None, multiple=False):
    if kind == "directory":
        body = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog=New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description='Select a folder'; "
            "if($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
            "{[Console]::Out.Write($dialog.SelectedPath)}"
        )
    else:
        filetypes = filetypes or [("Supported files", "*.mzML *.mzXML *.raw *.h5ad"), ("All files", "*.*")]
        filters = "|".join(f"{label}|{';'.join(pattern.split())}" for label, pattern in filetypes).replace("'", "''")
        body = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog=New-Object System.Windows.Forms.OpenFileDialog; "
            "$dialog.Title='Select file'; "
            f"$dialog.Filter='{filters}'; "
            f"$dialog.Multiselect=${str(multiple).lower()}; "
            "if($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK)"
            "{[Console]::Out.Write(($dialog.FileNames -join [Environment]::NewLine))}"
        )
    body = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); " + body
    completed = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-STA", "-Command", body],
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    result = completed.stdout.strip()
    return result.splitlines() if multiple else result


def choose_local_path(kind="file", *, filetypes=None, multiple=False):
    """Open an OS-native chooser on the machine running the local web server."""
    if kind not in {"file", "directory"}:
        raise ValueError("kind must be 'file' or 'directory'")
    if sys.platform == "win32":
        return _windows_dialog(kind, filetypes, multiple)
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        if kind == "directory":
            return filedialog.askdirectory()
        chooser = filedialog.askopenfilenames if multiple else filedialog.askopenfilename
        return chooser(
            filetypes=filetypes or [
                ("Supported files", "*.mzML *.mzXML *.raw *.h5ad"),
                ("All files", "*.*"),
            ]
        )
    finally:
        root.destroy()


def choose_data_paths():
    """Choose folders and multiple MS files in the same browser window."""
    if sys.platform == "win32":
        from scMM.i18n import current_language, tr
        result = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass",
             "-File", str(Path(__file__).with_name("select_inputs.ps1"))],
            capture_output=True, encoding="utf-8", check=True,
            env={**os.environ, "SCMM_UI_LANGUAGE": current_language(),
                 "SCMM_PICKER_TITLE": tr("选择质谱文件或文件夹"),
                 "SCMM_PICKER_FOLDER": tr("选择当前文件夹")},
        )
        return result.stdout.strip().splitlines()
    paths = choose_local_path(filetypes=[("MS files", "*.mzML *.mzXML *.raw")], multiple=True)
    return list(paths)


def choose_save_path(default_filename, initial_directory=None):
    """Choose a figure or dataset destination on the computer hosting SCMM."""
    extension = Path(default_filename).suffix.lstrip(".").lower()
    title = "Save SCMM dataset" if extension == "h5ad" else "Save SCMM figure"
    file_label = "H5AD dataset" if extension == "h5ad" else f"{extension.upper()} image"
    if sys.platform == "win32":
        filename = str(default_filename).replace("'", "''")
        directory = str(initial_directory or "").replace("'", "''")
        script = (
            "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.SaveFileDialog; "
            f"$dialog.Title = '{title}'; "
            f"$dialog.FileName = '{filename}'; "
            f"$dialog.InitialDirectory = '{directory}'; "
            f"$dialog.Filter = '{file_label}|*.{extension}'; "
            f"$dialog.DefaultExt = '{extension}'; "
            "$dialog.AddExtension = $true; $dialog.OverwritePrompt = $true; "
            "$owner = New-Object System.Windows.Forms.Form; $owner.TopMost = $true; "
            "try { if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) "
            "{ [Console]::Out.Write($dialog.FileName) } } "
            "finally { $dialog.Dispose(); $owner.Dispose() }"
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        result = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-STA", "-EncodedCommand", encoded],
            capture_output=True, encoding="utf-8", check=True,
        )
        return result.stdout.strip()
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.asksaveasfilename(
            parent=root, title=title, initialfile=default_filename,
            initialdir=str(initial_directory) if initial_directory else None,
            defaultextension=f".{extension}", filetypes=[(file_label, f"*.{extension}")],
        )
    finally:
        root.destroy()
