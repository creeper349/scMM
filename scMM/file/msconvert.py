"""ProteoWizard MSConvert discovery and vendor RAW conversion."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class MSConvertError(RuntimeError):
    """Raised when MSConvert cannot be found or a conversion fails."""


def find_msconvert(executable=None) -> Path:
    """Return the configured MSConvert executable.

    Resolution order is an explicit argument, ``SCMM_MSCONVERT``, ``PATH``,
    then common ProteoWizard installation directories on Windows.
    """
    candidates = []
    if executable:
        candidates.append(Path(executable).expanduser())
    configured = os.environ.get("SCMM_MSCONVERT")
    if configured:
        candidates.append(Path(configured).expanduser())
    discovered = shutil.which("msconvert") or shutil.which("msconvert.exe")
    if discovered:
        candidates.append(Path(discovered))

    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            candidates.extend(
                sorted(
                    Path(root).glob("ProteoWizard*/msconvert.exe"),
                    reverse=True,
                )
            )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise MSConvertError(
        "MSConvert was not found. Install ProteoWizard, pass msconvert_path, "
        "or set the SCMM_MSCONVERT environment variable."
    )


def convert_raw(
    input_path,
    output_dir,
    *,
    executable=None,
    output_format="mzML",
    filters=None,
) -> Path:
    """Convert one vendor RAW input to mzML/mzXML and return its path."""
    source = Path(input_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output_format = str(output_format)
    if output_format not in {"mzML", "mzXML"}:
        raise ValueError("output_format must be 'mzML' or 'mzXML'")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    program = find_msconvert(executable)

    command = [
        str(program),
        str(source),
        f"--{output_format}",
        "--64",
        "--zlib",
        "--outdir",
        str(destination),
    ]
    for item in filters or ():
        command.extend(["--filter", str(item)])
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise MSConvertError(f"MSConvert failed for {source}: {details}") from exc
    except OSError as exc:
        raise MSConvertError(f"Could not start MSConvert at {program}: {exc}") from exc

    expected = destination / f"{source.stem}.{output_format}"
    if expected.is_file():
        return expected
    matches = sorted(
        destination.glob(f"*.{output_format}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    details = (completed.stderr or completed.stdout or "no output").strip()
    raise MSConvertError(
        f"MSConvert completed but produced no {output_format} file: {details}"
    )
