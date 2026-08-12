"""Subprocess entry point for one detached scMM processing task."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from scMM.file.data import CyESIData

from .processing import ProcessingParameters
from .tasks import load_request, update_task, utc_now


def build_parser() -> argparse.ArgumentParser:
    """Build the private worker CLI parser."""
    parser = argparse.ArgumentParser(description="Run one validated scMM processing request")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--start-gate", type=Path)
    return parser


def run_request(request_path: str | Path, state_path: str | Path) -> Path:
    """Run processing and save a reproducibility manifest."""
    payload = load_request(request_path)
    state = update_task(
        state_path,
        status="running",
        pid=os.getpid(),
        started_at=utc_now(),
        error=None,
    )
    source = Path(payload["input_path"]).resolve(strict=True)
    output_root = Path(payload["output_root"]).resolve(strict=True)
    expected_result = Path(payload["result_path"])
    if expected_result.parent.resolve(strict=True) != output_root:
        raise PermissionError("Worker result path is outside its validated output root")

    parameters = ProcessingParameters(**payload["parameters"])
    logging.getLogger(__name__).info("Processing %s as task %s", source, state.task_id)
    dataset = CyESIData.load_from_file(
        source,
        ref_mz=parameters.ref_mz,
        **parameters.load_kwargs(),
    )
    original_name = dataset.get_name()
    dataset.file_meta["name"] = payload["result_name"]
    dataset.file_meta["source_name"] = original_name
    result_path = dataset.save(output_root, overwrite=bool(payload["overwrite"]))
    if result_path.resolve() != expected_result.resolve():
        raise RuntimeError(f"Unexpected result path: {result_path}")
    _write_manifest(result_path, payload)
    update_task(
        state_path,
        status="succeeded",
        finished_at=utc_now(),
        result_path=str(result_path.resolve()),
    )
    return result_path


def _write_manifest(result_path: Path, request: dict) -> None:
    try:
        scmm_version = version("scMM")
    except PackageNotFoundError:
        scmm_version = "source-tree"
    manifest = {
        "schema_version": 1,
        "task_id": request["task_id"],
        "created_at": utc_now(),
        "input_path": request["input_path"],
        "result_path": str(result_path.resolve()),
        "parameters": request["parameters"],
        "warnings": request.get("warnings", []),
        "software": {
            "scMM": scmm_version,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    with (result_path / "scmm-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Execute a worker request and record terminal failure state."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.start_gate is not None:
            _wait_for_start_gate(args.start_gate)
        run_request(args.request, args.state)
    except Exception as exc:
        logging.getLogger(__name__).exception("Processing failed")
        update_task(
            args.state,
            status="failed",
            finished_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1
    return 0


def _wait_for_start_gate(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Task manager did not release the worker start gate")
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
