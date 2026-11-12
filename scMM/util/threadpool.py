from __future__ import annotations

import functools
import sys


def _safe_method(method, fallback):
    if method is None or getattr(method, "_scmm_windows_safe", False):
        return method

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except OSError:
            return fallback

    wrapper._scmm_windows_safe = True
    return wrapper


def configure_windows_threadpools():
    """Protect sklearn from broken optional BLAS/OpenMP controller DLLs.

    Scientific wheels used by SCMM are OpenBLAS based. Some Conda builds of
    optional packages can additionally load an unused MKL runtime whose thread
    query function raises Windows error 0xc06d007f. threadpoolctl probes every
    loaded runtime before many sklearn operations, so one broken optional DLL
    can otherwise break unrelated algorithms.
    """
    if sys.platform != "win32":
        return
    try:
        import threadpoolctl
    except ImportError:
        return
    controller_names = (
        "MKLController",
        "OpenBLASController",
        "OpenMPController",
        "BLISController",
        "FlexiBLASController",
    )
    for name in controller_names:
        controller = getattr(threadpoolctl, name, None)
        if controller is None:
            continue
        for method_name, fallback in (
            ("get_num_threads", 1),
            ("set_num_threads", None),
            ("get_version", None),
        ):
            method = getattr(controller, method_name, None)
            if method is not None:
                setattr(controller, method_name, _safe_method(method, fallback))
