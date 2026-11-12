from scMM.util.threadpool import _safe_method, configure_windows_threadpools


def test_safe_threadpool_method_falls_back_only_for_oserror():
    def broken():
        raise OSError("broken optional runtime")

    assert _safe_method(broken, 1)() == 1

    def valid():
        return 8

    assert _safe_method(valid, 1)() == 8


def test_threadpool_configuration_is_idempotent():
    configure_windows_threadpools()
    configure_windows_threadpools()
