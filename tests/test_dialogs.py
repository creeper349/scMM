from types import SimpleNamespace

from scMM.file.dialogs import _windows_dialog


def test_batch_dialog_returns_folders_and_multiple_files(monkeypatch):
    from scMM.file.dialogs import choose_data_paths
    def run(args, **kwargs):
        assert args[-1].endswith("select_inputs.ps1")
        return SimpleNamespace(stdout="D:\\数据\r\nD:\\other\\a.raw\r\nD:\\other\\b.mzML")
    monkeypatch.setattr("scMM.file.dialogs.sys.platform", "win32")
    monkeypatch.setattr("scMM.file.dialogs.subprocess.run", run)
    assert choose_data_paths() == ["D:\\数据", "D:\\other\\a.raw", "D:\\other\\b.mzML"]


def test_save_dialog_uses_unicode_default_name_and_selected_directory(monkeypatch):
    import base64
    from scMM.file.dialogs import choose_save_path
    observed = {}
    def run(args, **kwargs):
        observed['script'] = base64.b64decode(args[-1]).decode('utf-16-le')
        assert kwargs['encoding'] == 'utf-8'
        return SimpleNamespace(stdout="D:\\图像\\PC 40_10.svg")
    monkeypatch.setattr('scMM.file.dialogs.subprocess.run', run)
    monkeypatch.setattr('scMM.file.dialogs.sys.platform', 'win32')
    result = choose_save_path("SCMM_组别 O'Brien.svg", 'D:\\图像')
    assert result == 'D:\\图像\\PC 40_10.svg'
    assert 'SaveFileDialog' in observed['script']
    assert "SCMM_组别 O''Brien.svg" in observed['script']
    assert '$dialog.OverwritePrompt = $true' in observed['script']


def test_annotation_file_dialog_filters_and_multiple_paths(monkeypatch):
    observed = {}
    def run(args, **kwargs):
        observed['command'] = args[-1]
        return SimpleNamespace(stdout='D:\\data\\a.msp\r\nD:\\data\\b.msp\r\n')
    monkeypatch.setattr('scMM.file.dialogs.subprocess.run', run)
    paths = _windows_dialog('file', [('Libraries', '*.msp *.mgf')], multiple=True)
    assert paths == ['D:\\data\\a.msp', 'D:\\data\\b.msp']
    assert '$dialog.Multiselect=$true' in observed['command']
    assert '*.msp;*.mgf' in observed['command']
