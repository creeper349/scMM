from types import SimpleNamespace

import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from scMM.webapp import _image_filename, _save_image


def test_content_filename_covers_embedding_and_spectrum():
    embedding = go.Figure()
    embedding.update_layout(meta={'scmm_color_label':'group'})
    assert _image_filename('embedding_X_umap','svg',figure=embedding) == 'SCMM_UMAP降维图_group.svg'
    spectrum = go.Figure()
    spectrum.update_layout(title='PC 40:10 · 826.53810')
    name = _image_filename('msms_mirror','png',figure=spectrum)
    assert 'MSMS镜像谱' in name and 'PC 40_10' in name and name.endswith('.png')


def test_image_cancel_and_failed_replace_preserve_file(monkeypatch, tmp_path):
    state = SimpleNamespace(session_state={}, success=lambda _:None, error=lambda _:None)
    monkeypatch.setattr('scMM.file.dialogs.choose_save_path', lambda *args:'')
    _save_image(state, b'image', 'image.png')
    assert not list(tmp_path.iterdir())
    destination = tmp_path / 'existing.svg'
    destination.write_bytes(b'original')
    monkeypatch.setattr('scMM.file.dialogs.choose_save_path', lambda *args:str(destination))
    def fail(*args):
        raise OSError('Disk failure')
    monkeypatch.setattr('pathlib.Path.replace', fail)
    _save_image(state, b'new', 'image.svg')
    assert destination.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [destination]


def test_redraw_dialog_saves_preview_to_chosen_path(monkeypatch, tmp_path):
    destination = tmp_path / '所选图.svg'
    requested = []
    def choose(filename, directory):
        requested.append(filename)
        return str(destination)
    monkeypatch.setattr('scMM.file.dialogs.choose_save_path', choose)
    script = '''
import streamlit as st
import plotly.graph_objects as go
from scMM.webapp import _plotly_with_export
figure=go.Figure(go.Scatter(x=[1,2,3],y=[2,1,4]))
figure.update_layout(title='QC total intensity')
_plotly_with_export(st,figure,'qc_histogram')
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    app.button(key='qc_histogram_export_open').click().run()
    app.button(key='qc_histogram_export_open').click()
    app.button(key='qc_histogram_preview').click().run()
    assert not app.exception
    assert not app.button(key='qc_histogram_download').disabled
    app.button(key='qc_histogram_export_open').click()
    app.button(key='qc_histogram_download').click().run()
    assert not app.exception
    assert requested == ['SCMM_质控分布图_QC total intensity.svg']
    assert destination.read_bytes() == app.session_state['qc_histogram_matplotlib_preview']['payload']
    assert b'<text' in destination.read_bytes()
    assert app.session_state['image_save_directory'] == str(tmp_path)
