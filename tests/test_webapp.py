from streamlit.testing.v1 import AppTest
from pathlib import Path

from scMM.webapp import (
    _gate_selected_cells,
    _matplotlib_plotly_bytes,
    _substep_from_message,
)


def _assert_page_renders(page, *, with_data=False):
    setup = """
import streamlit as st
from scMM.webapp import _state
_state(st)
"""
    if with_data:
        setup += """
import numpy as np
import pandas as pd
from scMM.file.data import CyESIData
if st.session_state.data is None:
    st.session_state.data = CyESIData(
    X=np.random.default_rng(2).random((20, 6)).astype("float32"),
    obs=pd.DataFrame(
        {"group": ["a"] * 10 + ["b"] * 10,
         "source_file": ["f1"] * 10 + ["f2"] * 10},
        index=[f"c{i}" for i in range(20)],
    ),
    var=pd.DataFrame(
        {"mz": np.arange(100, 106, dtype=float)},
        index=[str(i) for i in range(6)],
    ),
)
"""
    script = setup + f"\nfrom scMM.webapp import {page}\n{page}()\n"
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    return app


def test_default_webapp_page_renders():
    path = Path(__file__).parents[1] / "scMM" / "webapp.py"
    app = AppTest.from_file(path, default_timeout=30).run()
    assert not app.exception


def test_batch_dialog_autofills_acquisition_and_preserves_manual_edits(tmp_path, monkeypatch):
    source = tmp_path / "input.mzML"
    source.write_text('''<mzML><spectrum><cvParam accession="MS:1000511" value="1"/>
    <scanList><scan><cvParam accession="MS:1000800" value="50000"/>
    <scanWindow><cvParam accession="MS:1000501" value="80"/>
    <cvParam accession="MS:1000500" value="1500"/></scanWindow></scan></scanList></spectrum></mzML>''')
    monkeypatch.setattr("scMM.file.dialogs.choose_data_paths", lambda: [str(source)])
    app = _assert_page_renders("page_data")
    radio = next(widget for widget in app.radio if widget.label == "输入模式")
    radio.set_value(radio.options[2]).run()
    next(button for button in app.button if button.label == "📁 选择文件夹或多个质谱文件").click().run()
    assert not app.exception
    assert app.number_input(key="processing_mz_min").value == 80.
    assert app.number_input(key="processing_mz_max").value == 1500.
    assert app.number_input(key="processing_resolution").value == 50000.
    app.number_input(key="processing_resolution").set_value(45000.).run()
    assert app.number_input(key="processing_resolution").value == 45000.
    app.text_area(key="input_path").set_value("").run()
    assert app.number_input(key="processing_resolution").value == 35000.
    assert app.number_input(key="processing_mz_max").value == 1000.


def test_correlation_distribution_calculates_labels_and_exports():
    app = _assert_page_renders("page_analysis", with_data=True)
    app.button(key="corr_compute").click().run()
    assert not app.exception and not app.error
    result = app.session_state.correlation_distribution_result["result"]
    assert len(result) == 5
    app.selectbox(key="corr_label_mode").set_value("仅手动")
    app.multiselect(key=f"corr_named_features_{id(result)}").set_value([result.feature_id.iloc[0]]).run()
    assert not app.exception
    import json
    figure = next(json.loads(chart.proto.spec) for chart in app.get("plotly_chart")
                  if json.loads(chart.proto.spec).get("layout", {}).get("meta", {}).get("scmm_plot") == "correlation_distribution")
    assert sum(bool(label) for label in figure["layout"]["meta"]["scmm_correlation_labels"]) == 1
    app.button(key="correlation_distribution_export_open").click().run()
    app.button(key="correlation_distribution_export_open").click()
    app.button(key="correlation_distribution_preview").click().run()
    assert not app.exception and not app.error
    assert b"<text" in app.session_state.correlation_distribution_matplotlib_preview["payload"]


def test_background_training_commits_model_and_report():
    app = _assert_page_renders("page_analysis", with_data=True)
    next(widget for widget in app.number_input if widget.label == "CV folds").set_value(2)
    next(button for button in app.button if button.label == "训练、交叉验证并生成质控").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state.supervised is not None
    assert not app.session_state.result_table.empty
    assert app.session_state.background_job is None


def test_subpages_render_without_data():
    for page in (
        "page_preview",
        "page_preprocess",
        "page_discovery",
        "page_analysis",
        "page_results",
        "page_terminal",
        "page_annotation",
    ):
        _assert_page_renders(page)


def test_analysis_subpages_render_with_data():
    for page in (
        "page_preprocess",
        "page_discovery",
        "page_analysis",
        "page_results",
        "page_annotation",
    ):
        _assert_page_renders(page, with_data=True)


def test_terminal_substep_progress_parsing():
    assert _substep_from_message("Shared spectrum: 2/4 files") == (
        "Shared spectrum",
        50,
    )
    assert _substep_from_message("sample.mzML: Feature extraction block 3/10") == (
        "sample.mzML: Feature extraction",
        30,
    )
    assert _substep_from_message("Building summed spectrum") == (
        "Building summed spectrum",
        None,
    )
    assert _substep_from_message("叠加质谱图: 2/5 files") == (
        "叠加质谱图",
        40,
    )


def test_matplotlib_svg_export_preserves_text_objects():
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2], mode="lines+markers"))
    payload = _matplotlib_plotly_bytes(
        figure,
        file_format="svg",
        width=5,
        height=4,
        dpi=150,
        font_family="sans-serif",
        font_size=10,
        title="Example title",
        xlabel="x label",
        ylabel="y label",
        cmap="viridis",
        marker_size=20,
        alpha=0.8,
        line_width=1.5,
        grid=True,
        legend=False,
    )

    text = payload.decode("utf-8")
    assert "<text" in text
    assert "Example title" in text


def test_gate_selection_accepts_streamlit_customdata_formats():
    selection = {
        "selection": {
            "points": [
                {"customdata": {"cell_position": 7}},
                {"customdata": [3]},
                {"point_index": 1},
            ]
        }
    }

    selected = _gate_selected_cells(selection, plotted_positions=[10, 11, 12])
    assert selected.tolist() == [3, 7, 11]


def test_gate_selection_fraction_and_active_export_color(monkeypatch):
    import streamlit as st

    points = []
    original_chart = st.plotly_chart

    def chart(*args, **kwargs):
        result = original_chart(*args, **kwargs)
        if kwargs.get("key") == "cell_gate_plot":
            return {"selection": {"points": points}}
        return result

    monkeypatch.setattr(st, "plotly_chart", chart)
    app = _assert_page_renders("page_preprocess", with_data=True)
    assert any("0 / 20（0.00%" in caption.value for caption in app.caption)
    points.extend([{"customdata": [2]}, {"customdata": [7]}])
    app.run()
    assert not app.exception
    assert any("2 / 20（10.00%" in caption.value for caption in app.caption)
    points.clear()
    app.run()
    assert any("0 / 20（0.00%" in caption.value for caption in app.caption)
    next(button for button in app.button if button.key == "cell_gate_export_open").click().run()
    assert not app.exception
    app.color_picker(key="cell_gate_active_color").set_value("#ef4444")
    next(button for button in app.button if button.key == "cell_gate_export_open").click()
    next(button for button in app.button if button.key == "cell_gate_preview").click().run()
    assert not app.exception
    payload = app.session_state["cell_gate_matplotlib_preview"]["payload"]
    assert b"#ef4444" in payload


def test_targeted_analysis_and_matplotlib_preview_dialog():
    app = _assert_page_renders("page_analysis", with_data=True)
    next(widget for widget in app.radio if widget.label == "差异分析模式").set_value("靶向差异分析")
    app.run()
    assert not app.exception
    assert any(widget.label == "靶向特征" for widget in app.multiselect)
    next(widget for widget in app.selectbox if widget.key == "target_compare_feature_b").set_value("1")
    next(button for button in app.button if button.key == "target_compare_run").click()
    app.run()
    assert not app.exception
    assert any("A: 0" in message.value and "B: 1" in message.value for message in app.success)
    assert any("使用方法：" in caption.value for caption in app.caption)
    next(button for button in app.button if button.key == "targeted_distribution_export_open").click()
    app.run()
    assert not app.exception
    next(button for button in app.button if button.key == "targeted_distribution_preview").click()
    app.run()
    assert not app.exception


def test_gate_label_filtering_is_controlled_by_table(monkeypatch):
    import json
    import streamlit as st

    positions = [1, 4]
    original_chart = st.plotly_chart

    def chart(*args, **kwargs):
        result = original_chart(*args, **kwargs)
        if kwargs.get("key") == "cell_gate_plot":
            return {"selection": {"points": [{"customdata": [index]} for index in positions]}}
        return result

    monkeypatch.setattr(st, "plotly_chart", chart)
    app = AppTest.from_string('''
import streamlit as st
import numpy as np
from scMM.file.data import CyESIData
from scMM.webapp import _state, page_preprocess
_state(st)
if st.session_state.data is None:
    st.session_state.data = CyESIData(X=np.random.default_rng(2).random((20, 3)))
page_preprocess()
''', default_timeout=30).run()
    next(button for button in app.button if button.label == "写入圈门标签").click().run()
    assert not app.exception
    column = "cell_keep_manual_gate"
    assert app.session_state.data.obs[column].eq(0).sum() == 2
    assert app.session_state.data.cell_keep_mask().all()

    for enabled in [True, False, True]:
        editor = next(table for table in app.dataframe if "启用筛选" in table.value.columns)
        states = app._tree.get_widget_states()
        state = next((state for state in states.widgets if state.id == editor.proto.id), None)
        if state is None:
            state = states.widgets.add()
            state.id = editor.proto.id
        state.string_value = json.dumps({"edited_rows": {0: {"启用筛选": enabled}},
                                        "added_rows": [], "deleted_rows": []})
        app._run(widget_state=states)
        assert not app.exception
        assert app.session_state.data.cell_keep_mask().sum() == (18 if enabled else 20)
        assert app.session_state.data.obs[column].eq(0).sum() == 2

    positions[:] = [2]
    next(button for button in app.button if button.label == "写入圈门标签").click().run()
    assert not app.exception
    assert app.session_state.data.cell_keep_mask().sum() == 19


def test_targeted_distribution_export_controls():
    app = _assert_page_renders("page_analysis", with_data=True)
    next(widget for widget in app.radio if widget.label == "差异分析模式").set_value("靶向差异分析").run()
    app.selectbox(key="target_plot_type").set_value("箱线图").run()
    app.multiselect(key="target_extra_pairs").set_value([("a", "b")]).run()
    assert not app.exception
    app.button(key="targeted_distribution_export_open").click().run()
    assert not app.exception
    app.selectbox(key="targeted_distribution_distribution_kind").set_value("violin")
    app.selectbox(key="targeted_distribution_export_significance").set_value("p 值")
    app.color_picker(key="targeted_distribution_export_shape_a").set_value("#123456")
    app.color_picker(key="targeted_distribution_export_points_a").set_value("#fedcba")
    app.button(key="targeted_distribution_export_open").click()
    app.button(key="targeted_distribution_preview").click().run()
    assert not app.exception
    assert not app.error
    svg = app.session_state["targeted_distribution_matplotlib_preview"]["payload"]
    assert b"#123456" in svg and b"#fedcba" in svg and b"p=" in svg


def test_untargeted_distribution_can_hide_statistics_in_box_export():
    app = _assert_page_renders("page_analysis", with_data=True)
    next(button for button in app.button if button.label == "差异分析（BH-FDR）").click().run()
    app.button(key="differential_distribution_export_open").click().run()
    assert not app.exception
    app.selectbox(key="differential_distribution_distribution_kind").set_value("box")
    app.checkbox(key="differential_distribution_export_statistics").uncheck()
    app.color_picker(key="differential_distribution_export_shape_a").set_value("#123456")
    app.button(key="differential_distribution_export_open").click()
    app.button(key="differential_distribution_preview").click().run()
    assert not app.exception
    assert not app.error
    svg = app.session_state["differential_distribution_matplotlib_preview"]["payload"]
    assert b"#123456" in svg and b"p=" not in svg


def test_heatmap_actions_for_cells_features_and_files():
    app = _assert_page_renders("page_analysis", with_data=True)
    next(button for button in app.button if button.key == "correlation_run").click()
    app.run()
    assert not app.exception
    assert "correlation_heatmap_result" in app.session_state

    next(widget for widget in app.multiselect if widget.key == "differential_features").set_value(["0", "1", "2"])
    next(button for button in app.button if button.key == "differential_run").click()
    app.run()
    assert not app.exception
    assert "differential_heatmap_result" in app.session_state

    qc = _assert_page_renders("page_preprocess", with_data=True)
    next(button for button in qc.button if button.key == "file_qc_run").click()
    qc.run()
    assert not qc.exception
    assert "file_qc_heatmap_result" in qc.session_state


def test_shap_ui_displays_all_three_plot_types():
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.file.data import CyESIData
from scMM.analysis import SupervisedAnalyzer
from scMM.webapp import _render_shap_plots
if "shap_analyzer" not in st.session_state:
    rng = np.random.default_rng(2)
    data = CyESIData(X=rng.random((50, 4)).astype("float32"),
        obs=pd.DataFrame({"group": ["a"] * 25 + ["b"] * 25}, index=[str(i) for i in range(50)]),
        var=pd.DataFrame(index=["f0", "f1", "f2", "f3"]))
    analyzer = SupervisedAnalyzer(data, "group")
    analyzer.evaluate("logistic", cv=2)
    analyzer.explain_shap(max_background=8, max_samples=6)
    st.session_state.shap_analyzer = analyzer
_render_shap_plots(st, st.session_state.shap_analyzer)
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    for kind in ["蜂群图", "特征重要性条形图", "特征依赖图"]:
        app.selectbox(key="shap_plot_kind").set_value(kind).run()
        assert not app.exception
        assert not app.error
        assert len(app.get("image")) == 1


def test_sidebar_layer_selection_and_processing():
    import numpy as np

    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.file.data import CyESIData
from scMM.webapp import _layer_selector
if "data" not in st.session_state:
    st.session_state.data = CyESIData(X=np.asarray([[1., 2.], [3., 4.]], dtype="float32"),
        obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["f1", "f2"]),
        layers={"custom": np.full((2, 2), 10., dtype="float32")})
data = st.session_state.data
_layer_selector(st, data)
if st.button("normalize"):
    data.normalize("total", scale=1.)
    st.rerun()
st.write(float(data.X.sum()))
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    app.selectbox(key="analysis_layer").set_value("custom").run()
    assert app.session_state.data.uns["active_layer"] == "custom"
    assert app.session_state.data.X.sum() == 40
    app.button[0].click().run()
    assert not app.exception
    assert app.selectbox(key="analysis_layer").value == "normalized_total"
    assert app.session_state.data.X.sum() == 2
    np.testing.assert_allclose(app.session_state.data.layers["custom"], 10.)


def test_gating_numeric_and_density_log_colorbars():
    app = _assert_page_renders("page_preprocess", with_data=True)
    for mode in ["二维点密度", "特征 / obs 数值"]:
        app.selectbox(key="gate_color_mode").set_value(mode).run()
        app.selectbox(key="gate_color_scale").set_value("log10").run()
        assert not app.exception
        assert not app.error


def test_untargeted_differential_summary_is_displayed_and_exported():
    app = _assert_page_renders("page_analysis", with_data=True)
    next(button for button in app.button if button.label == "差异分析（BH-FDR）").click().run()
    assert not app.exception
    assert not app.error
    assert {"direction", "regulation", "FC", "FDR"} <= set(app.session_state.result_table.columns)
    assert any({"FC", "FDR"} <= set(table.value.columns) for table in app.dataframe)
