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
