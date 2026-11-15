import numpy as np
import pandas as pd
from scipy import sparse

from scMM.file.data import CyESIData
from scMM.analysis.statistics import feature_correlations
from scMM.plot.correlation import correlation_figure
from scMM.plot.export import render_figure


def test_correlations_respect_filters_nan_zero_and_self_exclusion():
    x = np.arange(8.)
    data = CyESIData(X=np.column_stack([x, -x, np.ones(8), x * 2]),
                     var=pd.DataFrame(index=["positive", "negative", "constant", "hidden"]))
    data.set_feature_filter("test", [1, 1, 1, 0])
    data.set_cell_filter("test", [1] * 7 + [0])
    result = feature_correlations(data, x, include_zeros=False)
    assert result.feature_id.tolist() == ["negative", "positive", "constant"]
    np.testing.assert_allclose(result.correlation[:2], [-1, 1])
    assert result.n.tolist() == [6, 6, 6]
    assert np.isnan(result.correlation.iloc[2])
    data.X = sparse.csr_matrix(data.X)
    result = feature_correlations(data, x, method="spearman", exclude_feature="positive")
    assert result.feature_id.tolist() == ["negative", "constant"]


def test_ranked_plot_and_matplotlib_keep_labels_and_background():
    result = pd.DataFrame({"feature_id": ["a", "b", "c", "d"],
                           "correlation": [-.4, -.2, .3, .5], "n": [30] * 4})
    figure = correlation_figure(result, ["ATP", "", "", "AMP"], target="obs value")
    assert figure.layout.meta["scmm_correlation_counts"] == [2, 2, 0]
    assert list(figure.data[0].customdata[:, 0]) == result.feature_id.tolist()
    assert len(figure.layout.shapes) == 3
    arrows = [a for a in figure.layout.annotations if a.showarrow]
    assert [(a.x, a.y) for a in arrows] == [(1, -.4), (4, .5)]
    payload = render_figure(figure, file_format="svg")
    assert b"<text" in payload and b"ATP" in payload and b"50.0%" in payload


def test_spectrum_overview_has_fixed_equal_width_top_and_linked_range():
    import plotly.graph_objects as go
    from scMM.plot.spectrum_overview import spectrum_overview_figure, spectrum_overview_html
    detail = go.Figure(go.Scatter(x=[100., 200., 300.], y=[0., 10., 0.], mode="lines"))
    detail.update_layout(xaxis_range=[190., 210.], yaxis_range=[0., 12.])
    figure = spectrum_overview_figure(detail)
    assert figure.layout.xaxis.fixedrange and figure.layout.yaxis.fixedrange
    assert figure.layout.xaxis.domain == figure.layout.xaxis2.domain
    assert figure.layout.xaxis.range == (100., 300.)
    assert (figure.layout.shapes[0].x0, figure.layout.shapes[0].x1) == (190., 210.)
    assert figure.layout.yaxis.domain[1] - figure.layout.yaxis.domain[0] < .3
    markup = spectrum_overview_html(detail)
    assert "plotly_relayout" in markup and "shapes[0].x0" in markup
