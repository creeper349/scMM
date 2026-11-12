import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scMM.plot.diagnostics import qc_distribution
from scMM.plot.interaction import constrain_zoom


def test_numeric_log_focus_and_zoom_bounds_are_stable():
    figure = go.Figure(go.Scatter(x=[0., 10., np.nan], y=[1., 1000., np.nan]))
    figure.update_yaxes(type="log")
    figure.update_xaxes(range=[3., 6.])
    constrain_zoom(figure)
    assert figure.layout.xaxis.range == (3., 6.)
    assert figure.layout.xaxis.minallowed < 0 < 10 < figure.layout.xaxis.maxallowed
    assert figure.layout.yaxis.minallowed < 0 < 3 < figure.layout.yaxis.maxallowed
    previous = figure.to_dict()
    constrain_zoom(figure)
    assert figure.to_dict() == previous


def test_facets_categories_shared_qc_axes_and_heatmaps():
    frame = pd.DataFrame({"feature": ["a"] * 6 + ["b"] * 6, "group": ["control", "case"] * 6,
                          "intensity": np.arange(12)})
    figure = px.violin(frame, x="group", y="intensity", facet_col="feature", color="group")
    constrain_zoom(figure)
    assert figure.layout.xaxis.categoryarray == ("control", "case")
    assert figure.layout.xaxis.minallowed == -.5
    assert figure.layout.xaxis.maxallowed == 1.5
    assert figure.layout.yaxis.range == figure.layout.yaxis2.range
    missing_groups = px.violin(pd.DataFrame({'feature':['a','a','b','b'], 'group':['control','case','case','other'],
                                            'intensity':[1,2,3,4]}), x='group',y='intensity',facet_col='feature')
    constrain_zoom(missing_groups)
    assert missing_groups.layout.xaxis.categoryarray == missing_groups.layout.xaxis2.categoryarray == ('control', 'case', 'other')
    assert missing_groups.layout.xaxis.maxallowed == 2.5
    sample = np.random.default_rng(4).normal(size=40)
    violin = px.violin(pd.DataFrame({'group': ['a']*20 + ['b']*20, 'value': sample}), x='group', y='value', color='group')
    constrain_zoom(violin)
    assert violin.layout.yaxis.minallowed < -3.2679755689
    assert violin.layout.yaxis.maxallowed > 3.4178864908
    qc = qc_distribution(pd.DataFrame({"source_file": ["a"] * 10 + ["b"] * 10,
                                       "intensity": np.arange(20)}), "intensity")
    constrain_zoom(qc)
    assert qc.layout.xaxis.range == qc.layout.xaxis2.range
    assert qc.layout.yaxis.type == "category"
    assert qc.layout.yaxis2.minallowed <= 0
    heatmap = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], x=["a", "b"], y=["c", "d"]))
    heatmap.update_yaxes(autorange="reversed")
    constrain_zoom(heatmap)
    assert heatmap.layout.yaxis.range == (1.5, -.5)
    numeric = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]]))
    constrain_zoom(numeric)
    assert numeric.layout.xaxis.minallowed <= -.5
    assert numeric.layout.xaxis.maxallowed >= 1.5


def test_pies_and_dates():
    pie = go.Figure(go.Pie(labels=["a", "b"], values=[1, 2]))
    constrain_zoom(pie)
    assert pie.layout.xaxis.minallowed is None
    dates = px.line(x=pd.date_range("2026-01-01", periods=5), y=np.arange(5))
    constrain_zoom(dates)
    assert dates.layout.xaxis.type == "date"
    assert isinstance(dates.layout.xaxis.range[0], str)
    assert pd.Timestamp(dates.layout.xaxis.minallowed) < pd.Timestamp("2026-01-01")
    assert pd.Timestamp(dates.layout.xaxis.maxallowed) > pd.Timestamp("2026-01-05")
