from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from scMM.analysis.heatmaps import expression_matrix, heatmap_figure
from scMM.file.data import CyESIData
from scMM.plot.diagnostics import point_density, qc_distribution, spectrum_curve
from scMM.plot.export import render_figure


def test_qc_shared_counts_and_horizontal_boxes_export(tmp_path, monkeypatch):
    metrics = pd.DataFrame({"source_file": ["control"] * 40 + ["treated"] * 40,
                            "intensity": np.r_[np.arange(40), np.arange(40) + 20]})
    metrics.loc[79, "intensity"] = 120
    figure = qc_distribution(metrics, "intensity", bins=20)
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert sum(bars[0].y) == sum(bars[1].y) == 40
    np.testing.assert_allclose(bars[0].x, bars[1].x)
    captured = []
    original = Figure.savefig

    def record(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    svg = render_figure(figure, file_format="svg", title="QC shared bins")
    canvas = captured[-1]
    assert len(canvas.axes) == 2
    assert all(ax.get_subplotspec() is not None for ax in canvas.axes)
    histogram = next(ax for ax in canvas.axes if len(ax.patches) == 40)
    marginal = next(ax for ax in canvas.axes if len(ax.patches) == 2)
    assert sum(patch.get_height() for patch in histogram.patches) == 80
    assert marginal.get_position().y0 > histogram.get_position().y1
    np.testing.assert_allclose(marginal.get_xlim(), histogram.get_xlim())
    for _, group in metrics.groupby("source_file"):
        median = group.intensity.median()
        assert any(np.asarray(line.get_xdata()).shape == (2,) and np.allclose(line.get_xdata(), median)
                   for line in marginal.lines)
    (tmp_path / "qc.svg").write_bytes(svg)
    (tmp_path / "qc.png").write_bytes(render_figure(figure, file_format="png", dpi=110, title="QC shared bins"))
    # Also cover existing Plotly histogram+horizontal marginal figures.
    legacy = px.histogram(metrics, x="intensity", color="source_file", marginal="box")
    assert b"<svg" in render_figure(legacy, file_format="svg")
    print(f"QC_PREVIEW={tmp_path / 'qc.png'}")


def test_multiple_panels_use_nonoverlapping_subplots(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"feature": np.repeat([f"metabolite_{i}_long_label" for i in range(6)], 30),
                          "group": np.tile(np.repeat(["control group", "treated group", "other group"], 10), 6),
                          "intensity": rng.normal(5, 1, 180)})
    figure = px.violin(frame, x="group", y="intensity", color="group", facet_col="feature", facet_col_wrap=3,
                       points="all", box=True, color_discrete_sequence=px.colors.qualitative.Plotly)
    for xaxis, yaxis in dict.fromkeys((trace.xaxis or "x", trace.yaxis or "y") for trace in figure.data):
        figure.add_annotation(x=.5, y=1.02, xref=f"{xaxis} domain", yref=f"{yaxis} domain",
                              text="ANOVA: p=0.00012", showarrow=False)
    captured = []
    original = Figure.savefig

    def record(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    png = render_figure(figure, file_format="png", dpi=110, title="Six-panel targeted analysis")
    canvas = captured[-1]
    backend = FigureCanvasAgg(canvas)
    backend.draw()
    assert len(canvas.axes) == 6
    assert all(ax.get_subplotspec() is not None for ax in canvas.axes)
    boxes = [ax.get_tightbbox(backend.get_renderer()) for ax in canvas.axes]
    assert all(not left.overlaps(right) for left, right in combinations(boxes, 2))
    for ax in canvas.axes:
        tickboxes = [label.get_window_extent(backend.get_renderer()) for label in ax.get_xticklabels() if label.get_visible()]
        assert all(not left.overlaps(right) for left, right in combinations(tickboxes, 2))
    (tmp_path / "six_panels.png").write_bytes(png)
    (tmp_path / "six_panels.svg").write_bytes(render_figure(figure, file_format="svg", title="Six panels"))
    print(f"PANELS_PREVIEW={tmp_path / 'six_panels.png'}")


def test_spectrum_markers_remain_on_reduced_curve():
    mz = np.linspace(100, 101, 10001)
    intensity = np.exp(-((mz - 100.54321) / .0003) ** 2)
    markers = [100.54321, 100.5433]
    plot_x, plot_y = spectrum_curve(mz, intensity, 100, markers)
    assert all(value in plot_x for value in markers)
    np.testing.assert_allclose(np.interp(markers, plot_x, plot_y), np.interp(markers, mz, intensity))
    assert plot_y.max() == intensity.max()


def test_density_and_pseudobulk_heatmaps(tmp_path):
    density = point_density(np.r_[np.zeros(40), 10.], np.r_[np.zeros(40), 10.])
    assert density[0] > density[-1]
    data = CyESIData(
        X=np.asarray([[1, 2, 4], [3, 4, 6], [2, 4, 8], [6, 8, 12]], dtype=np.float32),
        obs=pd.DataFrame({"source_file": ["a.raw", "a.raw", "b.raw", "b.raw"],
                          "group": ["control", "control", "treated", "treated"]}, index=list("abcd")),
        var=pd.DataFrame({"mz": [100, 200, 300], "name": ["A", "B", "C"]}, index=["f1", "f2", "f3"]),
    )
    matrix, labels = expression_matrix(data, ["f1", "f2", "f3"], pseudobulk=True, aggregate="mean", obs_key="group")
    np.testing.assert_allclose(matrix.loc["a.raw"], [2, 3, 5])
    assert labels.tolist() == ["control", "treated"]
    correlation = matrix.T.corr() ** 2
    np.testing.assert_allclose(correlation, 1)
    figure = heatmap_figure(correlation, title="Pseudobulk R2", row_labels=labels, column_labels=labels,
                            annotate=True, cmap="Viridis", value_range=(0, 1))
    svg = render_figure(figure, file_format="svg")
    assert b"control" in svg and b"treated" in svg and b"1.00" in svg
    (tmp_path / "pseudobulk.png").write_bytes(render_figure(figure, file_format="png", dpi=110))
    print(f"HEATMAP_PREVIEW={tmp_path / 'pseudobulk.png'}")
