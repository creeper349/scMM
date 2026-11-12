import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from scMM.plot.ticks import tidy_numeric_ticks


def test_linear_ticks_are_nice_and_categorical_labels_are_preserved():
    figure = Figure()
    ax, categories = figure.subplots(1, 2)
    points = ax.scatter([3.14, 8.27], [.001234, .001889])
    before = points.get_offsets().copy()
    categories.set_xticks([0, 1], ["734.5929", "feature A"])
    tidy_numeric_ticks(figure)
    FigureCanvasAgg(figure).draw()
    np.testing.assert_allclose(ax.get_xticks(), np.round(ax.get_xticks()))
    assert [label.get_text() for label in categories.get_xticklabels()] == ["734.5929", "feature A"]
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert len(labels) == len(set(labels))
    assert all(".000" not in label for label in labels)
    np.testing.assert_allclose(points.get_offsets(), before)


def test_log_axes_and_colorbars_use_decade_ticks():
    figure = Figure()
    ax = figure.subplots()
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e2)
    points = ax.scatter([1e-4, 1, 1e2], [1, 2, 3], c=[1e-4, 1, 1e2], norm=LogNorm(1e-4, 1e2))
    colorbar = figure.colorbar(points, ax=ax)
    tidy_numeric_ticks(figure)
    FigureCanvasAgg(figure).draw()
    for ticks in [ax.get_xticks(), colorbar.get_ticks()]:
        np.testing.assert_allclose(np.log10(ticks), np.round(np.log10(ticks)))
    assert all("10^{" in label.get_text() for label in colorbar.ax.get_yticklabels() if label.get_text())


def test_narrow_log_range_has_distinct_readable_ticks():
    figure = Figure()
    ax = figure.subplots()
    ax.set_xscale("log")
    ax.set_xlim(.0021, .0042)
    tidy_numeric_ticks(figure)
    FigureCanvasAgg(figure).draw()
    inside = ax.get_xticks()[(ax.get_xticks() >= .0021) & (ax.get_xticks() <= .0042)]
    assert len(inside) >= 2
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert len(labels) == len(set(labels))
