import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from scMM.plot.distribution import annotate_distribution, significance_label
from scMM.plot.export import render_figure


def example_figure():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"feature": np.repeat(["f1", "f2"], 36),
                          "group": np.tile(np.repeat(["A", "B", "C"], 12), 2),
                          "intensity": rng.lognormal(size=72)})
    figure = px.violin(frame, x="group", y="intensity", color="group", facet_col="feature",
                       points="all", custom_data=["feature", "group"])
    figure.update_yaxes(matches=None)
    comparisons = [dict(feature=feature, groups=pair, p_value=p)
                   for feature in ["f1", "f2"]
                   for pair, p in [(["A", "B"], .0004), (["A", "C"], .005), (["B", "C"], .0167)]]
    annotate_distribution(figure, comparisons)
    return figure


def test_significance_thresholds_and_plotly_brackets():
    assert [significance_label(p) for p in [0, .0001, .001, .01, .05, .051, np.nan]] == [
        "****", "****", "***", "**", "*", "ns", ""]
    figure = example_figure()
    shapes = [shape for shape in figure.layout.shapes if shape.name == "scmm_significance"]
    labels = [label for label in figure.layout.annotations if label.name == "scmm_significance"]
    assert len(shapes) == 18
    assert [label.text for label in labels] == ["***", "**", "*"] * 2
    assert [(shape.x0, shape.x1) for shape in shapes[1:9:3]] == [(0, 1), (0, 2), (1, 2)]
    assert len(set(label.y for label in labels[:3])) == 3
    assert all(label.y < figure.layout["yaxis" + label.yref[1:]].range[1] for label in labels)


def test_distribution_export_type_colors_and_annotations(tmp_path, monkeypatch):
    figure = example_figure()
    source = figure.to_json()
    captured = []
    original = Figure.savefig

    def record(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    colors = {group: dict(shape="#5599ee", points="#ef3333") for group in ["A", "B", "C"]}
    for kind in ["box", "violin"]:
        svg = render_figure(figure, distribution_kind=kind, distribution_colors=colors, significance_mode="p")
        assert b"p=0.0004" in svg and b"p=0.005" in svg and b"p=0.0167" in svg
        assert b"<text" in svg
        canvas = captured[-1]
        assert len(canvas.axes) == 2
        for ax in canvas.axes:
            assert len(ax.patches) == (3 if kind == "box" else 0)
            assert all(patch.get_facecolor() == to_rgba("#5599ee", .8) for patch in ax.patches)
            dots = [item for item in ax.collections if len(item.get_offsets()) == 12]
            assert len(dots) == 3
            for dots_group in dots:
                np.testing.assert_allclose(dots_group.get_facecolors()[0], to_rgba("#ef3333", .8))
            if kind == "violin":
                for collection in ax.collections:
                    if isinstance(collection, LineCollection):
                        np.testing.assert_allclose(collection.get_colors()[0], to_rgba("#5599ee"))
            assert [label.get_text() for label in ax.get_xticklabels()] == ["A", "B", "C"]
            assert len([text for text in ax.texts if text.get_text().startswith("p=")]) == 3
        (tmp_path / f"{kind}.svg").write_bytes(svg)
        png = render_figure(figure, distribution_kind=kind, distribution_colors=colors,
                            significance_mode="stars", file_format="png", dpi=110)
        (tmp_path / f"{kind}.png").write_bytes(png)
    print(f"DISTRIBUTION_PREVIEW={tmp_path / 'box.png'}")
    render_figure(figure, significance_mode="none")
    assert not any(text.get_text().startswith("p=") or text.get_text() in ["*", "**", "***"]
                   for ax in captured[-1].axes for text in ax.texts)
    render_figure(figure, significance_mode="stars", yscale="log")
    assert all(np.isfinite(ax.get_ylim()).all() for ax in captured[-1].axes)
    assert figure.to_json() == source
