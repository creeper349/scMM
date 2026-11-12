import numpy as np
import pandas as pd
import plotly.express as px

from scMM.analysis.statistics import compare_groups, differential_summary, numeric_association
from scMM.plot.export import render_figure
from scMM.webapp import _match_distribution_colors


def test_group_tests_and_permutation_association():
    rng = np.random.default_rng(12)
    a, b, c = [rng.normal(mean, .2, 30) for mean in [0, 2, 4]]
    for method in ["ANOVA", "Kruskal-Wallis"]:
        assert compare_groups([a, b, c], method)[1] < .001
    for method in ["Welch", "Mann-Whitney"]:
        assert compare_groups([a, c], method)[1] < .001
    x = np.arange(40.)
    result = numeric_association(x, 2 * x + rng.normal(0, .1, 40), permutations=199)
    assert result["r2"] > .99
    assert result["permutation_p"] <= .01
    assert (result["line"].upper >= result["line"].lower).all()


def test_differential_summary_fc_direction_and_thresholds():
    result = pd.DataFrame({"feature_id": ["up", "down", "weak", "equal", "untested"],
                           "log2_fold_change": [2., -2., .5, 0., 3.],
                           "fdr": [.01, .02, .01, .01, np.nan]})
    summary = differential_summary(result, fdr_limit=.05, log2fc_limit=1.)
    np.testing.assert_allclose(summary.FC, [4., .25, np.sqrt(2), 1., 8.])
    assert summary.direction.tolist() == ["上调", "下调", "上调", "不变", "上调"]
    assert summary.regulation.tolist() == ["上调", "下调", "无显著变化", "无显著变化", "无法判定"]
    assert "FC" not in result


def test_grouped_facets_preserve_positions_and_pvalue_in_svg():
    frame = pd.DataFrame({
        "feature": np.repeat(["f1", "f2"], 24),
        "group": np.tile(np.repeat(["control", "treated", "other"], 8), 2),
        "intensity": np.arange(48.) + 1,
    })
    figure = px.violin(frame, x="group", y="intensity", color="group", facet_col="feature", box=True, points="all")
    _match_distribution_colors(figure)
    for trace in figure.data:
        assert trace.marker.color == trace.fillcolor == trace.line.color
    figure.add_annotation(x=.5, y=1.02, xref="x domain", yref="y domain", text="p=0.001", showarrow=False)
    svg = render_figure(figure, file_format="svg").decode()
    assert "control" in svg and "treated" in svg and "other" in svg
    assert "f1" in svg and "f2" in svg
    assert "p=0.001" in svg
    assert "axes_2" in svg
    assert "<text" in svg
