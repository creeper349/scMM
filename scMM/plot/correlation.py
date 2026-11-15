"""Ranked feature correlations with selectable metabolite labels."""
import io

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import plotly.graph_objects as go


def correlation_figure(result, labels, *, target, method="pearson", annotation_size=10.):
    valid = np.isfinite(result.correlation.to_numpy(float))
    rows = result.loc[valid]
    y = rows.correlation.to_numpy(float)
    x = np.arange(1, len(rows) + 1)
    colors = np.where(y < 0, "#5B5FEF", np.where(y > 0, "#D7301F", "#888888"))
    labels = np.asarray(labels, str)[valid]
    negative, positive = int((y < 0).sum()), int((y > 0).sum())
    figure = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", marker=dict(size=5, color=colors),
        customdata=rows[["feature_id", "n"]].to_numpy(), showlegend=False,
        hovertemplate="%{customdata[0]}<br>r=%{y:.4f}<br>n=%{customdata[1]}<extra></extra>"))
    figure.update_layout(template="plotly_white", height=620, title=f"{target} · {method.title()} correlations",
                         margin=dict(t=110), title_y=.99,
                         xaxis_title="Features ranked by correlation", yaxis_title=f"{method.title()} r",
                         meta=dict(scmm_plot="correlation_distribution", scmm_correlation_labels=labels.tolist(),
                                   scmm_annotation_size=float(annotation_size),
                                   scmm_correlation_counts=[negative, positive, len(y) - negative - positive]))
    native = draw_correlations(figure, font_family="DejaVu Sans", annotation_size=annotation_size)
    ax = native.axes[0]
    figure.update_xaxes(range=list(ax.get_xlim()), showgrid=False)
    figure.update_yaxes(range=list(ax.get_ylim()), showgrid=False, zeroline=False)
    for left, right, color, name, count in _regions(y):
        figure.add_vrect(x0=left, x1=right, fillcolor=color, opacity=.17, line_width=0, layer="below")
        center = (left + right) / 2
        figure.add_annotation(x=center, y=1.02, yref="paper", text=name, showarrow=False, font=dict(color=color))
        figure.add_annotation(x=center, y=0., text=f"{count / len(y):.1%}", yshift=12,
                              showarrow=False, font=dict(color=color))
    figure.add_hline(y=0, line_dash="dash", line_color="#999999", line_width=1)
    texts = [text for text in ax.texts if text.get_gid() == "feature_label"]
    for text, xx, yy in zip(texts, x[labels != ""], y[labels != ""]):
        tx, ty = text.get_position()
        figure.add_annotation(x=xx, y=yy, text=text.get_text(), showarrow=True,
                              ax=tx, ay=ty, axref="x", ayref="y", arrowhead=0, arrowwidth=.7,
                              xanchor="center", yanchor="middle", font=dict(size=annotation_size * 4 / 3))
    return figure


def _regions(y):
    for mask, color, name in [(y < 0, "#5B5FEF", "Inversely correlated"),
                              (y > 0, "#D7301F", "Positively correlated")]:
        positions = np.flatnonzero(mask) + 1
        if len(positions):
            yield positions[0] - .5, positions[-1] + .5, color, name, len(positions)


def draw_correlations(figure, *, width=8., height=6., dpi=150, font_family="sans-serif", font_size=10.,
                      title="", xlabel="", ylabel="", marker_size=20., alpha=.8, line_width=1.,
                      marker="o", label_size=10., title_size=12., annotation_size=None,
                      grid=False, xlim=None, ylim=None, xscale="linear", yscale="linear", **kwargs):
    from adjustText import adjust_text
    from .ticks import tidy_numeric_ticks
    x, y = np.asarray(figure.data[0].x, float), np.asarray(figure.data[0].y, float)
    labels = figure.layout.meta["scmm_correlation_labels"]
    annotation_size = annotation_size or figure.layout.meta["scmm_annotation_size"]
    with mpl.rc_context({"font.family": font_family, "font.size": font_size, "svg.fonttype": "none"}):
        canvas = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        FigureCanvasAgg(canvas)
        ax = canvas.add_subplot()
        for left, right, color, name, count in _regions(y):
            ax.axvspan(left, right, color=color, alpha=.17, linewidth=0)
            ax.text((left + right) / 2, 1.02, name, color=color, ha="center",
                    transform=ax.get_xaxis_transform(), fontsize=label_size)
            ax.annotate(f"{count / len(y):.1%}", ((left + right) / 2, 0), xytext=(0, 6),
                        textcoords="offset points", ha="center", color=color, fontsize=label_size)
        ax.scatter(x, y, s=marker_size, c=list(figure.data[0].marker.color), alpha=alpha, marker=marker, edgecolors="none")
        ax.axhline(0, color="#999999", ls="--", lw=line_width)
        ax.set(xlabel=xlabel or figure.layout.xaxis.title.text, ylabel=ylabel or figure.layout.yaxis.title.text,
               xscale=xscale, yscale=yscale)
        ax.set_title(title or figure.layout.title.text, pad=36)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.title.set_size(title_size)
        ax.set_xlim(xlim or figure.layout.xaxis.range or (.5, max(len(x), 1) + .5))
        ax.set_ylim(ylim or figure.layout.yaxis.range or (min(-.05, y.min() - .06) if len(y) else -1,
                                                       max(.05, y.max() + .06) if len(y) else 1))
        labelled = [(xx, yy, text) for xx, yy, text in zip(x, y, labels) if text]
        texts = [ax.text(xx, yy, text, fontsize=annotation_size, gid="feature_label") for xx, yy, text in labelled]
        if texts:
            adjust_text(texts, ax=ax, x=x, y=y,
                        target_x=[row[0] for row in labelled], target_y=[row[1] for row in labelled],
                        arrowprops=dict(arrowstyle="-", color="black", lw=line_width * .7),
                        min_arrow_len=0, iter_lim=100)
        if grid:
            ax.grid(True, alpha=.25)
        else:
            ax.grid(False)
        tidy_numeric_ticks(canvas)
        if yscale == "linear":
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4, steps=[1, 2, 5, 10]))
        return canvas


def render_correlations(figure, *, file_format="svg", **params):
    canvas = draw_correlations(figure, **params)
    output = io.BytesIO()
    with mpl.rc_context({"svg.fonttype": "none"}):
        canvas.savefig(output, format=file_format, dpi=params.get("dpi", 300), bbox_inches="tight")
    return output.getvalue()
