"""Continuous fold-change colour and FDR-sized volcano plots."""
import io

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import plotly.graph_objects as go


def _sizes(values, lower, upper):
    span = upper - lower
    return 10 + (np.asarray(values) - lower) / span * 110 if span > 0 else np.full_like(values, 10., dtype=float)


def labelled_features(automatic, overrides, maximum, mode="自动＋手动"):
    if mode == "仅自动":
        return list(automatic)[:maximum]
    manual = [feature for feature, enabled in overrides.items() if enabled]
    automatic = [] if mode == "仅手动" else [feature for feature in automatic if overrides.get(feature, True)]
    return list(dict.fromkeys([*reversed(manual), *automatic]))[:maximum]


def selected_features(selection, feature_ids):
    state = selection.get("selection", {})
    chosen = []
    for point in state.get("points", []):
        position = point.get("point_index", point.get("point_number"))
        if point.get("curve_number", 0) == 0 and position is not None and 0 <= int(position) < len(feature_ids):
            chosen.append(str(feature_ids[int(position)]))
    return list(dict.fromkeys(chosen))


def volcano_figure(result, labels, *, group_a, group_b, annotation_size=10.):
    x = result["log2_fold_change"].to_numpy(float)
    fdr = result["fdr"].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(fdr) & (fdr >= 0) & (fdr <= 1)
    x, y = x[valid], -np.log10(fdr[valid].clip(min=np.finfo(float).tiny))
    labels = np.asarray(labels, str)[valid]
    extent = (float(np.max(np.abs(x))) if len(x) else 0.) or 1.
    lower, upper = (float(y.min()), float(y.max())) if len(y) else (0., 0.)
    sizes = _sizes(y, lower, upper)
    legend_values = [value for value in [1, 2, 5, 10] if lower <= value <= upper]
    if not legend_values and len(y):
        legend_values = [upper]
    figure = go.Figure(go.Scatter(
        x=x, y=y, mode="markers", showlegend=False,
        customdata=result.loc[valid, ["feature_id", "mz", "mean_a", "mean_b", "p_value", "fdr"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>m/z=%{customdata[1]:.6f}<br>log2 FC=%{x:.4g}<br>FDR=%{customdata[5]:.4g}<extra></extra>",
        marker=dict(size=sizes, sizemode="area", sizeref=1, opacity=.75, color=x,
                    colorscale=[[float(value), mpl.colors.to_hex(mpl.colormaps["coolwarm"](value))] for value in np.linspace(0, 1, 33)],
                    cmin=-extent, cmax=extent, colorbar=dict(title=dict(text=f"log2 Fold Change ({group_a} / {group_b})", side="right"),
                                                         x=1.02, y=.65, len=.7, thickness=18))))
    for value in legend_values:
        figure.add_scatter(x=[None], y=[None], mode="markers", name=f"{value:g}",
                           marker=dict(size=float(_sizes([value], lower, upper)[0]), sizemode="area", sizeref=1,
                                       color="gray", opacity=.6))
    figure.update_layout(template="plotly_white", height=650, title=f"{group_a} vs {group_b}",
                         xaxis_title=f"log2 Fold Change ({group_a} / {group_b})", yaxis_title="−log10(FDR)",
                         xaxis=dict(showgrid=False, showline=True, mirror=True, zeroline=False),
                         yaxis=dict(showgrid=False, showline=True, mirror=True, zeroline=False),
                         margin=dict(r=190), legend=dict(title="−log10(FDR)", x=1.02, y=.18, xanchor="left"),
                         meta=dict(scmm_plot="volcano", scmm_cmap="coolwarm", scmm_volcano_labels=labels.tolist(),
                                   scmm_volcano_annotation_size=float(annotation_size),
                                   scmm_volcano_size_range=[lower, upper], scmm_volcano_legend=legend_values))
    native = draw_volcano(figure)
    figure.update_xaxes(range=list(native.axes[0].get_xlim()))
    figure.update_yaxes(range=list(native.axes[0].get_ylim()))
    for text, xx, yy in zip(native.axes[0].texts, x[labels != ""], y[labels != ""]):
        tx, ty = text.get_position()
        figure.add_annotation(x=xx, y=yy, xref="x", yref="y", text=text.get_text(), showarrow=True,
                              ax=tx, ay=ty, axref="x", ayref="y", xanchor="center", yanchor="middle",
                              arrowhead=0, arrowwidth=.7, arrowcolor="black", standoff=0, startstandoff=0,
                              font=dict(size=annotation_size * 4 / 3, color="black"))
    return figure


def draw_volcano(figure, *, width=10., height=7., dpi=150, font_family="Microsoft YaHei", font_size=10.,
                 title="", xlabel="", ylabel="", cmap="coolwarm", marker_size=20., alpha=.75,
                 line_width=1., marker="o", grid=False, legend=True, label_size=10., title_size=12., legend_size=9.,
                 xscale="linear", yscale="linear", xlim=None, ylim=None, annotation_size=None, **kwargs):
    from adjustText import adjust_text
    from .ticks import tidy_numeric_ticks

    trace = figure.data[0]
    x, y = np.asarray(trace.x, float), np.asarray(trace.y, float)
    metadata = figure.layout.meta
    annotation_size = metadata.get("scmm_volcano_annotation_size", 10.) if annotation_size is None else annotation_size
    with mpl.rc_context({"font.family": font_family, "font.size": font_size, "svg.fonttype": "none"}):
        canvas = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        FigureCanvasAgg(canvas)
        grid_spec = canvas.add_gridspec(2, 2, width_ratios=[20, .8], height_ratios=[7, 1.5])
        ax = canvas.add_subplot(grid_spec[:, 0])
        color_axis, legend_axis = canvas.add_subplot(grid_spec[0, 1]), canvas.add_subplot(grid_spec[1, 1])
        scatter = ax.scatter(x, y, c=x, s=np.asarray(trace.marker.size) * marker_size / 20.,
                             norm=mpl.colors.TwoSlopeNorm(vmin=trace.marker.cmin, vcenter=0, vmax=trace.marker.cmax),
                             cmap=cmap, alpha=alpha, marker=marker, edgecolors="none")
        colorbar = canvas.colorbar(scatter, cax=color_axis)
        colorbar.set_label(figure.layout.xaxis.title.text, size=label_size)
        color_axis.xaxis.set_visible(False)
        color_axis.tick_params(labelsize=font_size)
        lower, upper = metadata["scmm_volcano_size_range"]
        legend_axis.set_axis_off()
        if legend:
            values = metadata["scmm_volcano_legend"]
            handles = [legend_axis.scatter([], [], s=float(_sizes([value], lower, upper)[0]) * marker_size / 20.,
                                          color="gray", alpha=.6) for value in values]
            if handles:
                legend_axis.legend(handles, [f"{value:g}" for value in values], title="−log10(FDR)",
                                   loc="center", frameon=False, fontsize=legend_size, title_fontsize=legend_size)
        ax.set(title=title or figure.layout.title.text, xlabel=xlabel or figure.layout.xaxis.title.text,
               ylabel=ylabel or "−log10(FDR)", xscale=xscale, yscale=yscale)
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.tick_params(labelsize=font_size)
        ax.grid(grid, alpha=.25) if grid else ax.grid(False)
        if xlim is not None or figure.layout.xaxis.range is not None:
            ax.set_xlim(*(xlim if xlim is not None else figure.layout.xaxis.range))
        if ylim is not None or figure.layout.yaxis.range is not None:
            ax.set_ylim(*(ylim if ylim is not None else figure.layout.yaxis.range))
        labelled = [(xx, yy, label) for xx, yy, label in zip(x, y, metadata["scmm_volcano_labels"]) if label]
        offset = (float(np.ptp(y)) if len(y) else 0.) or 1.
        texts = [ax.text(xx, yy + index * offset * 1e-6, label, fontsize=annotation_size)
                 for index, (xx, yy, label) in enumerate(labelled)]
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="black", lw=line_width * .7),
                        target_x=[item[0] for item in labelled], target_y=[item[1] for item in labelled], iter_lim=100)
        tidy_numeric_ticks(canvas)
        return canvas


def render_volcano(figure, *, file_format="svg", **params):
    canvas = draw_volcano(figure, **params)
    output = io.BytesIO()
    with mpl.rc_context({"svg.fonttype": "none"}):
        canvas.savefig(output, format=file_format, dpi=params.get("dpi", 300), bbox_inches="tight")
    return output.getvalue()
