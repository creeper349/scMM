"""Group-comparison labels shared by interactive and exported distributions."""

import numpy as np


def significance_label(p_value, mode="stars"):
    if not np.isfinite(p_value):
        return ""
    if mode == "p":
        return f"p={p_value:.3g}"
    for threshold, label in [(1e-4, "****"), (1e-3, "***"), (.01, "**"), (.05, "*")]:
        if p_value <= threshold:
            return label
    return "ns"


def bracket_levels(low, high, count):
    span = high - low or max(abs(high) * .2, 1.)
    return [(high + span * (.1 + .16 * index), span * .035) for index in range(count)]


def annotate_distribution(figure, comparisons, mode="stars"):
    records = []
    for comparison in comparisons:
        trace = next((trace for trace in figure.data if trace.type in {"violin", "box"}
                      and trace.customdata is not None and len(trace.customdata)
                      and str(trace.customdata[0][0]) == str(comparison["feature"])), None)
        if trace is not None and np.isfinite(comparison["p_value"]):
            records.append(dict(comparison, xaxis=trace.xaxis or "x", yaxis=trace.yaxis or "y"))
    meta = dict(figure.layout.meta or {})
    meta.update(scmm_plot="distribution", scmm_comparisons=records, scmm_significance_mode=mode)
    figure.update_layout(meta=meta)
    if mode == "none":
        return figure
    for pair in dict.fromkeys((row["xaxis"], row["yaxis"]) for row in records):
        traces = [trace for trace in figure.data if (trace.xaxis or "x", trace.yaxis or "y") == pair]
        xaxis = figure.layout["xaxis" + pair[0][1:]]
        categories = list(xaxis.categoryarray or dict.fromkeys(str(x) for trace in traces for x in trace.x))
        xaxis.update(type="category", categoryorder="array", categoryarray=categories)
        values = np.concatenate([np.asarray(trace.y, float) for trace in traces])
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        rows = [row for row in records if (row["xaxis"], row["yaxis"]) == pair]
        levels = bracket_levels(values.min(), values.max(), len(rows))
        for row, (height, tip) in zip(rows, levels):
            positions = [categories.index(group) for group in row["groups"] if group in categories]
            if not positions:
                continue
            left, right = min(positions), max(positions)
            for x0, y0, x1, y1 in [(left, height - tip, left, height), (left, height, right, height),
                                  (right, height, right, height - tip)]:
                figure.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, xref=pair[0], yref=pair[1],
                                 line=dict(color="black", width=1), name="scmm_significance")
            label = significance_label(row["p_value"], mode)
            figure.add_annotation(x=(left + right) / 2, y=height + tip, xref=pair[0], yref=pair[1],
                                  text=row.get("prefix", "") + label, showarrow=False, yanchor="bottom",
                                  name="scmm_significance")
        span = values.max() - values.min() or max(abs(values.max()) * .2, 1.)
        figure.layout["yaxis" + pair[1][1:]].update(range=[values.min() - span * .08, levels[-1][0] + span * .16])
    return figure


def draw_comparisons(ax, records, categories, mode, *, fontsize, linewidth, fixed_ylim=False):
    rows = [row for row in records if all(group in categories for group in row["groups"])]
    if mode == "none" or not rows:
        return
    transform = ax.yaxis.get_transform()
    inverse = transform.inverted()
    low, high = transform.transform(ax.get_ylim())
    levels = bracket_levels(low, high, len(rows))
    for row, (height, tip) in zip(rows, levels):
        positions = [categories.index(group) for group in row["groups"]]
        left, right = min(positions), max(positions)
        ax.plot([left, left, right, right], inverse.transform([height - tip, height, height, height - tip]),
                color="black", linewidth=linewidth)
        ax.text((left + right) / 2, inverse.transform(height + tip),
                row.get("prefix", "") + significance_label(row["p_value"], mode),
                ha="center", va="bottom", fontsize=fontsize)
    if not fixed_ylim:
        ax.set_ylim(inverse.transform([low, levels[-1][0] + (high - low or 1.) * .16]))
