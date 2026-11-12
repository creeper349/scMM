"""Matplotlib rendering of the workbench's Plotly figures."""

import io
import re
import textwrap

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure


def render_figure(figure, *, file_format="svg", width=8, height=6, dpi=300,
                  font_family="sans-serif", font_size=10, title="", xlabel="", ylabel="",
                  cmap="viridis", marker_size=20, alpha=0.8, line_width=1.5,
                  grid=False, legend=True, marker="o", jitter=0.08, violin_width=0.75,
                  bins=40, label_size=10, title_size=12, legend_size=9,
                  legend_location="best", xscale="linear", yscale="linear",
                  xlim=None, ylim=None, annotation_size=None):
    if isinstance(figure.layout.meta, dict) and figure.layout.meta.get("scmm_plot") == "volcano":
        params = locals().copy()
        params.pop("figure")
        from .volcano import render_volcano
        return render_volcano(figure, **params)
    with mpl.rc_context({"font.family": font_family, "font.size": font_size, "svg.fonttype": "none"}):
        pairs = list(dict.fromkeys(
            (getattr(trace, "xaxis", None) or "x", getattr(trace, "yaxis", None) or "y")
            for trace in figure.data
        )) or [("x", "y")]
        axes, categories = {}, {}
        domains = {}
        for pair in pairs:
            xaxis = getattr(figure.layout, "xaxis" + pair[0][1:], None)
            yaxis = getattr(figure.layout, "yaxis" + pair[1][1:], None)
            domains[pair] = (tuple(xaxis.domain or [0, 1]) if xaxis else (0, 1),
                             tuple(yaxis.domain or [0, 1]) if yaxis else (0, 1))
        xdomains = sorted(set(domain[0] for domain in domains.values()))
        ydomains = sorted(set(domain[1] for domain in domains.values()), reverse=True)
        canvas = Figure(figsize=(max(width, len(xdomains) * 3.5), max(height, len(ydomains) * 3)), dpi=dpi,
                        layout="constrained")
        canvas.get_layout_engine().set(w_pad=.12, h_pad=.12, wspace=.12, hspace=.22)
        gridspec = canvas.add_gridspec(len(ydomains), len(xdomains),
                                      width_ratios=[end - start for start, end in xdomains],
                                      height_ratios=[end - start for start, end in ydomains])
        rng = np.random.default_rng(42)

        def clean(text):
            return re.sub(r"<[^>]+>", "", str(text or "").replace("<br>", "\n"))

        for pair in pairs:
            xaxis = getattr(figure.layout, "xaxis" + pair[0][1:], None)
            yaxis = getattr(figure.layout, "yaxis" + pair[1][1:], None)
            xd, yd = domains[pair]
            ax = canvas.add_subplot(gridspec[ydomains.index(yd), xdomains.index(xd)])
            axes[pair] = ax
            category_order = list(xaxis.categoryarray or []) if xaxis else []
            for trace in figure.data:
                if (getattr(trace, "xaxis", None) or "x", getattr(trace, "yaxis", None) or "y") != pair:
                    continue
                if trace.type in {"violin", "box"} and getattr(trace, "orientation", None) != "h" and trace.y is not None:
                    for value in trace.x if trace.x is not None else []:
                        if str(value) not in category_order:
                            category_order.append(str(value))
            categories[pair] = category_order

        for pair, ax in axes.items():
            xaxis = getattr(figure.layout, "xaxis" + pair[0][1:], None)
            yaxis = getattr(figure.layout, "yaxis" + pair[1][1:], None)
            if xaxis and xaxis.matches:
                shared = next((other for ref, other in axes.items() if ref[0] == xaxis.matches), None)
                if shared is not None and shared is not ax:
                    ax.sharex(shared)
            if yaxis and yaxis.matches:
                shared = next((other for ref, other in axes.items() if ref[1] == yaxis.matches), None)
                if shared is not None and shared is not ax:
                    ax.sharey(shared)

        previous = {}
        pie_axes = set()
        for index, trace in enumerate(figure.data):
            pair = (getattr(trace, "xaxis", None) or "x", getattr(trace, "yaxis", None) or "y")
            ax = axes[pair]
            name = trace.name or ""
            raw_x, raw_y = getattr(trace, "x", None), getattr(trace, "y", None)
            x = list(raw_x) if raw_x is not None else []
            y = list(raw_y) if raw_y is not None else []
            color = getattr(getattr(trace, "line", None), "color", None) or getattr(getattr(trace, "marker", None), "color", None)
            if not isinstance(color, str) or color.startswith("rgba"):
                color = mpl.colormaps[cmap](index / max(1, len(figure.data) - 1))
            if isinstance(color, str) and color.startswith("rgb("):
                color = tuple(float(v) / 255 for v in color[4:-1].split(","))
            if trace.type == "pie":
                labels = [clean(label) for label in trace.labels]
                colors = mpl.colormaps[cmap](np.linspace(.1, .9, len(labels)))
                wedges, _, _ = ax.pie(trace.values, labels=None, colors=colors, autopct="%1.0f%%",
                                     wedgeprops={"width": 1 - (trace.hole or 0), "linewidth": line_width, "edgecolor": "white"})
                if legend:
                    ax.legend(wedges, labels, loc="upper left", bbox_to_anchor=(1, 1), fontsize=legend_size)
                pie_axes.add(ax)
            elif trace.type in {"violin", "box"}:
                if getattr(trace, "orientation", None) == "h" or (trace.y is None and trace.x is not None):
                    values = np.asarray(x, float)
                    values = values[np.isfinite(values)]
                    horizontal = [t for t in figure.data if t.type == "box" and
                                  (getattr(t, "xaxis", None) or "x", getattr(t, "yaxis", None) or "y") == pair]
                    position = next(i for i, item in enumerate(horizontal) if item is trace)
                    if len(values):
                        ax.boxplot([values], positions=[position], vert=False, widths=violin_width,
                                   showfliers=False, patch_artist=True,
                                   boxprops={"facecolor": color, "alpha": alpha, "linewidth": line_width},
                                   medianprops={"color": color, "linewidth": line_width},
                                   whiskerprops={"color": color, "linewidth": line_width})
                    ax.set_yticks(range(len(horizontal)), [t.name or "" for t in horizontal])
                    continue
                order = categories[pair]
                for group in dict.fromkeys(map(str, x)):
                    values = np.asarray([value for label, value in zip(x, y) if str(label) == group], dtype=float)
                    values = values[np.isfinite(values)]
                    if not len(values):
                        continue
                    position = order.index(group)
                    if trace.type == "violin" and len(values) > 1 and np.ptp(values) > 0:
                        parts = ax.violinplot([values], positions=[position], widths=violin_width, showmedians=True)
                        for body in parts["bodies"]:
                            body.set_facecolor(color); body.set_alpha(alpha); body.set_linewidth(line_width)
                        for key, part in parts.items():
                            if key != "bodies":
                                part.set_linewidth(line_width)
                        if getattr(getattr(trace, "box", None), "visible", False):
                            ax.boxplot([values], positions=[position], widths=violin_width * .15,
                                       showfliers=False, boxprops={"linewidth": line_width},
                                       medianprops={"linewidth": line_width})
                    else:
                        ax.boxplot([values], positions=[position], widths=violin_width * .6,
                                   boxprops={"linewidth": line_width}, whiskerprops={"linewidth": line_width},
                                   medianprops={"linewidth": line_width}, showfliers=False)
                    show_points = getattr(trace, "points", None) or getattr(trace, "boxpoints", None)
                    if show_points:
                        ax.scatter(position + rng.uniform(-jitter, jitter, len(values)), values,
                                   s=marker_size, alpha=alpha, marker=marker, color=color, linewidths=line_width * .3)
                rotate = 35 if len(order) > 3 or any(len(str(value)) > 10 for value in order) else 0
                ax.set_xticks(range(len(order)), [textwrap.fill(str(value), 14) for value in order],
                              rotation=rotate, ha="right" if rotate else "center")
                ax.set_xlim(-.6, max(.6, len(order) - .4))
            elif trace.type in {"scatter", "scattergl"}:
                mode = trace.mode or "lines"
                if categories[pair]:
                    x = [categories[pair].index(str(value)) if str(value) in categories[pair] else value for value in x]
                if getattr(trace, "fill", None) == "tonexty" and pair in previous:
                    ax.fill_between(np.asarray(x, dtype=float), previous[pair], np.asarray(y, dtype=float), color=color, alpha=.2)
                if "lines" in mode:
                    dash = getattr(trace.line, "dash", None)
                    ax.plot(x, y, linewidth=line_width, alpha=alpha, color=color,
                            linestyle="--" if dash in {"dash", "dot", "dashdot"} else "-", label=name)
                if "markers" in mode:
                    colors = getattr(trace.marker, "color", None)
                    trace_meta = trace.meta if isinstance(trace.meta, dict) else {}
                    if "scmm_color_values" in trace_meta:
                        colors = np.asarray(trace_meta["scmm_color_values"], dtype=float)
                    numeric_colors = isinstance(colors, (list, tuple, np.ndarray)) and np.asarray(colors).dtype.kind in "fiu"
                    meta = figure.layout.meta if isinstance(figure.layout.meta, dict) else {}
                    norm = None
                    if numeric_colors and meta.get("scmm_color_scale") in {"log10", "symlog"}:
                        color_values = np.asarray(colors, float)
                        finite = color_values[np.isfinite(color_values)]
                        if len(finite):
                            low, high = float(finite.min()), float(finite.max())
                            if low == high:
                                low, high = (low / 1.01, high * 1.01) if low > 0 else (low - .01, high + .01)
                            norm = mpl.colors.LogNorm(vmin=low, vmax=high) if meta["scmm_color_scale"] == "log10" else mpl.colors.SymLogNorm(
                                linthresh=meta.get("scmm_linthresh", 1.0), vmin=low, vmax=high, base=10)
                    scatter_x = np.asarray(x, dtype=float) + rng.uniform(-jitter, jitter, len(x)) if categories[pair] else x
                    scatter = ax.scatter(scatter_x, y, s=marker_size, alpha=alpha, marker=marker,
                                         c=colors if numeric_colors else [color], cmap=cmap if numeric_colors else None,
                                         norm=norm,
                                         linewidths=line_width * .3, label=name)
                    if numeric_colors:
                        canvas.colorbar(scatter, ax=ax, label=meta.get("scmm_color_label") or "")
                if "text" in mode and trace.text is not None:
                    for xx, yy, text in zip(x, y, trace.text):
                        if text:
                            ax.annotate(clean(text), (xx, yy), fontsize=label_size)
                previous[pair] = np.asarray(y, dtype=float) if y and all(v is not None for v in y) else y
            elif trace.type == "histogram":
                related = [t for t in figure.data if t.type == "histogram" and (t.xaxis or "x", t.yaxis or "y") == pair]
                pooled = np.concatenate([np.asarray(t.x, float) for t in related])
                edges = np.histogram_bin_edges(pooled[np.isfinite(pooled)], bins=bins)
                ax.hist(x, bins=edges, alpha=alpha, color=color, label=name)
            elif trace.type == "bar":
                bar_width = getattr(trace, "width", None)
                ax.bar(x, y, width=bar_width if bar_width is not None else .8,
                       color=color, alpha=trace.opacity if trace.opacity is not None else alpha, label=name)
            elif trace.type == "heatmap":
                z = np.asarray(trace.z, dtype=float)
                image = ax.imshow(z, cmap=cmap, aspect="auto", vmin=trace.zmin, vmax=trace.zmax,
                                  interpolation="nearest", origin="upper")
                canvas.colorbar(image, ax=ax)
                if x: ax.set_xticks(range(len(x)), x)
                if y: ax.set_yticks(range(len(y)), y)
                if getattr(trace, "texttemplate", None):
                    for row, column in np.ndindex(z.shape):
                        if np.isfinite(z[row, column]):
                            ax.text(column, row, f"{z[row, column]:.2f}", ha="center", va="center", fontsize=label_size,
                                    color="white" if image.norm(z[row, column]) < .4 else "black")

        for annotation in figure.layout.annotations or []:
            xref, yref = annotation.xref or "x", annotation.yref or "y"
            pair = (xref.replace(" domain", ""), yref.replace(" domain", ""))
            if pair in axes:
                ax = axes[pair]
                if "domain" in xref and "domain" in yref:
                    ax.text(annotation.x, annotation.y, clean(annotation.text), transform=ax.transAxes,
                            ha="center", va="bottom", fontsize=label_size)
                elif xref == pair[0] and yref == pair[1]:
                    xx = categories[pair].index(str(annotation.x)) if str(annotation.x) in categories[pair] else annotation.x
                    ax.annotate(clean(annotation.text), (xx, annotation.y), ha="center", fontsize=label_size)
            elif xref == "paper" and yref == "paper":
                candidates = [(pair, domain) for pair, domain in domains.items()
                              if abs(annotation.x - sum(domain[0]) / 2) < .03
                              and abs(annotation.y - domain[1][1]) < .06]
                if candidates:
                    pair, _ = candidates[0]
                    axes[pair].set_title(textwrap.fill(clean(annotation.text), 36), fontsize=title_size, pad=36)
                else:
                    canvas.text(annotation.x, annotation.y, clean(annotation.text), ha="center", fontsize=label_size)
        for shape in figure.layout.shapes or []:
            if shape.type != "line":
                continue
            for pair, ax in axes.items():
                xref, yref = shape.xref or "x", shape.yref or "y"
                if xref == pair[0] and yref == pair[1] + " domain" and shape.x0 == shape.x1:
                    ax.axvline(shape.x0, color="#666666", linestyle="--", linewidth=line_width)
                elif yref == pair[1] and xref == pair[0] + " domain" and shape.y0 == shape.y1:
                    ax.axhline(shape.y0, color="#666666", linestyle="--", linewidth=line_width)
        for pair, ax in axes.items():
            if ax in pie_axes:
                ax.set_axis_off()
                continue
            xaxis = getattr(figure.layout, "xaxis" + pair[0][1:], None)
            yaxis = getattr(figure.layout, "yaxis" + pair[1][1:], None)
            ax.set_xlabel(xlabel or (xaxis.title.text if xaxis else "") or "", fontsize=label_size)
            ax.set_ylabel(ylabel or (yaxis.title.text if yaxis else "") or "", fontsize=label_size)
            for descriptor, axis, setter in [(xaxis, ax.xaxis, ax.set_xticks), (yaxis, ax.yaxis, ax.set_yticks)]:
                if descriptor is not None:
                    if descriptor.tickvals is not None and descriptor.ticktext is not None:
                        setter(descriptor.tickvals, [textwrap.fill(str(value), 22) for value in descriptor.ticktext])
                    if descriptor.showticklabels is False:
                        axis.set_tick_params(labelbottom=False, labelleft=False, labelright=False)
                    if descriptor.visible is False:
                        axis.set_visible(False)
                    if descriptor.tickangle:
                        for label in axis.get_ticklabels():
                            label.set_rotation(-descriptor.tickangle)
                            label.set_ha("right")
            if xaxis and yaxis and xaxis.visible is False and yaxis.visible is False:
                ax.set_axis_off()
            if xaxis and xaxis.range is not None: ax.set_xlim(*xaxis.range)
            if yaxis and yaxis.range is not None: ax.set_ylim(*yaxis.range)
            if xaxis and xaxis.autorange == "reversed": ax.invert_xaxis()
            if yaxis and yaxis.autorange == "reversed": ax.invert_yaxis()
            if xscale != "linear": ax.set_xscale(xscale)
            if yscale != "linear": ax.set_yscale(yscale)
            if xlim: ax.set_xlim(*xlim)
            if ylim: ax.set_ylim(*ylim)
            ax.tick_params(labelsize=font_size)
            if grid: ax.grid(True, alpha=.25)
            else: ax.grid(False)
            if legend and ax.get_legend_handles_labels()[0]:
                ax.legend(loc=legend_location, fontsize=legend_size)
        canvas.suptitle(clean(title), fontsize=title_size)
        from .ticks import tidy_numeric_ticks
        tidy_numeric_ticks(canvas)
        output = io.BytesIO()
        canvas.savefig(output, format=file_format, dpi=dpi, bbox_inches="tight")
        return output.getvalue()
