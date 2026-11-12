"""Matplotlib set relationships and grouped compound-class composition."""
import warnings

import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from upsetplot import UpSet


class _CompatibleUpSet(UpSet):
    def plot_matrix(self, ax):
        # UpSetPlot 0.9 leaves matrix styles missing under pandas Copy-on-Write.
        original_scatter = ax.scatter

        def scatter_with_styles(*args, **kwargs):
            facecolors = pd.Series(kwargs["facecolors"]).fillna(self._facecolor)
            kwargs.update(facecolors=facecolors,
                          edgecolors=pd.Series(kwargs["edgecolors"]).fillna(facecolors),
                          linewidths=pd.Series(kwargs["linewidths"]).fillna(1.),
                          linestyles=pd.Series(kwargs["linestyles"]).fillna("solid"))
            return original_scatter(*args, **kwargs)

        ax.scatter = scatter_with_styles
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="A value is trying to be set on a copy.*", module=r"upsetplot\.plotting")
                super().plot_matrix(ax)
        finally:
            ax.scatter = original_scatter


def group_set_figure(membership, *, kind="UpSet", composition=None, width=9., height=6., font_family="Microsoft YaHei",
                     font_size=10., cmap="tab10", alpha=.6, line_width=1., top=30, percentages=False):
    with mpl.rc_context({"font.family": font_family, "font.size": font_size, "svg.fonttype": "none"}):
        figure = Figure(figsize=(width, height))
        colors = mpl.colormaps[cmap](np.linspace(0, 1, max(2, len(membership.columns))))
        if kind == "Venn":
            from matplotlib_venn import venn2, venn3
            if len(membership.columns) not in (2, 3):
                raise ValueError("Venn 图支持 2 或 3 组，请调整所选组别。")
            sets = [set(membership.index[membership[group]]) for group in membership]
            plot = (venn2 if len(sets) == 2 else venn3)(sets, set_labels=list(membership.columns),
                                                     set_colors=colors[:len(sets)], alpha=alpha, ax=figure.add_subplot())
            for patch in plot.patches:
                if patch is not None:
                    patch.set_linewidth(line_width)
                    patch.set_edgecolor("white")
        elif kind == "UpSet":
            if len(membership.columns) < 2:
                raise ValueError("UpSet 集合关系需要至少两个组别。")
            values = pd.Series(np.ones(len(membership)), index=pd.MultiIndex.from_frame(membership.astype(bool)))
            plot = _CompatibleUpSet(values, subset_size="count", sort_by="cardinality", sort_categories_by="input",
                         max_subset_rank=top, show_counts=True, element_size=None,
                         facecolor=mpl.colors.to_hex(mpl.colors.to_rgba(colors[0], alpha), keep_alpha=True), other_dots_color=.15)
            plot.style_subsets(linewidth=line_width)
            plot.plot(fig=figure)
        else:
            values = composition.copy()
            if percentages:
                values = values.div(values.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
            axis = figure.add_subplot()
            values.plot.bar(stacked=True, ax=axis, colormap=cmap, alpha=alpha, linewidth=line_width,
                            edgecolor="white", rot=35)
            axis.set_ylabel("Composition (%)" if percentages else "Unique compounds")
            axis.set_xlabel("")
            axis.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
            figure.set_layout_engine("constrained")
        if kind != "Venn":
            from .ticks import tidy_numeric_ticks
            tidy_numeric_ticks(figure)
        return figure
