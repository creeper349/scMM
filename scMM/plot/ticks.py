"""Consistent readable numeric ticks for exported Matplotlib figures."""

import re
import math

from matplotlib import ticker


class ConciseScalarFormatter(ticker.ScalarFormatter):
    def __call__(self, value, pos=None):
        label = super().__call__(value, pos)
        return re.sub(r"(\d+)\.(\d+)", lambda match: match[1] + ("." + match[2].rstrip("0") if match[2].rstrip("0") else ""), label)


def tidy_numeric_ticks(figure):
    """Choose nice numeric tick positions; retain feature/group/category labels."""
    for ax in figure.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            if not axis.get_visible():
                continue
            formatter = axis.get_major_formatter()
            if not isinstance(formatter, (ticker.ScalarFormatter, ticker.LogFormatter)):
                continue
            scale = axis.get_scale()
            if scale == "log":
                low, high = sorted(axis.get_view_interval())
                powers = math.floor(math.log10(high)) - math.ceil(math.log10(low)) + 1
                if powers >= 2:
                    axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.,), numticks=7))
                    axis.set_major_formatter(ticker.LogFormatterMathtext(base=10, labelOnlyBase=True))
                else:
                    exponent = math.floor(math.log10(low))
                    nice = [multiple * 10. ** power for power in range(exponent, exponent + 3)
                            for multiple in range(1, 10) if low <= multiple * 10. ** power <= high]
                    if len(nice) >= 2:
                        axis.set_major_locator(ticker.FixedLocator(nice, nbins=6))
                        axis.set_major_formatter(ticker.LogFormatterMathtext(base=10, labelOnlyBase=False, minor_thresholds=(math.inf, math.inf)))
                    else:
                        axis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
                        concise = ConciseScalarFormatter(useMathText=True, useOffset=True)
                        concise.set_powerlimits((-3, 4))
                        axis.set_major_formatter(concise)
                axis.set_minor_formatter(ticker.NullFormatter())
            elif scale == "symlog":
                threshold = axis._scale.linthresh
                axis.set_major_locator(ticker.SymmetricalLogLocator(base=10, linthresh=threshold, subs=(1.,)))
                axis.set_major_formatter(ticker.LogFormatterMathtext(base=10, linthresh=threshold, labelOnlyBase=True))
                axis.set_minor_formatter(ticker.NullFormatter())
            elif scale == "linear":
                axis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True, min_n_ticks=2))
                concise = ConciseScalarFormatter(useMathText=True, useOffset=True)
                concise.set_powerlimits((-3, 4))
                axis.set_major_formatter(concise)
    return figure
