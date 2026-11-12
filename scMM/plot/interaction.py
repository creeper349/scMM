"""Keep interactive Cartesian viewports within their full-data extents."""
import numpy as np
import pandas as pd


def _violin_extent(trace, values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    lower, upper = values.min(), values.max()
    spread = upper - lower
    bandwidth = trace.bandwidth or 0.
    if len(values) > 1 and not bandwidth:
        q1, q3 = np.quantile(values, [.25, .75], method="hazen")
        bandwidth = max(1.059 * min(values.std(ddof=1), (q3 - q1) / 1.349) * len(values) ** -.2, spread / 100)
    if trace.spanmode != "hard":
        lower, upper = lower - 2 * bandwidth, upper + 2 * bandwidth
    if trace.spanmode == "manual" and trace.span:
        lower = trace.span[0] if trace.span[0] is not None else lower
        upper = trace.span[1] if trace.span[1] is not None else upper
    return lower, upper


def constrain_zoom(figure):
    coordinates = {}
    for trace in figure.data:
        if not hasattr(trace, "xaxis"):
            continue
        for dimension in ("x", "y"):
            reference = getattr(trace, f"{dimension}axis", None) or dimension
            name = f"{dimension}axis{reference[1:]}"
            values = getattr(trace, dimension, None)
            if values is None and trace.type == "heatmap":
                values = np.arange(np.shape(trace.z)[1 if dimension == "x" else 0])
            if values is None and trace.type in {"box", "violin"}:
                category_dimension = "y" if trace.orientation == "h" else "x"
                if dimension == category_dimension:
                    values = [trace.name or ""]
            if values is None:
                continue
            values = np.asarray(values).reshape(-1)
            entry = coordinates.setdefault(name, dict(values=[], traces=[]))
            entry["values"].extend(pd.Series(values).tolist() if pd.api.types.is_datetime64_any_dtype(values) else values.tolist())
            entry["traces"].append(trace)

    limits = {}
    for name, entry in coordinates.items():
        axis = getattr(figure.layout, name, None)
        if axis is None:
            figure.update_layout(**{name: {}})
            axis = figure.layout[name]
        values = pd.Series(entry["values"])
        if axis.type == "date" or pd.api.types.is_datetime64_any_dtype(values):
            axis.update(type="date")
            timestamps = pd.to_datetime(values, errors="coerce").dropna()
            if timestamps.empty:
                continue
            numeric = pd.DatetimeIndex(timestamps).as_unit("ms").asi8.astype(float)
        elif axis.type == "category" or (axis.type is None and not pd.api.types.is_numeric_dtype(values)):
            categories = list(dict.fromkeys([*(axis.categoryarray or []), *values.dropna().tolist()]))
            if not categories:
                continue
            axis.update(type="category", categoryorder="array", categoryarray=categories)
            limits[name] = (-.5, len(categories) - .5)
            continue
        else:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
        numeric = numeric[np.isfinite(numeric)]
        if axis.type == "log":
            numeric = np.log10(numeric[numeric > 0])
        if not len(numeric):
            continue
        lower, upper = float(numeric.min()), float(numeric.max())
        dimension = name[0]
        for trace in entry["traces"]:
            trace_values = getattr(trace, dimension, None)
            value_dimension = "x" if getattr(trace, "orientation", None) == "h" else "y"
            if trace.type == "violin" and dimension == value_dimension:
                extent = _violin_extent(trace, trace_values)
                if extent is not None and axis.type != "log":
                    lower, upper = min(lower, extent[0]), max(upper, extent[1])
            if trace.type == "bar" and axis.type != "log":
                if dimension == value_dimension:
                    lower, upper = min(0., lower), max(0., upper)
                elif trace.width is not None:
                    centers = np.asarray(trace_values, float)
                    half_width = np.asarray(trace.width, float) / 2
                    lower = min(lower, float(np.nanmin(centers - half_width)))
                    upper = max(upper, float(np.nanmax(centers + half_width)))
            if trace.type == "heatmap" and len(np.unique(numeric)) > 1:
                half_cell = np.min(np.diff(np.unique(numeric))) / 2
                lower, upper = min(lower, numeric.min() - half_cell), max(upper, numeric.max() + half_cell)
        padding = (upper - lower) * .04 or max(abs(lower) * .04, .5)
        limits[name] = (lower - padding, upper + padding)

    # Shared axes use one common extent, including the marginal panels.
    groups = {name: {name} for name in limits}
    for name in limits:
        match = figure.layout[name].matches
        target = f"{match[0]}axis{match[1:]}" if match else None
        if target in groups:
            group = groups[name] | groups[target]
            for member in group:
                groups[member] = group
    for name, group in groups.items():
        axis = figure.layout[name]
        if axis.type == "category":
            categories = list(dict.fromkeys(value for member in limits if member in group
                                            for value in figure.layout[member].categoryarray or []))
            axis.update(categoryarray=categories)
            lower, upper = -.5, len(categories) - .5
        else:
            lower = min(limits[member][0] for member in group)
            upper = max(limits[member][1] for member in group)
        current = axis.range
        if current:
            current_values = pd.DatetimeIndex(pd.to_datetime(list(current))).as_unit("ms").asi8 if axis.type == "date" else current
            lower, upper = min(lower, *current_values), max(upper, *current_values)
        reversed_axis = axis.autorange == "reversed" or (current and current[0] > current[1])
        bounds = [upper, lower] if reversed_axis else [lower, upper]
        if axis.type == "date":
            bounds = [pd.Timestamp(value, unit="ms").isoformat() for value in bounds]
            allowed = sorted(bounds)
        else:
            allowed = [lower, upper]
        axis.update(minallowed=allowed[0], maxallowed=allowed[1], range=current or bounds, autorange=False)
    return figure
