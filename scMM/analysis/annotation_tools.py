"""External annotation tables and sampled cell-frame MS1 spectra."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms

from scMM.file.io import load_single_file, sum_spec


def read_annotation_table(source, *, header=True, sheet_name=0):
    name = str(getattr(source, "name", source)).lower()
    options = dict(header=0 if header else None)
    if hasattr(source, "seek"):
        source.seek(0)
    if name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(source, sheet_name=sheet_name, **options)
    else:
        try:
            frame = pd.read_csv(source, sep="\t" if name.endswith(".tsv") else ",", dtype=str, encoding="utf-8-sig", **options)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            frame = pd.read_csv(source, sep="\t" if name.endswith(".tsv") else ",", dtype=str, encoding="gb18030", **options)
    frame.columns = frame.columns.astype(str)
    return frame


def orient_annotation_table(frame, *, transpose=False, field_names=True):
    if transpose:
        frame = frame.set_index(frame.columns[0]).T if field_names else frame.T
        frame = frame.rename_axis("feature_id").reset_index()
    frame = frame.copy()
    frame.columns = frame.columns.astype(str)
    return frame


def _key_values(values, *, numeric=False):
    values = pd.Series(values)
    if numeric:
        values = pd.to_numeric(values, errors="coerce").astype("Float64")
    return values.astype("string").str.strip().fillna("").to_numpy()


def _mass_matches(imported, observed, ppm):
    masses = pd.to_numeric(imported, errors="coerce").to_numpy(float)
    order = np.flatnonzero(np.isfinite(masses) & (masses > 0))
    order = order[np.argsort(masses[order])]
    tolerance = float(ppm) * 1e-6
    left = np.searchsorted(masses[order], observed / (1 + tolerance), side="left")
    right = np.searchsorted(masses[order], observed / (1 - tolerance), side="right")
    return order, left, right


def _manual_alignment(data, frame, value_columns, ppm):
    var_keys = {**{str(key): data.var[key] for key in data.var}, "feature_id": data.var_names}
    candidates = []
    for column in frame:
        if column in value_columns and column.lower() not in {"feature_id", "mz", "m/z", "observed_mz"}:
            continue
        target = {"m/z": "mz", "observed_mz": "mz"}.get(column.lower(), column)
        target = target if target in var_keys else "feature_id"
        numeric = pd.api.types.is_numeric_dtype(var_keys[target])
        keys = _key_values(frame[column], numeric=numeric)
        count = np.isin(_key_values(var_keys[target], numeric=numeric), keys[keys != ""]).sum()
        if target == "mz":
            _, left, right = _mass_matches(frame[column], data.var["mz"].to_numpy(float), ppm)
            count = (right > left).sum()
        candidates.append((int(count), column, target))
    count, source_key, var_key = max(candidates, key=lambda item: item[0], default=(0, None, None))
    return ("key" if count else "position"), source_key, var_key


def align_manual_annotations(data, frame, value_column, *, mode="auto", source_key=None, var_key=None, ppm=5.):
    """Return aligned values, matched mask and the chosen alignment method."""
    values = frame[value_column].fillna("").astype(str).str.strip().to_numpy()
    var_keys = {**{str(key): data.var[key] for key in data.var}, "feature_id": data.var_names}
    if mode == "auto":
        mode, source_key, var_key = _manual_alignment(data, frame, [value_column], ppm)
    if mode == "position":
        if len(frame) != data.n_vars:
            raise ValueError(f"按顺序导入需要 {data.n_vars} 条记录，当前有 {len(frame)} 条。请选择匹配列。")
        return pd.Series(values, index=data.var_names), np.ones(data.n_vars, dtype=bool), "按 var 顺序"
    if var_key == "mz":
        observed = data.var["mz"].to_numpy(float)
        # ppm error is relative to the imported reference mass.
        order, left, right = _mass_matches(frame[source_key], observed, ppm)
        names = values[order]
        result = ["/".join(dict.fromkeys(name for name in names[a:b] if name)) for a, b in zip(left, right)]
        matched = right > left
        description = f"{source_key} → mz（{ppm:g} ppm）"
    else:
        numeric = pd.api.types.is_numeric_dtype(var_keys[var_key])
        keys = _key_values(frame[source_key], numeric=numeric)
        grouped = pd.DataFrame({"key": keys, "name": values})
        grouped = grouped[grouped["key"] != ""].groupby("key", sort=False)["name"].agg(
            lambda names: "/".join(dict.fromkeys(name for name in names if name)))
        targets = pd.Series(_key_values(var_keys[var_key], numeric=numeric), index=data.var_names)
        matched = targets.isin(grouped.index).to_numpy()
        result = targets.map(grouped).fillna("").to_numpy()
        description = f"{source_key} → {var_key}"
    return pd.Series(result, index=data.var_names, dtype=object), matched, description


def validate_var_mapping(mapping):
    mapping = {field: "" if pd.isna(target) else str(target).strip() for field, target in mapping.items()}
    if not mapping:
        raise ValueError("请选择至少一个写入字段。")
    targets = list(mapping.values())
    if not all(targets) or len(set(targets)) != len(targets):
        raise ValueError("目标 var 键名需填写完整，并为每个字段指定不同的键名。")
    return mapping


def import_row_positions(selection, length):
    """Parse one-based inclusive ranges such as 1-5,8,10-12."""
    if not selection.strip():
        return np.arange(length)
    positions = []
    for part in selection.replace("，", ",").split(","):
        ends = [int(value.strip()) for value in part.strip().split("-")]
        start, stop = ends[0], ends[-1]
        if len(ends) > 2 or start < 1 or stop < start or stop > length:
            raise ValueError(f"记录行号需位于 1–{length}，支持 1-5,8,10-12。")
        positions.extend(range(start - 1, stop))
    return np.array(list(dict.fromkeys(positions)), dtype=int)


def manual_annotation_columns(data, frame, mapping, *, row_positions=None, mode="auto", source_key=None, var_key=None, ppm=5.):
    mapping = validate_var_mapping(mapping)
    if mode == "auto":
        mode, source_key, var_key = _manual_alignment(data, frame, mapping, ppm)
    positions = np.arange(len(frame)) if row_positions is None else np.asarray(row_positions, dtype=int)
    working = frame if mode == "position" else frame.iloc[positions]
    columns = {}
    for source, target in mapping.items():
        values, matched, description = align_manual_annotations(data, working, source, mode=mode,
                                                               source_key=source_key, var_key=var_key, ppm=ppm)
        if mode == "position":
            matched[:] = False
            matched[positions] = True
        columns[target] = values
    return pd.DataFrame(columns, index=data.var_names), matched, description


def write_manual_annotation_columns(data, columns, matched, *, clear_unmatched=False):
    updated = data.var.copy()
    for target in columns:
        values = pd.Series("", index=data.var_names, dtype=object) if clear_unmatched or target not in updated else updated[target].astype(object).copy()
        values.loc[matched] = columns.loc[matched, target]
        updated[target] = values
    data.var = updated
    return int(np.count_nonzero(matched))


def sum_cell_frames(data, source_paths, *, count=20, seed=42, active_only=True, sample_mode="total",
                    mz_range=(100., 1000.), resolution_200=35000., points_per_fwhm=5., progress=None):
    """Sample exclusively from explicitly selected sources using original frame_id."""
    if not source_paths:
        raise ValueError("请选择至少一个质谱源文件。")
    eligible = data.obs["source_file"].astype(str).isin(source_paths).to_numpy()
    if active_only:
        eligible &= data.cell_keep_mask()
    positions = np.flatnonzero(eligible)
    if not len(positions):
        raise ValueError("所选文件中没有符合筛选条件的细胞帧。")
    rng = np.random.default_rng(seed)
    if sample_mode == "per_file":
        labels = data.obs["source_file"].astype(str).to_numpy()
        pools = [positions[labels[positions] == source] for source in source_paths]
        chosen = np.concatenate([rng.choice(pool, size=min(int(count), len(pool)), replace=False) for pool in pools])
    else:
        chosen = rng.choice(positions, size=min(int(count), len(positions)), replace=False)
    selected = data.obs.iloc[np.sort(chosen)]
    experiment = oms.MSExperiment()
    reports = []
    for index, (source, path) in enumerate(source_paths.items()):
        frames = selected.loc[selected["source_file"].astype(str).eq(source), "frame_id"].to_numpy(dtype=int)
        if progress:
            progress(index / (len(source_paths) + 1), f"读取 {source} · 选用 {len(frames)} 个细胞帧")
        if len(frames):
            raw, _ = load_single_file(str(Path(path).expanduser()), format="auto")
            for frame in frames:
                if frame < 0 or frame >= raw.size():
                    raise ValueError(f"{source} 的 frame_id={frame} 超出文件范围，请确认源文件对应关系。")
                spectrum = raw[int(frame)]
                if spectrum.getMSLevel() != 1:
                    raise ValueError(f"{source} 的 frame_id={frame} 为 MS{spectrum.getMSLevel()}，请确认源文件对应关系。")
                experiment.addSpectrum(spectrum)
            del raw
        reports.append(dict(source_file=source, path=str(path), n_cell_frames=len(frames), frame_ids=frames.tolist()))
    if progress:
        progress(len(source_paths) / (len(source_paths) + 1), f"叠加 {experiment.size()} 个 MS1 细胞帧")
    summed = sum_spec(experiment, mz_range=mz_range, resolution_200=resolution_200, points_per_fwhm=points_per_fwhm)
    mz, intensity = summed.get_peaks()
    if progress:
        progress(1., "细胞帧叠加完成")
    return dict(mz=mz, intensity=intensity, sources=reports, cells=selected.index.astype(str).tolist(),
                resolution_200=float(resolution_200))
