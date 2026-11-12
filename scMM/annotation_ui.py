"""Streamlit annotation page; heavy calculations live in analysis.annotation."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scMM.analysis.annotation import (
    ADDUCTS, accept_annotations, annotate_features, annotation_report, collect_msms,
    deisotope_features, export_annotations, load_annotation_database,
    sync_isotope_flags, write_annotations, standardize_lipid_annotations,
)
from scMM.util.annotation import DEFAULT_ADDUCTS_POS, DEFAULT_ADDUCTS_NEG


def _invalidate_group_sets(st):
    for key in ("annotation_sets_result", "qc_sets_result"):
        st.session_state.pop(key, None)


def _var_mapping_editor(st, fields, *, key, defaults, prefix=""):
    from scMM.analysis.annotation_tools import validate_var_mapping
    rows = pd.DataFrame({"write": [field in defaults for field in fields], "source": fields,
                         "target": [defaults.get(field, f"{prefix}{field}") for field in fields]})
    edited = st.data_editor(rows, hide_index=True, use_container_width=True, disabled=["source"], key=key,
                            height=330,
                            column_config={"write": st.column_config.CheckboxColumn("写入", required=True),
                                           "source": st.column_config.TextColumn("来源字段 / 列", width="medium"),
                                           "target": st.column_config.TextColumn("目标 var 键名", required=True, width="medium")})
    selected = edited.loc[edited["write"]]
    try:
        return validate_var_mapping(dict(zip(selected["source"], selected["target"])))
    except ValueError as exc:
        st.info(str(exc))
        return {}


def _number(st, name, key, default, minimum, maximum, *, step=None, format=None):
    from scMM.webapp import _parameter
    value, help_text = _parameter(f"annotation.{key}", default)
    value = default if value is None else value
    kwargs = dict(min_value=minimum, max_value=maximum, value=value, key=f"ann_{key}", help=help_text)
    if step is not None:
        kwargs["step"] = step
    if format is not None:
        kwargs["format"] = format
    return st.number_input(name, **kwargs)


def _path_input(st, label, key, *, multiple=False, types=None):
    from scMM.file.dialogs import choose_local_path
    def browse():
        try:
            selected = choose_local_path(filetypes=[("Annotation files", " ".join(f"*.{suffix}" for suffix in types)), ("All files", "*.*")], multiple=multiple)
            if selected:
                st.session_state[f"ann_path_{key}"] = "\n".join(selected) if multiple else selected
        except Exception as exc:
            st.error(f"无法打开本机选择器：{exc}")
    left, right = st.columns([5, 1])
    right.button("选择文件", key=f"ann_browse_{key}", on_click=browse)
    widget = left.text_area if multiple else left.text_input
    path = widget(label, key=f"ann_path_{key}", help="本机文件路径；多个文件每行一个，也可在下方一次上传。")
    uploaded = st.file_uploader(f"上传{label}", type=types, accept_multiple_files=multiple, key=f"ann_upload_{key}")
    return path, uploaded


def _materialize(path, uploads, directory, prefix):
    if uploads is None:
        uploads = []
    if not isinstance(uploads, list):
        uploads = [uploads]
    files = [Path(p.strip().strip('"')) for p in path.splitlines() if p.strip()]
    for i, uploaded in enumerate(uploads):
        folder = Path(directory) / f"{prefix}_{i}"
        folder.mkdir()
        target = folder / Path(uploaded.name).name
        with target.open("wb") as stream:
            stream.write(uploaded.getbuffer())
        files.append(target)
    return files


_COLUMNS = [
    ("feature_id", "Feature ID", 135), ("accepted", "接受注释", 85), ("is_isotope", "同位素", 75),
    ("name", "候选物质名称", 230), ("formula", "分子式", 140), ("category", "类别", 140),
    ("original_name", "代表谱的原始库名称", 280), ("isomer_count", "合并名称数", 110),
    ("report_level", "报告层级", 140),
    ("source_library", "来源谱库", 230),
    ("observed_mz", "Observed m/z", 135), ("theoretical_mz", "Theoretical m/z", 145),
    ("ppm_error", "ppm error", 115), ("adduct", "Adduct", 135),
    ("isotope_similarity", "Isotope dot product", 165), ("has_msms", "有 MS/MS", 100),
    ("msms_similarity", "MS/MS dot product", 160), ("combined_score", "综合分数", 110),
    ("msms_status", "MS/MS 匹配状态", 180),
    ("n_matched_fragments", "匹配碎片数", 110), ("match_level", "匹配级别", 110),
    ("msms_scan_number", "MS/MS scan number", 270), ("matched_fragments", "匹配碎片（JSON）", 300),
    ("database_id", "Database ID", 140), ("mirror", "镜像谱", 125),
]


def apply_table_edit(data, table, row_id, column, value):
    if column == "is_isotope":
        feature = table.at[row_id, "feature_id"]
        data.var.loc[feature, "annotation_isotope"] = np.uint8(value)
        sync_isotope_flags(data)
        table = data.uns["annotation"]["table"]
        if "feature_keep_annotation_isotope" in data.var:
            data.var["feature_keep_annotation_isotope"] = (1 - data.var["annotation_isotope"].astype(np.uint8)).astype(np.uint8)
    elif column == "accepted":
        if not table.at[row_id, "is_isotope"] and table.at[row_id, "name"] and table.at[row_id, "match_level"] in {"MS1", "MS/MS"}:
            table.at[row_id, column] = bool(value)
    elif column in {"name", "formula", "category", "adduct", "database_id"}:
        table.at[row_id, column] = str(value)
        if column == "name":
            table.at[row_id, "report_level"] = "manual"
    data.uns["annotation"]["table"] = table


def _dynamic_table(st, data, table):
    import streamlit.components.v1 as components
    component = components.declare_component("scmm_annotation_table", path=str(Path(__file__).with_name("components") / "annotation_table"))
    row = st.session_state.get("ann_table_row", 0)
    row = min(row, max(0, len(table) - 1))
    chunk = table.iloc[row:row+100].reset_index()
    records = chunk.astype(object).where(chunk.notna(), None).to_dict("records")
    columns = [dict(key=k, label=label, width=width, editable=k in {"name", "formula", "category", "adduct", "database_id"}) for k, label, width in _COLUMNS]
    event = component(dataset=str(id(data)), total=len(table), row=row, columns=columns, records=records,
                      selected_feature=st.session_state.get("ann_spectrum_feature"),
                      key=f"ann_table_{id(data)}", default=None)
    if event and event.get("event_id") != st.session_state.get("ann_last_event"):
        st.session_state.ann_last_event = event["event_id"]
        st.session_state.ann_table_row = event.get("row", row)
        if event["action"] == "edit":
            apply_table_edit(data, table, event["id"], event["column"], event["value"])
            _invalidate_group_sets(st)
        elif event["action"] == "mirror":
            st.session_state.ann_mirror = event["id"]
        elif event["action"] == "select":
            st.session_state.ann_spectrum_feature = table.at[event["id"], "feature_id"]
        st.rerun()


def _mirror_dialog(st, annotation, row_id):
    from scMM.plot.annotation import mirror_spectrum
    from scMM.plot.interaction import constrain_zoom
    from scMM.webapp import _matplotlib_plotly_bytes, _image_filename, _save_image

    @st.dialog("实测 / 标准 MS/MS 对比", width="large")
    def show():
        row = annotation["table"].loc[row_id]
        observed = annotation["spectra"]["observed"][row["spectrum_key"]]
        reference = annotation["spectra"]["reference"].get(row["library_id"], np.empty((0, 2)))
        figure = mirror_spectrum(observed, reference, f"{row['name']} · {row['observed_mz']:.5f}")
        st.caption(f"代表标准谱：{row.get('original_name', row['name'])} · dot product={row['msms_similarity']:.4g} · 匹配碎片={row['n_matched_fragments']}。此名称记录标准谱来源，报告采用总组成名称。")
        st.plotly_chart(constrain_zoom(figure), use_container_width=True)
        controls, preview = st.columns([1, 1.2])
        with controls:
            fmt = st.selectbox("导出格式", ["svg", "png"], key="ann_mirror_format")
            width = st.number_input("宽度（inch）", 3., 20., 8., key="ann_mirror_width")
            height = st.number_input("高度（inch）", 2., 20., 5., key="ann_mirror_height")
            linewidth = st.number_input("线宽（pt）", .1, 10., 1., key="ann_mirror_line")
            fontsize = st.number_input("字号", 6., 30., 10., key="ann_mirror_font")
            dpi = st.number_input("DPI", 72, 1200, 300, key="ann_mirror_dpi")
            params = dict(width=width, height=height, dpi=dpi, font_family="sans-serif", font_size=fontsize,
                          title=figure.layout.title.text, xlabel="m/z", ylabel="Relative intensity (%)", cmap="viridis",
                          marker_size=10., alpha=1., line_width=linewidth, grid=False, legend=True)
            signature = repr(params) + fmt + row_id
            if st.button("预览", key="ann_mirror_preview"):
                png = _matplotlib_plotly_bytes(figure, file_format="png", **params)
                payload = _matplotlib_plotly_bytes(figure, file_format=fmt, **params) if fmt == "svg" else png
                st.session_state.ann_mirror_cache = dict(signature=signature, png=png, payload=payload)
            cache = st.session_state.get("ann_mirror_cache", {})
            current = cache.get("signature") == signature
            if st.button("下载", disabled=not current, key="ann_mirror_download"):
                _save_image(st, cache["payload"], _image_filename("msms_mirror", fmt, figure=figure))
        if current:
            preview.image(cache["png"], width="stretch")
    show()


def _results(st, data, terminal):
    from scMM.webapp import _plotly_with_export
    from scMM.group_sets_ui import render_group_sets
    import plotly.graph_objects as go
    annotation = data.uns.get("annotation")
    if annotation is None:
        st.info("设置数据库和匹配参数后，点击“开始注释 / 刷新”。")
        _spectrum_assistance(st, data, terminal)
        render_group_sets(st, data, key="annotation_sets", composition=True)
        return
    table = annotation["table"]
    if annotation.get("name_policy") != "regular_lipids_v2":
        table = standardize_lipid_annotations(table)
        annotation.update(table=table, name_policy="regular_lipids_v2")
    if "reference_msms_available" not in table:
        reference = annotation.get("spectra", {}).get("reference", {})
        table["reference_msms_available"] = [bool(len(reference.get(k, []))) for k in table["library_id"]]
    if "msms_status" not in table:
        table["msms_status"] = np.select(
            [~table["reference_msms_available"], ~table["has_msms"], table["match_level"].eq("MS/MS")],
            ["无参考 MS/MS", "无实测 MS/MS", "MS/MS 匹配"], default="匹配未达阈值")
    left, middle, right = st.columns(3)
    if left.button("Accept All", help="接受全部候选，包括同一特征的多个候选。"):
        accept_annotations(table)
        _invalidate_group_sets(st)
    if middle.button("Accept First", help="每个特征接受综合评分最高的第一个候选。"):
        accept_annotations(table, first=True)
        _invalidate_group_sets(st)
    if right.button("标准化脂质名称", help="按 LIPID MAPS species 层级合并为类别、总碳数、总 DBE 和附加氧数；代表谱按 MS/MS dot product 优先、匹配碎片数次优选择。"):
        table = standardize_lipid_annotations(table)
        data.uns["annotation"]["table"] = table
    st.caption("脂质自动按总组成合并；原始候选保存在 candidate_metrics 中。代表谱按 dot product 降序、匹配碎片数降序选择。特征按 m/z 升序，候选按综合分数降序排列。同位素位于末尾，文本可编辑。")
    _dynamic_table(st, data, table)
    row_id = st.session_state.pop("ann_mirror", None)
    if row_id in table.index:
        _mirror_dialog(st, annotation, row_id)
    a, b = st.columns(2)
    if a.button("去同位素", help="沿用可逆 feature_keep 标签，将同位素特征标记为隐藏。"):
        data.set_feature_filter("annotation_isotope", ~data.var["annotation_isotope"].astype(bool).to_numpy())
        a.success("已写入 var['feature_keep_annotation_isotope']，可在预处理页恢复。")
    with st.expander("选择注释写入字段与 var 键名", expanded=True):
        mapping = _var_mapping_editor(st, table.columns.tolist(), key="ann_var_mapping", defaults={"name": "annotation"}, prefix="annotation_")
        if st.button("写入 var", disabled=not mapping):
            count = write_annotations(data, table, mapping=mapping)
            _invalidate_group_sets(st)
            st.success(f"已将 {count} 个特征的 {len(mapping)} 项注释指标写入 var。")
    if st.button("导出注释表格"):
        st.download_button("下载注释表格 CSV", export_annotations(table).to_csv(index=False).encode("utf-8-sig"),
                           "scmm_annotations.csv", "text/csv", on_click="ignore")
    st.caption("规则脂质按总组成输出；复杂色素、天然产物、衍生物及小分子保留各自原名并逐行导出。candidate_metrics 保留原始指标。")
    _spectrum_assistance(st, data, terminal)
    levels, categories = annotation_report(data, table)
    left, right = st.columns(2)
    if levels:
        figure = go.Figure(go.Pie(labels=list(levels), values=list(levels.values()), hole=.35))
        figure.update_layout(title="匹配覆盖率（独立特征）")
        _plotly_with_export(st, figure, "annotation_coverage", container=left)
    if categories:
        figure = go.Figure(go.Pie(labels=[k or "未分类" for k in categories], values=list(categories.values()), hole=.35))
        figure.update_layout(title="已接受注释的物质类别（特征-类别计数）")
        _plotly_with_export(st, figure, "annotation_categories", container=right)
    render_group_sets(st, data, key="annotation_sets", annotation=annotation, composition=True)


def _manual_annotations(st, data):
    from scMM.analysis.annotation_tools import (read_annotation_table, orient_annotation_table, import_row_positions,
                                               manual_annotation_columns, write_manual_annotation_columns)
    from scMM.webapp import _choice, _parameter
    st.caption("CSV / Excel 每行可对应一个特征，也可每行存放一个注释字段。通过特征 ID、已有 var 元信息或 m/z 对齐，并预览写入结果。")
    path, uploaded = _path_input(st, "手动注释表", "manual", types=["csv", "tsv", "xlsx", "xls"])
    a, b = st.columns(2)
    value, help_text = _parameter("annotation.manual_header", True)
    header = a.checkbox("首行是列名", value, key="ann_manual_header", help=help_text)
    value, help_text = _parameter("annotation.manual_sheet", "")
    sheet = b.text_input("Excel 工作表（空为第一个）", value, key="ann_manual_sheet", help=help_text)
    if st.button("读取注释表", key="ann_manual_read", disabled=not (path or uploaded)):
        try:
            source = uploaded if uploaded is not None else Path(path.strip().strip('"'))
            st.session_state.ann_manual_frame = read_annotation_table(source, header=header, sheet_name=sheet or 0)
        except Exception as exc:
            st.error(f"读取注释表失败：{exc}")
    frame = st.session_state.get("ann_manual_frame")
    if frame is None:
        return
    orientations = ["每行一个特征", "每行一个注释字段"]
    index, help_text = _choice("annotation.manual_orientation", orientations)
    orientation = a.selectbox("表格方向", orientations, index=index, key="ann_manual_orientation", help=help_text)
    value, help_text = _parameter("annotation.manual_field_names", True)
    field_names = b.checkbox("第一列是字段名称", value, key="ann_manual_field_names", disabled=orientation == "每行一个特征", help=help_text)
    frame = orient_annotation_table(frame, transpose=orientation == "每行一个注释字段", field_names=field_names)
    columns = frame.columns.tolist()
    default_column = next((key for key in ["name", "annotation", "名称", "物质名称"] if key in columns), columns[-1])
    configured, _ = _parameter("annotation.manual_value", "")
    default_column = configured if configured in columns else default_column
    value, help_text = _parameter("annotation.manual_rows", "")
    row_selection = st.text_input("导入记录行号", value, key="ann_manual_rows", help=help_text)
    methods = ["自动识别", "按 var 顺序", "按键值匹配"]
    index, help_text = _choice("annotation.manual_mode", methods)
    method = b.selectbox("对齐方式", methods, index=index, key="ann_manual_mode", help=help_text)
    options = dict(mode={"自动识别": "auto", "按 var 顺序": "position", "按键值匹配": "key"}[method])
    if method == "按键值匹配":
        index, help_text = _choice("annotation.manual_source_key", columns)
        options["source_key"] = a.selectbox("导入表的匹配列", columns, index=index, key="ann_manual_source_key", help=help_text)
        keys = list(dict.fromkeys(["feature_id", *data.var.columns]))
        index, help_text = _choice("annotation.manual_var_key", keys)
        options["var_key"] = b.selectbox("var 匹配字段", keys, index=index, key="ann_manual_var_key", help=help_text)
    options["ppm"] = _number(st, "手动注释 m/z 匹配 ppm", "manual_ppm", 5., .01, 500.)
    value, help_text = _parameter("annotation.manual_target", "manual_annotation")
    mapping = _var_mapping_editor(st, columns, key="ann_manual_mapping", defaults={default_column: value})
    value, help_text = _parameter("annotation.manual_clear", False)
    clear = b.checkbox("未匹配特征的注释置空", value, key="ann_manual_clear", help=help_text)
    if not mapping:
        return
    try:
        positions = import_row_positions(row_selection, len(frame))
        values, matched, description = manual_annotation_columns(data, frame, mapping, row_positions=positions, **options)
    except (ValueError, KeyError) as exc:
        st.info(str(exc))
        return
    st.caption(f"{description} · 匹配 {int(matched.sum())} / {data.n_vars} 个特征。同一匹配键的多个名称以 / 合并。预览匹配结果前 100 条。")
    st.dataframe(values.loc[matched].head(100).rename_axis("feature_id"), use_container_width=True)
    if st.button("写入手动注释", key="ann_manual_write", disabled=not matched.any()):
        count = write_manual_annotation_columns(data, values, matched, clear_unmatched=clear)
        _invalidate_group_sets(st)
        st.success(f"已写入 {len(mapping)} 个 var 字段，匹配 {count} 个特征。")


def _spectrum_assistance(st, data, terminal):
    from scMM.analysis.annotation_tools import sum_cell_frames
    from scMM.file.dialogs import choose_local_path
    from scMM.plot.annotation import cell_spectrum_figure
    from scMM.webapp import _execute, _plotly_with_export, _parameter, _choice
    st.subheader("细胞帧叠加谱辅助注释")
    if not {"source_file", "frame_id"}.issubset(data.obs):
        st.info("叠加细胞帧需要 obs 中的 source_file 和原始 frame_id；重新处理原始文件可生成这两项信息。")
        return
    plot_area = st.container()
    sources = data.obs["source_file"].astype(str).unique().tolist()
    selected = st.multiselect("参与叠加的质谱源文件（可多选）", sources, key="ann_spectrum_sources",
                             help="可指定多个质谱文件，仅从所选文件对应的细胞帧中抽取并叠加。")
    index, help_text = _choice("annotation.cell_sample_mode", ["每个文件", "所选文件合计"])
    sample_mode = st.radio("抽样帧数口径", ["每个文件", "所选文件合计"], index=index, horizontal=True,
                           help=help_text, key="ann_cell_sample_mode")
    source_paths = {}
    for source in selected:
        key = f"ann_cell_path_{id(data)}_{source}"
        if key not in st.session_state:
            st.session_state[key] = str(data.uns.get("files", {}).get(source, {}).get("path", ""))
        def browse(source=source, key=key):
            try:
                path = choose_local_path(filetypes=[("MS files", "*.raw *.mzML *.mzXML"), ("All files", "*.*")])
                if path:
                    st.session_state[key] = path
            except Exception as exc:
                st.error(f"选择文件失败：{exc}")
        left, right = st.columns([5, 1])
        source_paths[source] = left.text_input(f"{source} 的原始文件路径", key=key).strip().strip('"')
        right.button("选择文件", key=f"ann_cell_browse_{source}", on_click=browse)
    a, b, c = st.columns(3)
    count = _number(a, "每个文件叠加细胞帧数" if sample_mode == "每个文件" else "叠加细胞帧总数", "cell_count", 20, 1, 100000)
    seed = _number(b, "抽样随机种子", "cell_seed", 42, 0, 2147483647)
    value, help_text = _parameter("annotation.cell_active", True)
    active = c.checkbox("仅使用有效细胞", value, key="ann_cell_active", help=help_text)
    finite_mz = data.var["mz"].to_numpy(float)
    mz_min = _number(a, "叠加 m/z 下限", "cell_mz_min", max(.01, float(np.nanmin(finite_mz)) - 2.), .01, 100000.)
    mz_max = _number(b, "叠加 m/z 上限", "cell_mz_max", float(np.nanmax(finite_mz)) + 5., .02, 100000.)
    resolution = _number(c, "200 m/z 处分辨率", "cell_resolution", 35000., 100., 1000000.)
    points = _number(a, "每 FWHM 重采样点数", "cell_points", 5., 2., 30.)
    eligible = data.obs["source_file"].astype(str).isin(selected).to_numpy()
    if active:
        eligible &= data.cell_keep_mask()
    settings = (id(data), tuple(source_paths.items()), count, seed, active, mz_min, mz_max, resolution, points,
                np.packbits(eligible).tobytes(), sample_mode)
    ready = bool(selected) and all(source_paths.values())
    if st.button("读取并叠加细胞帧", key="ann_cell_sum", disabled=not ready):
        def run(progress):
            if mz_max <= mz_min:
                raise ValueError("叠加 m/z 上限需大于下限。")
            return sum_cell_frames(data, source_paths, count=count, seed=seed, active_only=active,
                                   sample_mode="per_file" if sample_mode == "每个文件" else "total",
                                   mz_range=(mz_min, mz_max), resolution_200=resolution,
                                   points_per_fwhm=points, progress=progress)
        result = _execute(st, terminal, "辅助注释 · 细胞帧叠加", run, "annotation_cell_spectrum")
        if result is not None:
            st.session_state.ann_cell_spectrum = dict(settings=settings, **result)
    cache = st.session_state.get("ann_cell_spectrum", {})
    if cache.get("settings") != settings:
        plot_area.caption("选择下方源文件并读取细胞帧后显示叠加谱。参数更新后可重新读取。")
        return
    report = pd.DataFrame(cache["sources"])
    report["frame_ids"] = report["frame_ids"].map(lambda values: ", ".join(map(str, values)))
    st.dataframe(report.rename(columns={"source_file": "文件", "n_cell_frames": "实际帧数", "frame_ids": "原始帧序号（从 0 开始）"}),
                 hide_index=True, use_container_width=True)
    st.caption("点击上方注释表中的 Feature ID，或在下方选择特征，即可定位对应峰及同位素峰。切换特征复用已叠加的谱图。")
    feature = st.selectbox("谱图定位特征", data.var_names.tolist(), key="ann_spectrum_feature",
                           format_func=lambda key: f"{key} · m/z {data.var.at[key, 'mz']:.6f}")
    if not cache["mz"][0] <= float(data.var.at[feature, "mz"]) <= cache["mz"][-1]:
        st.info("当前特征位于叠加 m/z 范围之外，可调整范围后重新叠加。")
    half_window = _number(b, "局部放大半宽（Da）", "cell_zoom", 2., .01, 100.)
    figure = cell_spectrum_figure(data, cache, feature, half_window=half_window)
    _plotly_with_export(st, figure, "annotation_cell_spectrum", container=plot_area)


def render_annotation_page():
    import streamlit as st
    from scMM.webapp import _execute, _terminal, _parameter
    st.header("物质注释")
    data = st.session_state.data
    if data is None:
        st.info("请先导入并处理数据。")
        return
    if "annotation_isotope" not in data.var:
        data.var["annotation_isotope"] = np.zeros(data.n_vars, dtype=np.uint8)
    tabs = st.tabs(["1 · 同位素识别", "2 · MS1 数据库", "3 · MS/MS 匹配", "4 · 注释结果", "5 · 手动注释"])
    with tabs[0]:
        st.caption("调用 CyESIData.deisotope；使用 raw 层和有效细胞，保留强度矩阵与全部特征。缺失值按零值处理。")
        a, b, c = st.columns(3)
        isotope_params = dict(
            isotope_diff=_number(a, "同位素间隔（Da）", "isotope_diff", 1.003355, .1, 3., step=.000001, format="%.8f"),
            ppm_tol=_number(b, "同位素 ppm 容差", "isotope_ppm", 5., .01, 100.),
            max_isotope_order=_number(c, "最高同位素阶数", "isotope_order", 3, 1, 10),
            r_square_threshold=_number(a, "过原点回归 R² 下限", "isotope_r2", .95, 0., 1., step=.0001, format="%.6f"),
            carbon13_abundance=_number(b, "13C 丰度", "carbon13", .0109, .0001, .2, step=.000001, format="%.8f"),
            intensity_threshold=_number(c, "强度下限", "isotope_intensity", 0., 0., 1.e12),
            safety_factor=_number(a, "同位素强度上限安全系数", "isotope_safety", 1., .01, 10.),
        )
        if st.button("识别同位素", key="ann_deisotope"):
            with st.spinner("识别同位素…"):
                try:
                    pairs = deisotope_features(data, **isotope_params)
                    st.success(f"识别到 {int(data.var['annotation_isotope'].sum())} 个同位素特征。")
                    if "annotation" in data.uns:
                        st.info("同位素归属已更新；点击“开始注释 / 刷新”同步候选。")
                except Exception as exc:
                    st.error(str(exc))
        restored = st.multiselect("手动恢复同位素特征", data.var_names[data.var["annotation_isotope"].astype(bool)].tolist(), key="ann_restore")
        if st.button("恢复所选特征", disabled=not restored):
            data.var.loc[restored, "annotation_isotope"] = np.uint8(0)
            sync_isotope_flags(data)
            if "feature_keep_annotation_isotope" in data.var:
                data.var.loc[restored, "feature_keep_annotation_isotope"] = np.uint8(1)
            st.success("已恢复，点击刷新更新匹配。")
    with tabs[1]:
        st.caption("可一次选择或上传多个 CSV/TSV/SDF/JSON 数据库，合并参与匹配。多个本机路径每行一个。")
        db_path, db_upload = _path_input(st, "MS1 数据库", "database", multiple=True, types=["csv", "tsv", "sdf", "json"])
        default, help_text = _parameter("annotation.mode", "pos")
        mode = st.selectbox("采集模式", ["pos", "neg", "both"], index=["pos", "neg", "both"].index(default), help=help_text, key="ann_mode")
        available = list(DEFAULT_ADDUCTS_POS if mode == "pos" else DEFAULT_ADDUCTS_NEG if mode == "neg" else ADDUCTS)
        adducts = st.multiselect("加合模式", available, default=available, key=f"ann_adducts_{mode}", help="按离子精确质量计算；[M]+ 包含失电子质量修正。")
        ppm = _number(st, "MS1 匹配 ppm 容差", "ms1_ppm", 5., .01, 500.)
    with tabs[2]:
        st.caption("MS2 前体按 ppm 匹配；Thermo RAW 优先用官方质心谱汇总。其它 Profile 谱叠加后去基线与质心化；Profile 参数适用于此路径。MS1 与 MS3+ 不进入 MS/MS 匹配。")
        msms_path, msms_upload = _path_input(st, "实测 MS/MS 文件", "measured", multiple=True, types=["raw", "mzml", "mzxml"])
        library_path, library_upload = _path_input(st, "标准 MS/MS 谱库", "library", multiple=True, types=["msp", "mgf", "json", "csv", "tsv"])
        st.caption("可一次选择或上传多个标准谱库，与 MS1 数据库共同参与本次注释。")
        a, b, c = st.columns(3)
        precursor_ppm = _number(a, "前体匹配 ppm", "precursor_ppm", 10., .01, 500.)
        fragment_ppm = _number(b, "碎片匹配 / 合并 ppm", "fragment_ppm", 20., .01, 500.)
        min_score = _number(c, "MS/MS 相似度下限", "min_msms_score", .5, 0., 1.)
        min_matches = _number(a, "最少匹配碎片数", "min_matches", 3, 1, 100)
        step = _number(b, "Profile 重采样间隔（Da）", "profile_step", .005, .0001, .1, step=.001)
        baseline = _number(c, "Top-hat 窗口（Da）", "baseline_width", 1., 0., 50.)
        snr = _number(a, "质心化信噪比", "centroid_snr", 0., 0., 100.)
        relative = _number(b, "碎片相对强度下限", "min_relative_intensity", .01, 0., 1.)
        mz_min = _number(a, "碎片 m/z 下限", "fragment_mz_min", 30., 1., 3000.)
        mz_max = _number(b, "碎片 m/z 上限", "fragment_mz_max", 2000., 2., 10000.)
        iw = _number(a, "同位素评分权重", "isotope_weight", .5, 0., 1.)
        mw = _number(b, "MS/MS 评分权重", "msms_weight", .5, 0., 1.)
        st.caption("综合分数为可用证据的加权平均；缺失证据记为不可用。碎片 dot product 使用平方根强度归一化余弦，未匹配碎片计入分母。")
    terminal = _terminal(st)
    if st.button("开始注释 / 刷新", type="primary", key="ann_run"):
        def run(progress):
            if mz_max <= mz_min:
                raise ValueError("碎片 m/z 上限需大于下限。")
            if not adducts:
                raise ValueError("请选择至少一个加合模式。")
            with tempfile.TemporaryDirectory(prefix="scmm_annotation_") as directory:
                databases = _materialize(db_path, db_upload, directory, "db")
                libraries = _materialize(library_path, library_upload, directory, "lib")
                files = _materialize(msms_path, msms_upload, directory, "measured")
                frames, references = [], {}
                for i, path in enumerate(databases + libraries):
                    progress(.05 + .1 * i / max(1, len(databases + libraries)), f"读取数据库 {i+1}/{len(databases + libraries)}")
                    frame, spectra = load_annotation_database(path)
                    ids = {k: f"db{i}_{k}" for k in spectra}
                    frame["library_id"] = frame["library_id"].map(ids)
                    references.update({ids[k]: v for k, v in spectra.items()})
                    progress(.05 + .1 * (i+1) / max(1, len(databases + libraries)),
                             f"谱库 {path.name}：{len(frame)} 条记录，{sum(bool(len(v)) for v in spectra.values())} 条有效 MS/MS 谱")
                    frames.append(frame)
                if not frames:
                    raise ValueError("请提供 MS1 数据库或 MS/MS 谱库。")
                targets = data.var.loc[~data.var["annotation_isotope"].astype(bool), "mz"].to_dict()
                measured = collect_msms(files, targets, precursor_ppm=precursor_ppm, fragment_ppm=fragment_ppm,
                                        profile_step=step, baseline_width=baseline, snr=snr, min_relative_intensity=relative,
                                        mz_min=mz_min, mz_max=mz_max,
                                        progress=lambda f, m: progress(.15 + .5*f, m)) if files else {}
                table = annotate_features(data, pd.concat(frames, ignore_index=True), references, adducts=adducts, mode=mode,
                                          ppm=ppm, measured=measured, fragment_ppm=fragment_ppm, min_msms_score=min_score,
                                          min_matches=min_matches, isotope_weight=iw, msms_weight=mw,
                                          progress=lambda f, m: progress(.65 + .35*f, m))
                data.uns["annotation"]["parameters"].update(
                    precursor_ppm=precursor_ppm, profile_step=step, baseline_width=baseline, centroid_snr=snr,
                    min_relative_intensity=relative, fragment_mz_min=mz_min, fragment_mz_max=mz_max,
                    database_files=[p.name for p in databases + libraries], measured_files=[p.name for p in files])
                return table
        result = _execute(st, terminal, "物质注释", run, "annotation")
        if result is not None:
            st.success(f"已生成 {len(result)} 行注释结果。")
    with tabs[3]:
        _results(st, data, terminal)
    with tabs[4]:
        _manual_annotations(st, data)
