from __future__ import annotations

import io
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


_BUILTIN_UI_CONFIG = Path(__file__).with_name("config") / "defaults.json"


@lru_cache(maxsize=1)
def _ui_config():
    """Load UI defaults/help, optionally from ``SCMM_UI_CONFIG``."""
    configured = os.environ.get("SCMM_UI_CONFIG")
    path = Path(configured).expanduser() if configured else _BUILTIN_UI_CONFIG
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if configured:
            raise RuntimeError(f"无法读取 UI 配置 {path}: {exc}") from exc
        return {"parameters": {}}


def _parameter(parameter_id, fallback, fallback_help=""):
    entry = _ui_config().get("parameters", {}).get(parameter_id, {})
    return entry.get("default", fallback), entry.get("help", fallback_help)


def _choice(parameter_id, options, fallback_index=0):
    options = list(options)
    fallback = options[fallback_index]
    value, help_text = _parameter(parameter_id, fallback)
    index = options.index(value) if value in options else fallback_index
    return index, help_text


def _bounded_default(parameter_id, fallback, lower, upper):
    value, help_text = _parameter(parameter_id, fallback)
    return min(upper, max(lower, value)), help_text


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _substep_from_message(message):
    """Extract a stable substep label and its local progress from a report."""
    text = str(message).strip()
    match = re.search(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)", text)
    local_progress = None
    if match:
        current = int(match.group("current"))
        total = max(1, int(match.group("total")))
        local_progress = min(100, max(0, round(current / total * 100)))
        label = (text[: match.start()] + text[match.end() :]).strip(" :-|")
        label = re.sub(
            r"\b(files?|frames?|blocks?)\b",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip(" :-|")
    else:
        label = text.split(":", 1)[0].strip() if ":" in text else text
    return label or "Processing", local_progress


def _state(st):
    input_default, _ = _parameter("data.input_path", "")
    preview_default, _ = _parameter("preview.file", "")
    defaults = {
        "data": None,
        "logs": [f"[{_now()}] SCMM web workbench ready"],
        "tasks": [],
        "result_table": pd.DataFrame(),
        "supervised": None,
        "network": None,
        "differential_result": pd.DataFrame(),
        "differential_context": None,
        "preview_chrom": None,
        "preview_spectrum": None,
        "preview_cells": None,
        "input_path": input_default,
        "preview_path": preview_default,
        "completed_steps": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


class Terminal:
    """Persistent task terminal with one progress record per operation."""

    def __init__(self, st, progress_slot, report_slot, output_slot):
        self.st = st
        self.progress_slot = progress_slot
        self.report_slot = report_slot
        self.output_slot = output_slot
        self.current = None
        self.render()

    def render(self):
        tasks = self.st.session_state.tasks
        running = next(
            (task for task in reversed(tasks) if task["status"] == "running"),
            None,
        )
        if running is None:
            self.progress_slot.empty()
            self.report_slot.empty()
        else:
            self.progress_slot.progress(
                int(running["progress"]),
                text=(
                    f"⏳ {running['step']} · {running['progress']}% · "
                    f"{running['message']}"
                ),
            )
            active = next(
                (
                    step
                    for step in reversed(running.get("substeps", []))
                    if step["status"] == "running"
                ),
                None,
            )
            if active is None:
                self.report_slot.empty()
            else:
                self.report_slot.progress(
                    int(active["progress"]),
                    text=f"当前步骤：{active['label']} · {active['progress']}%",
                )
        self.output_slot.code(
            "\n".join(self.st.session_state.logs[-80:]),
            language=None,
            height=320,
        )

    def start(self, title):
        task = {
            "step": title,
            "status": "running",
            "progress": 0,
            "message": "Starting",
            "started": _now(),
            "substeps": [],
        }
        self.st.session_state.tasks.append(task)
        self.current = len(self.st.session_state.tasks) - 1
        self.write(f"START | {title}")
        self.progress_slot.progress(0, text=title)

    def update(self, fraction, message):
        percent = min(100, max(0, int(float(fraction) * 100)))
        if self.current is not None:
            task = self.st.session_state.tasks[self.current]
            task["progress"] = percent
            task["message"] = str(message)
            label, local_progress = _substep_from_message(message)
            substeps = task.setdefault("substeps", [])
            if not substeps or substeps[-1]["label"] != label:
                if substeps and substeps[-1]["status"] == "running":
                    substeps[-1]["progress"] = 100
                    substeps[-1]["status"] = "completed"
                substeps.append(
                    {
                        "label": label,
                        "progress": local_progress if local_progress is not None else 5,
                        "status": "running",
                    }
                )
            else:
                substeps[-1]["progress"] = (
                    local_progress
                    if local_progress is not None
                    else max(substeps[-1]["progress"], percent)
                )
        self.progress_slot.progress(percent, text=str(message))
        self.write(f"{percent:3d}% | {message}")

    def write(self, message):
        self.st.session_state.logs.append(f"[{_now()}] {message}")
        self.render()

    def finish(self, message):
        if self.current is not None:
            task = self.st.session_state.tasks[self.current]
            task["progress"] = 100
            task["message"] = str(message)
            task["status"] = "completed"
            for substep in task.get("substeps", []):
                substep["progress"] = 100
                substep["status"] = "completed"
        self.progress_slot.progress(100, text=str(message))
        self.write(f"100% | {message}")
        self.render()

    def fail(self, exc):
        if self.current is not None:
            task = self.st.session_state.tasks[self.current]
            task["status"] = "failed"
            task["message"] = str(exc)
            if task.get("substeps"):
                task["substeps"][-1]["status"] = "failed"
        self.write(f"ERROR | {type(exc).__name__}: {exc}")
        self.write(traceback.format_exc().strip())
        self.progress_slot.error(str(exc))


def _terminal(st):
    st.markdown("#### 实时任务终端")
    progress_slot = st.empty()
    report_slot = st.empty()
    output_slot = st.empty()
    return Terminal(st, progress_slot, report_slot, output_slot)


def _execute(st, terminal, title, function, completed_step=None):
    terminal.start(title)
    try:
        result = function(terminal.update)
    except Exception as exc:
        terminal.fail(exc)
        st.error(f"{title}失败：{exc}")
        return None
    terminal.finish(f"{title}完成")
    if completed_step:
        st.session_state.completed_steps.add(completed_step)
    return result


def _pipeline_status(st):
    steps = [
        ("data", "数据"),
        ("preprocess", "预处理"),
        ("embedding", "降维"),
        ("cluster", "聚类"),
        ("trajectory", "轨迹"),
        ("analysis", "统计/模型"),
    ]
    columns = st.columns(len(steps))
    done = st.session_state.completed_steps
    for column, (key, label) in zip(columns, steps):
        column.markdown(f"{'✅' if key in done else '○'} **{label}**")


def _engine(st):
    from scMM.plot.engine import PlotEngine

    return PlotEngine.from_adata(st.session_state.data, copy=False)


def _layer_selector(st, data):
    data.ensure_matrix_layers()
    active = str(data.uns["active_layer"])
    dataset = id(data)
    if st.session_state.get("layer_selector_seen") != (dataset, active):
        st.session_state.analysis_layer = active
    _, help_text = _parameter("data.analysis_layer", "raw", "选择后续计算、特征强度上色和结果矩阵使用的数据层。")
    selected = st.sidebar.selectbox("分析数据层（X）", list(data.layers.keys()),
                                    index=list(data.layers).index(active), key="analysis_layer", help=help_text)
    if selected != active:
        data.set_active_layer(selected)
    st.session_state.layer_selector_seen = (dataset, selected)
    st.sidebar.caption(f"当前 X = layers[{selected!r}]。预处理会生成新层并自动选中，可随时切回。")


def _browse(st, kind, state_key):
    from scMM.file.dialogs import choose_local_path

    try:
        selected = choose_local_path(kind)
    except Exception as exc:
        st.error(f"无法打开本机选择器：{exc}")
        return
    if selected:
        st.session_state[state_key] = selected


def _h5ad_bytes(data):
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
        path = Path(handle.name)
    try:
        data.save(path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _csv_zip(data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("X.csv", data.to_df().to_csv())
        archive.writestr("obs.csv", data.obs.to_csv())
        archive.writestr("var.csv", data.var.to_csv())
    return buffer.getvalue()


def _sticky_matrix_table(st, data, obs_key, var_key):
    import streamlit.components.v1 as components
    component = components.declare_component(
        "scmm_matrix_table",
        path=str(Path(__file__).with_name("components") / "matrix_table"),
    )
    identity = f"{id(data)}:{data.n_obs}:{data.n_vars}:{obs_key}:{var_key}"
    state_key = "matrix_viewport"
    if st.session_state.get("matrix_dataset") != identity:
        st.session_state.matrix_dataset = identity
        st.session_state[state_key] = {"row": 0, "col": 0}
    viewport = st.session_state[state_key]
    row = min(max(0, int(viewport.get("row", 0))), max(0, data.n_obs - 1))
    col = min(max(0, int(viewport.get("col", 0))), max(0, data.n_vars - 1))
    rows = slice(row, row + 100)
    columns = slice(col, col + 30)
    matrix = _matrix_frame(data, rows, columns)
    values = matrix.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, None).tolist()

    def metadata_values(series):
        return series.astype(object).where(series.notna(), None).tolist()

    request = component(
        dataset=identity,
        nrows=data.n_obs,
        ncols=data.n_vars,
        row=row,
        col=col,
        values=values,
        observations=metadata_values(data.obs[obs_key].iloc[rows]) if obs_key else data.obs_names[rows].tolist(),
        annotations=metadata_values(data.var[var_key].iloc[columns]) if var_key else [""] * matrix.shape[1],
        cell_ids=data.obs_names[rows].tolist(),
        features=data.var_names[columns].tolist(),
        obs_label=f"obs · {obs_key}" if obs_key else "cell ID",
        var_label=f"var · {var_key}" if var_key else "features",
        key=f"matrix_table_{identity}",
        default={"row": 0, "col": 0},
    )
    if request and request != viewport:
        st.session_state[state_key] = request
        st.rerun()


def _matrix_frame(data, rows, columns):
    return pd.DataFrame(
        _dense(data.X[rows, columns]),
        index=data.obs_names[rows],
        columns=data.var_names[columns],
    )


def _matplotlib_plotly_bytes(figure, **params):
    from scMM.plot.export import render_figure

    return render_figure(figure, **params)


def _image_filename(key, file_format, *, figure=None, title=""):
    labels = {
        "preview_chromatogram": "TIC_EIC", "preview_ms1_spectrum": "MS1质谱图",
        "preview_cell_extraction": "细胞提取预览", "processed_summed_spectrum": "叠加质谱图",
        "cell_gate": "细胞圈门图", "qc_histogram": "质控分布图",
        "differential_volcano": "差异分析火山图", "differential_distribution": "差异特征分布图",
        "targeted_distribution": "靶向特征组间比较", "differential_heatmap": "差异特征热图",
        "correlation_heatmap": "特征相关性热图", "file_qc_heatmap": "文件质控热图",
        "feature_network": "特征相关性网络", "numeric_association": "相关性散点图",
        "supervised_roc": "ROC", "supervised_latent": "监督学习得分图",
        "precision_recall": "PR曲线", "calibration": "校正曲线", "confusion_matrix": "混淆矩阵",
        "msms_mirror": "MSMS镜像谱", "annotation_coverage": "注释覆盖率", "annotation_categories": "注释物质类别",
        "annotation_cell_spectrum": "辅助注释细胞帧叠加谱",
    }
    label = labels.get(key, key)
    if key.startswith("embedding_"):
        label = key.removeprefix("embedding_").removeprefix("X_").upper() + "降维图"
    parts = [label]
    if figure is not None:
        title = title or figure.layout.title.text or ""
        metadata = figure.layout.meta
        if isinstance(metadata, dict) and metadata.get("scmm_color_label"):
            parts.append(str(metadata["scmm_color_label"]))
        facets = [a.text for a in figure.layout.annotations or [] if re.match(r"(?:feature(?:_label)?|mz)\s*=", a.text or "", re.I)]
        parts.extend(facets[:3])
        if key == "preview_chromatogram":
            parts.extend(list(dict.fromkeys(trace.name for trace in figure.data if trace.name))[:4])
        if key in {"cell_gate", "numeric_association"}:
            parts.extend([figure.layout.xaxis.title.text or "X", figure.layout.yaxis.title.text or "Y"])
    if title and title != label:
        parts.append(title)
    name = html.unescape(re.sub(r"<[^>]*>", "", "_".join(parts)))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")[:140].rstrip(" ._")
    return f"SCMM_{name}.{file_format}"


def _save_image(st, payload, default_filename):
    from scMM.file.dialogs import choose_save_path

    try:
        selected = choose_save_path(default_filename, st.session_state.get("image_save_directory"))
        if not selected:
            return
        destination = Path(selected)
        handle = tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False)
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(payload)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        st.session_state.image_save_directory = str(destination.parent)
        st.success(f"图片已保存至 {destination}")
    except Exception as exc:
        st.error(f"图片保存失败：{exc}")


def _plotly_with_export(st, figure, key, *, container=None, height=None, render=True):
    from scMM.plot.interaction import constrain_zoom
    constrain_zoom(figure)
    target = container or st
    if render:
        target.plotly_chart(figure, use_container_width=True, key=f"{key}_plotly", height=height or "content")
    if not target.button("Matplotlib 重绘与导出", key=f"{key}_export_open"):
        return

    @st.dialog("Matplotlib 重绘参数与预览", width="large")
    def export_dialog():
        controls, preview = st.columns([1, 1.2])
        with controls:
            a, b = st.columns(2)
            file_format = a.selectbox("格式", ["svg", "png"], key=f"{key}_format")
            dpi = b.number_input("DPI", 72, 1200, 300, key=f"{key}_dpi")
            width = a.number_input("宽度（inch）", 2.0, 30.0, 8.0, .5, key=f"{key}_width")
            height_value = b.number_input("高度（inch）", 2.0, 30.0, 6.0, .5, key=f"{key}_height")
            font_family = a.selectbox("字体", ["Arial", "Times New Roman", "SimHei", "Microsoft YaHei", "sans-serif"], key=f"{key}_font")
            font_size = b.number_input("刻度字号", 6.0, 40.0, 10.0, 1.0, key=f"{key}_fontsize")
            title = st.text_input("标题", value=figure.layout.title.text or "", key=f"{key}_title")
            xlabel = st.text_input("X 轴标题", value=figure.layout.xaxis.title.text or "", key=f"{key}_xlabel")
            ylabel = st.text_input("Y 轴标题", value=figure.layout.yaxis.title.text or "", key=f"{key}_ylabel")
            with st.expander("点、线、分布与坐标细节", expanded=True):
                a, b = st.columns(2)
                marker_size = a.number_input("点面积（pt²）", 0.1, 1000.0, 20.0, key=f"{key}_markersize")
                line_width = b.number_input("线宽（pt）", 0.1, 20.0, 1.5, .1, key=f"{key}_linewidth")
                marker = a.selectbox("点形状", ["o", ".", "s", "^", "D", "x", "+"], key=f"{key}_marker")
                alpha_default = .75 if isinstance(figure.layout.meta, dict) and figure.layout.meta.get("scmm_plot") == "volcano" else .8
                alpha = b.number_input("透明度", .05, 1.0, alpha_default, .05, key=f"{key}_alpha")
                cmap_options = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "RdBu", "RdBu_r", "Spectral", "coolwarm"]
                original_cmap = (figure.layout.meta or {}).get("scmm_cmap", "viridis") if isinstance(figure.layout.meta, dict) else "viridis"
                cmap_index = next((i for i, name in enumerate(cmap_options) if name.lower() == str(original_cmap).lower()), 0)
                cmap = a.selectbox("cmap", cmap_options, index=cmap_index, key=f"{key}_cmap")
                jitter = b.number_input("散点水平抖动", 0.0, .5, .08, .01, key=f"{key}_jitter")
                violin_width = a.number_input("小提琴/箱体宽度", .1, 1.5, .75, .05, key=f"{key}_violin_width")
                bins = b.number_input("直方图箱数", 5, 500, 40, key=f"{key}_bins")
                label_size = a.number_input("轴标题与标注字号", 6, 40, 10, key=f"{key}_label_size")
                annotation_size = None
                if isinstance(figure.layout.meta, dict) and figure.layout.meta.get("scmm_plot") == "volcano":
                    annotation_size = b.number_input("火山图标注字号（pt）", 6., 48.,
                                                    float(figure.layout.meta.get("scmm_volcano_annotation_size", 10.)),
                                                    key=f"{key}_annotation_size")
                title_size = b.number_input("标题字号", 6, 48, 12, key=f"{key}_title_size")
                legend_size = a.number_input("图例字号", 6, 40, 9, key=f"{key}_legend_size")
                legend_location = b.selectbox("图例位置", ["best", "upper right", "upper left", "lower right", "lower left"], key=f"{key}_legend_location")
                xscale = a.selectbox("X 轴尺度", ["linear", "log", "symlog"], key=f"{key}_xscale")
                yscale = b.selectbox("Y 轴尺度", ["linear", "log", "symlog"], key=f"{key}_yscale")
                xlim = ylim = None
                if a.checkbox("指定 X 轴范围", key=f"{key}_xlim_on"):
                    xlim = (a.number_input("X 最小值", value=0.0, key=f"{key}_xmin"), a.number_input("X 最大值", value=1.0, key=f"{key}_xmax"))
                if b.checkbox("指定 Y 轴范围", key=f"{key}_ylim_on"):
                    ylim = (b.number_input("Y 最小值", value=0.0, key=f"{key}_ymin"), b.number_input("Y 最大值", value=1.0, key=f"{key}_ymax"))
                grid = a.checkbox("网格", value=False, key=f"{key}_grid")
                legend = b.checkbox("图例", value=True, key=f"{key}_legend")
            params = dict(width=width, height=height_value, dpi=dpi, font_family=font_family,
                          font_size=font_size, title=title, xlabel=xlabel, ylabel=ylabel, cmap=cmap,
                          marker_size=marker_size, alpha=alpha, line_width=line_width, grid=grid,
                          legend=legend, marker=marker, jitter=jitter, violin_width=violin_width,
                          bins=bins, label_size=label_size, title_size=title_size, legend_size=legend_size,
                          legend_location=legend_location, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim,
                          annotation_size=annotation_size)
            signature = repr(params) + file_format + figure.to_json()
            cache_key = f"{key}_matplotlib_preview"
            if st.button("预览", key=f"{key}_preview", type="primary"):
                try:
                    png = _matplotlib_plotly_bytes(figure, file_format="png", **params)
                    payload = png if file_format == "png" else _matplotlib_plotly_bytes(figure, file_format="svg", **params)
                    st.session_state[cache_key] = dict(signature=signature, png=png, payload=payload)
                except Exception as exc:
                    st.error(f"重绘失败：{exc}")
            cached = st.session_state.get(cache_key, {})
            current = cached.get("signature") == signature
            if st.button("下载", disabled=not current, key=f"{key}_download"):
                filename = _image_filename(key, file_format, figure=figure, title=title)
                _save_image(st, cached["payload"], filename)
            st.caption("调整参数后点击预览，再下载对应图像。SVG 保留可编辑文字。")
        with preview:
            st.markdown("#### Matplotlib 预览")
            if current:
                st.image(cached["png"], width="stretch")
            else:
                st.info("点击左侧“预览”查看当前参数的绘图结果。")

    export_dialog()


def _plot_embedding(
    st,
    data,
    key,
    color_mode,
    color_key,
    feature,
    cmap,
    categorical_palette,
    color_scale="linear",
    linthresh=1.0,
):
    import plotly.express as px
    from scMM.plot.diagnostics import color_coordinates

    values = np.asarray(data.obsm[key])
    if values.shape[1] < 2:
        st.warning("该嵌入只有一个维度，无法绘制二维图。")
        return
    frame = pd.DataFrame(
        {"dimension 1": values[:, 0], "dimension 2": values[:, 1]},
        index=data.obs_names,
    )
    color = None
    if color_mode == "obs":
        color = color_key
        frame[color] = data.obs[color_key].to_numpy()
    elif color_mode == "feature":
        color = f"intensity: {feature}"
        frame[color] = _dense(data[:, feature].X).ravel()
    continuous = color is not None and pd.api.types.is_numeric_dtype(frame[color])
    valid = np.ones(len(frame), dtype=bool)
    plot_color = color
    if continuous and color_scale != "linear":
        frame["__scaled_color"] = color_coordinates(frame[color].to_numpy(dtype=float), color_scale, linthresh)
        valid = np.isfinite(frame["__scaled_color"].to_numpy())
        plot_color = "__scaled_color"
    figure = px.scatter(
        frame.loc[valid],
        x="dimension 1",
        y="dimension 2",
        color=plot_color,
        hover_name=frame.index[valid],
        hover_data={color: True, "__scaled_color": False} if continuous and color_scale != "linear" else None,
        render_mode="webgl",
        color_continuous_scale=cmap,
        color_discrete_sequence=getattr(
            px.colors.qualitative,
            categorical_palette,
        ),
    )
    if continuous:
        raw_values = frame.loc[valid, color].to_numpy(dtype=float)
        for trace in figure.data:
            trace.meta = {"scmm_color_values": raw_values.tolist()}
        if len(raw_values) and color_scale != "linear":
            raw_ticks = np.geomspace(raw_values.min(), raw_values.max(), 5) if color_scale == "log10" else np.linspace(raw_values.min(), raw_values.max(), 7)
            raw_ticks = np.unique(raw_ticks)
            figure.update_coloraxes(colorbar=dict(title=f"{color} ({color_scale})",
                                                  tickvals=color_coordinates(raw_ticks, color_scale, linthresh),
                                                  ticktext=[f"{value:.4g}" for value in raw_ticks]))
        if not valid.all():
            gray = px.scatter(frame.loc[~valid], x="dimension 1", y="dimension 2", hover_name=frame.index[~valid], hover_data=[color])
            for trace in gray.data:
                trace.marker.color = "#b8b8b8"
                trace.name = "非正值 / 缺失值" if color_scale == "log10" else "缺失值"
                trace.showlegend = True
                figure.add_trace(trace)
            st.caption("log10 上色仅对正值有效；非正值和缺失值以灰色显示。" if color_scale == "log10" else "缺失值以灰色显示。")
    figure.update_layout(meta={"scmm_color_scale": color_scale if continuous else "linear", "scmm_linthresh": linthresh,
                               "scmm_color_label": color, "scmm_cmap": cmap})
    figure.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
    _plotly_with_export(st, figure, f"embedding_{_safe_filter_suffix(key, 'embedding')}")


def page_data():
    import streamlit as st

    from scMM.file.data import CyESIData
    from scMM.workflows import ProcessingConfig, parse_reference_mz, run_processing

    st.header("数据导入与 CyESI 峰提取")
    st.caption("选择一种明确的输入模式；本机选择按钮打开运行 SCMM 的电脑上的原生对话框。")
    _pipeline_status(st)
    terminal = _terminal(st)
    if st.session_state.data is not None:
        _file_label_tool(st, st.session_state.data)

    modes = {
        "加载已有 H5AD": "h5ad",
        "处理单个 mzML / mzXML / RAW 文件": "single",
        "批量目录：统一质量轴（先累加所有文件，再用共同 m/z 特征提取）": "shared",
        "批量目录：独立质量轴（每个文件单独寻峰，再按 ppm 合并并保留特有峰）": "independent",
    }
    mode_index, mode_help = _choice("data.input_mode", list(modes))
    label = st.radio("输入模式", list(modes), index=mode_index, help=mode_help)
    mode = modes[label]
    choose_kind = "directory" if mode in {"shared", "independent"} else "file"
    if st.button("📁 选择文件夹" if choose_kind == "directory" else "📄 选择文件"):
        _browse(st, choose_kind, "input_path")
    input_path = st.text_input("已选择路径", key="input_path", help=_parameter("data.input_path", "")[1])

    if mode == "h5ad":
        uploaded = st.file_uploader("也可以从浏览器上传 H5AD", type=["h5ad"], help=_parameter("data.upload_h5ad", None)[1])

        def load_h5ad(progress):
            progress(0.15, "Reading H5AD")
            if uploaded is not None:
                with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as handle:
                    handle.write(uploaded.getbuffer())
                    temporary = Path(handle.name)
                try:
                    result = CyESIData.read(temporary)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                if not input_path:
                    raise ValueError("请选择 H5AD 文件")
                result = CyESIData.read(input_path)
            progress(0.9, f"Loaded {result.n_obs} cells × {result.n_vars} features")
            return result

        if st.button("加载 H5AD", type="primary", use_container_width=True):
            result = _execute(st, terminal, "加载 H5AD", load_h5ad, "data")
            if result is not None:
                st.session_state.data = result
                st.rerun()
        return

    st.subheader("Reference 与输出")
    c1, c2 = st.columns(2)
    reference_default, reference_help = _parameter("data.reference_mz", "734.5929")
    reference_text = c1.text_input("Reference m/z（逗号分隔）", value=str(reference_default), key="extraction_reference", help=reference_help)
    reference_index, reference_mode_help = _choice("data.reference_mode", ["union", "intersection"])
    reference_mode = c2.radio("多个 reference 的细胞判定", ["union", "intersection"], index=reference_index, horizontal=True, help=reference_mode_help)
    output_default, output_help = _parameter("data.output_path", "results/scmm_result.h5ad")
    output_default = Path(output_default)
    if not output_default.is_absolute(): output_default = Path.cwd() / output_default
    output_path = st.text_input("H5AD 输出路径", value=str(output_default), help=output_help)
    c1, c2 = st.columns(2)
    metadata_default, metadata_help = _parameter("data.metadata_path", "")
    metadata = c1.text_input("样本元数据 CSV（可选）", value=str(metadata_default), help=metadata_help)
    msconvert_default, msconvert_help = _parameter("data.msconvert", "")
    msconvert = c2.text_input("msconvert.exe（兼容旧配置）", value=str(msconvert_default), help=msconvert_help)
    frame_default, frame_help = _parameter("data.frame_metadata", ["rt"])
    frame_metadata = st.multiselect("写入 obs 的帧元数据", ["rt", "ms_level", "native_id"], default=frame_default, help=frame_help)
    st.caption("原始 frame_id 会强制写入 obs，不受此选项影响；MS2 及更高等级谱不会进入细胞提取。")

    st.subheader("质量轴与峰参数")
    a, b = st.columns(2)
    value, help_text = _parameter("data.mz_min", 100.0); processing_mz_min = a.number_input("处理 m/z 下限", 0.000001, 100000.0, float(value), help=help_text)
    value, help_text = _parameter("data.mz_max", 1000.0); processing_mz_max = b.number_input("处理 m/z 上限", 0.000001, 100000.0, float(value), help=help_text)
    a, b, c = st.columns(3)
    value, help_text = _parameter("data.ppm", 10.0); ppm = a.number_input("特征匹配 ppm", 0.01, 1000.0, float(value), help=help_text)
    value, help_text = _parameter("data.merge_ppm", 10.0); merge_ppm = b.number_input("文件间合并 ppm", 0.01, 1000.0, float(value), disabled=mode != "independent", help=help_text)
    value, help_text = _parameter("data.resolution", 35000.0); resolution = c.number_input("分辨率 @ m/z 200", 100.0, 2_000_000.0, float(value), help=help_text)
    value, help_text = _parameter("data.resample_points", 5.0); points = a.number_input("每 FWHM 重采样点", 1.0, 50.0, float(value), help=help_text)
    value, help_text = _parameter("data.spectrum_snr", 10.0); spectrum_snr = b.number_input("总谱峰 SNR", 0.0, 1000.0, float(value), help=help_text)
    value, help_text = _parameter("data.prominence", 0.0); prominence = c.number_input("峰 prominence ratio（0=自动）", 0.0, 1.0, float(value), help=help_text)
    value, help_text = _parameter("data.peak_distance", 3); distance = a.number_input("峰最小点距", 1, 1000, int(value), help=help_text)
    value, help_text = _parameter("data.block_size", 256); block = b.number_input("Feature block size", 1, 100000, int(value), help=help_text)
    value, help_text = _parameter("data.create_qc", True); create_qc = c.checkbox("生成 QC PDF", value=bool(value), help=help_text)

    st.subheader("细胞与特征判定")
    a, b, c = st.columns(3)
    value, help_text = _parameter("data.cell_threshold", 5.0); cell_threshold = a.number_input("Reference/基线阈值", 0.0, 100000.0, float(value), help=help_text)
    value, help_text = _parameter("data.feature_snr", 3.0); feature_snr = b.number_input("Feature SNR 阈值", 0.0, 1000.0, float(value), help=help_text)
    value, help_text = _parameter("data.max_zero", 0.9); max_zero = c.number_input("特征最大零比例", 0.0, 1.0, float(value), help=help_text)
    value, help_text = _parameter("data.baseline_window", 51); baseline = a.number_input("基线滑窗", 3, 10001, int(value), step=2, help=help_text)
    value, help_text = _parameter("data.noise_window", 51); noise = b.number_input("噪声滑窗", 3, 10001, int(value), step=2, help=help_text)

    def process(progress):
        if not input_path:
            raise ValueError("请选择输入文件或目录")
        config = ProcessingConfig(
            input_path=input_path,
            output_path=output_path,
            reference_mz=parse_reference_mz(reference_text),
            reference_mode=reference_mode,
            feature_strategy="independent" if mode == "independent" else "shared",
            feature_merge_ppm=merge_ppm,
            ppm_tol=ppm,
            mz_min=processing_mz_min,
            mz_max=processing_mz_max,
            resolution=resolution,
            resample_points_per_fwhm=points,
            ms_peak_snr_threshold=spectrum_snr,
            prominence_ratio=None if prominence == 0 else prominence,
            peak_distance=distance,
            cell_threshold=cell_threshold,
            feature_snr=feature_snr,
            baseline_window=baseline,
            noise_window=noise,
            max_zero_fraction=max_zero,
            feature_block_size=block,
            metadata_path=metadata,
            frame_metadata=tuple(frame_metadata),
            create_qc=create_qc,
            msconvert_path=msconvert,
        )
        return run_processing(config, progress_callback=progress)

    if st.button("开始峰提取", type="primary", use_container_width=True):
        result = _execute(st, terminal, "CyESI 峰提取", process, "data")
        if result is not None:
            st.session_state.data = result[0]
            st.rerun()


def _preview_source(st):
    if st.button("📄 选择用于预览的质谱文件"):
        _browse(st, "file", "preview_path")
    current = Path(st.session_state.get("input_path", ""))
    if current.is_dir():
        files = sorted(
            str(path) for path in current.iterdir()
            if path.suffix.lower() in {".mzml", ".mzxml", ".raw"}
        )
        if files:
            configured, help_text = _parameter("preview.directory_file", "")
            selected_index = files.index(configured) if configured in files else 0
            selected = st.selectbox("从当前输入目录选择", files, index=selected_index, help=help_text)
            if not st.session_state.preview_path:
                st.session_state.preview_path = selected
    return st.text_input("预览文件", key="preview_path", help=_parameter("preview.file", "")[1])


def page_preview():
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    from scMM.analysis import cell_detection_preview, chromatogram_preview, spectrum_preview
    from scMM.workflows import parse_reference_mz

    st.header("原始信号交互预览")
    st.caption("查看 TIC、EIC、任意 MS1 扫描的质谱，以及与正式算法一致的细胞提取结果。")
    terminal = _terminal(st)
    path = _preview_source(st)
    c1, c2 = st.columns(2)
    value, help_text = _parameter("preview.msconvert", ""); msconvert = c1.text_input("msconvert.exe（兼容旧配置）", value=str(value), key="preview_msconvert", help=help_text)
    value, help_text = _parameter("preview.ppm", 10.0); ppm = c2.number_input("EIC ppm", 0.01, 1000.0, float(value), help=help_text)

    tic_tab, spectrum_tab, cell_tab, processed_tab = st.tabs(
        ["TIC / EIC", "MS1 质谱", "细胞提取预览", "处理后叠加谱 / 特征"]
    )
    with tic_tab:
        preview_ref_default, preview_ref_help = _parameter("preview.eic_mz", st.session_state.get("extraction_reference", "734.5929"))
        references = st.text_input(
            "EIC m/z（逗号分隔）",
            value=str(preview_ref_default),
            key="preview_references",
            help=preview_ref_help,
        )
        if st.button("读取 TIC / EIC", type="primary"):
            result = _execute(
                st,
                terminal,
                "读取 TIC/EIC",
                lambda progress: chromatogram_preview(
                    path,
                    parse_reference_mz(references),
                    ppm=ppm,
                    msconvert_path=msconvert or None,
                    progress_callback=progress,
                ),
            )
            if result is not None:
                st.session_state.preview_chrom = result[0]
                st.rerun()
        frame = st.session_state.preview_chrom
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["cell_frame"] = False
            data = st.session_state.data
            if data is not None and {"source_file", "frame_id"} <= set(data.obs.columns):
                source_name = Path(path).name
                source_stem = Path(path).stem
                source_mask = data.obs["source_file"].astype(str).map(
                    lambda value: (
                        value.casefold() == source_name.casefold()
                        or Path(value).stem.casefold() == source_stem.casefold()
                    )
                )
                cell_frames = set(
                    data.obs.loc[source_mask, "frame_id"].dropna().astype(int)
                )
                frame["cell_frame"] = frame["frame_id"].isin(cell_frames)
                st.caption(
                    f"已从当前结果标记 {int(frame['cell_frame'].sum())} 个细胞帧"
                    f"（源文件：{source_name}）。"
                )
            x_index, x_help = _choice("preview.x_axis", ["rt", "scan"])
            x_axis = st.radio("横轴", ["rt", "scan"], index=x_index, horizontal=True, help=x_help)
            trace_default, trace_help = _parameter("preview.traces", [])
            trace_options = [
                column for column in frame
                if column not in {"rt", "scan", "frame_id", "cell_frame"}
            ]
            trace_default = [item for item in trace_default if item in trace_options] or trace_options
            traces = st.multiselect(
                "显示曲线", trace_options, default=trace_default, help=trace_help
            )
            long = frame.melt(
                id_vars=["frame_id", "scan", "rt", "cell_frame"],
                value_vars=traces,
                var_name="trace",
                value_name="intensity",
            )
            figure = px.line(
                long,
                x=x_axis,
                y="intensity",
                color="trace",
                render_mode="webgl",
                hover_data=["frame_id"],
            )
            marked = long[long["cell_frame"]]
            if not marked.empty:
                figure.add_scatter(
                    x=marked[x_axis],
                    y=marked["intensity"],
                    mode="markers",
                    marker=dict(size=8, symbol="x", color="black"),
                    text=marked["frame_id"],
                    hovertemplate="cell frame=%{text}<br>intensity=%{y:.6g}<extra></extra>",
                    name="cell frames",
                )
            figure.update_layout(height=600)
            _plotly_with_export(st, figure, "preview_chromatogram")
    with spectrum_tab:
        value, help_text = _parameter("preview.scan", 0); scan = st.number_input("MS1 scan 序号", 0, 10_000_000, int(value), help=help_text)
        a, b = st.columns(2)
        value, help_text = _parameter("preview.mz_min", 100.0); mz_min = a.number_input("m/z 下限", 0.0, 100000.0, float(value), help=help_text)
        value, help_text = _parameter("preview.mz_max", 1000.0); mz_max = b.number_input("m/z 上限", 0.0, 100000.0, float(value), help=help_text)
        value, help_text = _parameter("preview.max_points", 10000); max_points = st.number_input("最大绘图点数（按 m/z 顺序采样）", 100, 100000, int(value), help=help_text)
        value, help_text = _parameter("preview.line_color", "#1f77b4"); line_color = st.color_picker("谱线颜色", str(value), help=help_text)
        if st.button("读取质谱"):
            result = _execute(
                st,
                terminal,
                "读取 MS1 质谱",
                lambda progress: (
                    progress(0.2, "Loading raw-MS file"),
                    spectrum_preview(path, scan, msconvert_path=msconvert or None),
                )[1],
            )
            if result is not None:
                st.session_state.preview_spectrum = result
                st.rerun()
        result = st.session_state.preview_spectrum
        if result is not None:
            frame, rt, count, _ = result
            subset = frame[frame["mz"].between(mz_min, mz_max)]
            if len(subset) > max_points:
                selected = np.linspace(0, len(subset) - 1, max_points, dtype=int)
                subset = subset.iloc[np.unique(selected)]
            figure = go.Figure(
                go.Scattergl(
                    x=subset["mz"],
                    y=subset["intensity"],
                    mode="lines",
                    line=dict(width=1, color=line_color),
                    hovertemplate="m/z=%{x:.6f}<br>Intensity=%{y:.6g}<extra></extra>",
                )
            )
            figure.update_layout(title=f"MS1 scan {scan}/{count - 1}, RT={rt:g}", xaxis_title="m/z", yaxis_title="Intensity", height=600)
            _plotly_with_export(st, figure, "preview_ms1_spectrum")
    with cell_tab:
        value, help_text = _parameter("preview.cell_reference", st.session_state.get("extraction_reference", "734.5929")); references = st.text_input("Reference m/z", value=str(value), key="cell_preview_refs", help=help_text)
        a, b, c = st.columns(3)
        index, help_text = _choice("preview.reference_mode", ["union", "intersection"]); ref_mode = a.selectbox("Reference 组合", ["union", "intersection"], index=index, key="cell_preview_mode", help=help_text)
        value, help_text = _parameter("preview.cell_threshold", 5.0); threshold = b.number_input("Reference/基线阈值", 0.0, 10000.0, float(value), key="cell_preview_threshold", help=help_text)
        value, help_text = _parameter("preview.feature_snr", 3.0); feature_snr = c.number_input("Feature SNR", 0.0, 1000.0, float(value), key="cell_preview_snr", help=help_text)
        value, help_text = _parameter("preview.baseline_window", 51); baseline = a.number_input("基线窗口", 3, 10001, int(value), step=2, key="cell_preview_baseline", help=help_text)
        value, help_text = _parameter("preview.noise_window", 51); noise = b.number_input("噪声窗口", 3, 10001, int(value), step=2, key="cell_preview_noise", help=help_text)
        value, help_text = _parameter("preview.resolution", 35000.0); resolution = c.number_input("分辨率", 100.0, 2_000_000.0, float(value), key="cell_preview_resolution", help=help_text)
        d, e = st.columns(2)
        value, help_text = _parameter("preview.processing_mz_min", 100.0); cell_mz_min = d.number_input("预览处理 m/z 下限", 0.000001, 100000.0, float(value), key="cell_preview_mz_min", help=help_text)
        value, help_text = _parameter("preview.processing_mz_max", 1000.0); cell_mz_max = e.number_input("预览处理 m/z 上限", 0.000001, 100000.0, float(value), key="cell_preview_mz_max", help=help_text)
        if st.button("运行细胞提取预览", type="primary"):
            result = _execute(
                st,
                terminal,
                "细胞提取预览",
                lambda progress: cell_detection_preview(
                    path,
                    parse_reference_mz(references),
                    reference_mode=ref_mode,
                    msconvert_path=msconvert or None,
                    progress_callback=progress,
                    ppm_tol=ppm,
                    mz_range=(cell_mz_min, cell_mz_max),
                    resolution=resolution,
                    cell_signal_threshold=threshold,
                    feature_snr_threshold=feature_snr,
                    baseline_window=baseline,
                    noise_window=noise,
                ),
            )
            if result is not None:
                st.session_state.preview_cells = result[0]
                st.rerun()
        frame = st.session_state.preview_cells
        if frame is not None:
            signals = [column for column in frame if column.startswith("m/z ")]
            long = frame.reset_index().melt(id_vars=["x", "cell_window", "cell_apex"], value_vars=signals, var_name="reference", value_name="intensity")
            figure = px.line(long, x="x", y="intensity", color="reference", render_mode="webgl")
            apex = long[long["cell_apex"]]
            figure.add_scatter(x=apex["x"], y=apex["intensity"], mode="markers", marker=dict(size=9, symbol="x", color="black"), name="cell apex")
            figure.update_layout(height=620, xaxis_title="RT / frame")
            _plotly_with_export(st, figure, "preview_cell_extraction")
    with processed_tab:
        data = st.session_state.data
        spectra = {} if data is None else data.uns.get("summed_spectra", {})
        if not spectra:
            st.info("当前数据不含叠加谱。请先使用新版程序完成峰提取；旧 H5AD 不会自动补充原始叠加谱。")
        else:
            names = list(spectra)
            configured, help_text = _parameter("preview.processed_spectrum", "")
            selected_name = st.selectbox(
                "叠加谱来源",
                names,
                index=names.index(configured) if configured in names else 0,
                help=help_text,
            )
            payload = spectra[selected_name]
            summed_mz = np.asarray(payload["mz"], dtype=float)
            summed_intensity = np.asarray(payload["intensity"], dtype=float)
            value, help_text = _bounded_default(
                "preview.summed_max_points", 30000, 1000, 200000
            )
            max_points = st.number_input(
                "叠加谱最大绘图点数", 1000, 200000, int(value), help=help_text
            )
            if len(summed_mz) > max_points:
                selected_indices = np.unique(
                    np.linspace(0, len(summed_mz) - 1, max_points, dtype=int)
                )
                plot_mz = summed_mz[selected_indices]
                plot_intensity = summed_intensity[selected_indices]
            else:
                plot_mz, plot_intensity = summed_mz, summed_intensity
            figure = go.Figure()
            figure.add_scatter(
                x=plot_mz,
                y=plot_intensity,
                mode="lines",
                line=dict(width=1, color="#4c78a8"),
                name="summed MS1 spectrum",
            )
            a, b = st.columns(2)
            value, help_text = _parameter("preview.show_candidates", False)
            show_candidates = a.checkbox("标记候选峰", bool(value), help=help_text)
            value, help_text = _parameter("preview.show_retained", True)
            show_retained = b.checkbox("标记最终保留特征", bool(value), help=help_text)
            marker_sets = []
            if show_candidates:
                marker_sets.append(("candidate peaks", payload.get("candidate_mz", []), "#f2a900", "circle-open"))
            if show_retained:
                marker_sets.append(("retained features", payload.get("retained_feature_mz", data.var["mz"]), "#d62728", "x"))
            from scMM.plot.diagnostics import spectrum_curve
            all_marker_mz = np.concatenate([np.asarray(values, dtype=float) for _, values, _, _ in marker_sets]) if marker_sets else np.array([])
            plot_mz, plot_intensity = spectrum_curve(summed_mz, summed_intensity, max_points, all_marker_mz)
            figure.data[0].x = plot_mz
            figure.data[0].y = plot_intensity
            for label, marker_mz, color, symbol in marker_sets:
                marker_mz = np.asarray(marker_mz, dtype=float)
                marker_mz = marker_mz[(marker_mz >= plot_mz[0]) & (marker_mz <= plot_mz[-1])]
                if marker_mz.size:
                    figure.add_scatter(
                        x=marker_mz,
                        y=np.interp(marker_mz, plot_mz, plot_intensity),
                        mode="markers",
                        marker=dict(size=8, color=color, symbol=symbol),
                        name=label,
                        hovertemplate="m/z=%{x:.6f}<br>intensity=%{y:.6g}<extra></extra>",
                    )
            figure.update_layout(
                height=650,
                title=f"处理后叠加 MS1 谱：{selected_name}",
                xaxis_title="m/z",
                yaxis_title="Summed intensity",
            )
            _plotly_with_export(st, figure, "processed_summed_spectrum")
            st.subheader(f"最终特征表（{data.n_vars} 个）")
            feature_table = data.var.copy()
            feature_table.insert(0, "feature_id", data.var_names.astype(str))
            st.dataframe(feature_table, use_container_width=True, height=420, hide_index=True)


def _gate_coordinates(data):
    options = {}
    for key in data.obsm:
        values = np.asarray(data.obsm[key])
        if values.ndim == 2:
            for component in range(values.shape[1]):
                options[f"{key} · component {component + 1}"] = (
                    "obsm",
                    str(key),
                    component,
                )
    for column in data.obs:
        if pd.api.types.is_numeric_dtype(data.obs[column]):
            options[f"obs · {column}"] = ("obs", str(column), None)
    for index, feature in enumerate(data.var_names.astype(str)):
        mz = data.var.iloc[index].get("mz", feature)
        options[f"feature · {mz}"] = ("feature", feature, None)
    return options


def _gate_values(data, descriptor):
    source, key, component = descriptor
    if source == "obsm":
        return np.asarray(data.obsm[key], dtype=float)[:, int(component)]
    if source == "obs":
        return pd.to_numeric(data.obs[key], errors="coerce").to_numpy(dtype=float)
    return _dense(data[:, key].X).ravel().astype(float, copy=False)


def _gate_selected_cells(selection, plotted_positions=None):
    if not selection:
        return np.array([], dtype=int)
    state = (
        selection.get("selection")
        if hasattr(selection, "get")
        else getattr(selection, "selection", None)
    )
    if not state:
        return np.array([], dtype=int)
    plotted_positions = (
        np.asarray(plotted_positions, dtype=int)
        if plotted_positions is not None
        else None
    )
    positions = []
    for point in state.get("points", []):
        custom = point.get("customdata")
        value = None
        if hasattr(custom, "get"):
            value = custom.get("cell_position")
            if value is None and len(custom) == 1:
                value = next(iter(custom.values()))
        elif isinstance(custom, (list, tuple, np.ndarray)) and len(custom):
            value = custom[0]
        elif custom is not None:
            value = custom
        if value is None and plotted_positions is not None:
            point_index = point.get("point_index", point.get("pointNumber"))
            if point_index is not None and 0 <= int(point_index) < len(plotted_positions):
                value = plotted_positions[int(point_index)]
        if value is not None:
            positions.append(int(value))
    return np.unique(positions).astype(int)


def _safe_filter_suffix(value, fallback):
    text = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip()).strip("_")
    return text or fallback


def _cell_filter_controls(st, data):
    columns = data.cell_filter_columns()
    if not columns:
        return
    st.markdown("##### 细胞标签管理")
    active = list(data.uns.get("active_cell_filters", columns))
    active = [name for name in active if name in columns]
    selected = st.multiselect(
        "参与后续分析的细胞筛选标签",
        columns,
        default=active,
        help="启用的标签按逻辑与组合；任一标签为 0 的细胞会在后续分析中被忽略。",
    )
    if selected != active:
        data.set_active_cell_filters(selected)
    summary = pd.DataFrame(
        {
            "标签": columns,
            "保留": [int(data.obs[name].sum()) for name in columns],
            "忽略": [int((data.obs[name] == 0).sum()) for name in columns],
            "启用": [name in selected for name in columns],
        }
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)
    restore = st.selectbox("选择要还原的细胞标签", columns, key="restore_cell_filter")
    if st.button("将该标签全部还原为 1", key="restore_cell_filter_button"):
        data.restore_cell_filter(restore)
        st.rerun()


def _feature_filter_controls(st, data):
    columns = data.feature_filter_columns()
    if not columns:
        return
    st.markdown("##### 特征标签管理")
    active = list(data.uns.get("active_feature_filters", columns))
    active = [name for name in active if name in columns]
    selected = st.multiselect(
        "参与后续分析的特征筛选标签",
        columns,
        default=active,
        help="启用的标签按逻辑与组合；任一标签为 0 的特征会在后续分析中被忽略。",
    )
    if selected != active:
        data.set_active_feature_filters(selected)
    summary = pd.DataFrame(
        {
            "标签": columns,
            "保留": [int(data.var[name].sum()) for name in columns],
            "忽略": [int((data.var[name] == 0).sum()) for name in columns],
            "启用": [name in selected for name in columns],
        }
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)
    restore = st.selectbox("选择要还原的特征标签", columns, key="restore_feature_filter")
    if st.button("将该标签全部还原为 1", key="restore_feature_filter_button"):
        data.restore_feature_filter(restore)
        st.rerun()


def page_preprocess():
    import plotly.express as px
    import streamlit as st

    from scMM.analysis import quality_metrics

    st.header("归一化、插补与质量控制")
    _pipeline_status(st)
    terminal = _terminal(st)
    data = st.session_state.data
    if data is None:
        st.warning("请先在“数据导入”页面加载或处理数据。")
        return
    data.ensure_matrix_layers()
    st.subheader("归一化与标准化")
    st.caption(f"输入层：{data.uns['active_layer']}。原始 raw 层保留；已有低维坐标保持原结果，切换数据层后可重新运行降维。")
    options = ["total", "max", "quantile", "pqn", "zscore", "log", "minmax"]
    index, help_text = _choice("preprocess.normalize_method", options); method = st.selectbox("方法", options, index=index, help=help_text)
    a, b, c = st.columns(3)
    value, help_text = _parameter("preprocess.total_scale", 1.0); scale = a.number_input("Total scale", value=float(value), format="%.6g", help=help_text)
    index, help_text = _choice("preprocess.axis", [0, 1]); axis = b.selectbox("Axis", [0, 1], index=index, help=help_text)
    value, help_text = _parameter("preprocess.log_pseudo", 1e-6); pseudo = c.number_input("Log pseudo", value=float(value), format="%.8g", help=help_text)
    index, help_text = _choice("preprocess.pqn_reference", ["median", "mean"]); pqn_reference = a.selectbox("PQN reference", ["median", "mean"], index=index, help=help_text)
    layer_name = st.text_input("归一化/标准化输出 layer 名", value=f"normalized_{method}", key=f"normalize_output_{method}", help="同名层已存在时自动追加编号，保留之前的结果。")
    if st.button("应用归一化与标准化", type="primary"):
        params = {"total": {"scale": scale}, "zscore": {"axis": axis}, "minmax": {"axis": axis}, "log": {"pseudo": pseudo}, "pqn": {"reference": pqn_reference}}.get(method, {})
        result = _execute(st, terminal, "归一化与标准化", lambda progress: (progress(0.15, "Preparing selected layer"), data.normalize(method, layer_key=layer_name, **params), progress(0.9, "Saving output layer"))[1], "preprocess")
        if result is not None: st.rerun()
    st.subheader("缺失值插补")
    options = ["knn", "mean", "median", "most_frequent", "constant"]
    index, help_text = _choice("preprocess.impute_method", options); method = st.selectbox("插补方法", options, index=index, help=help_text)
    a, b = st.columns(2)
    value, help_text = _parameter("preprocess.knn_neighbors", 5); neighbors = a.number_input("KNN 邻居", 1, 100, int(value), help=help_text)
    value, help_text = _parameter("preprocess.constant_fill", 0.0); fill = b.number_input("Constant 填充值", value=float(value), help=help_text)
    impute_layer = st.text_input("插补输出 layer 名", value=f"imputed_{method}", key=f"impute_output_{method}")
    if st.button("应用插补"):
        kwargs = {"n_neighbors": neighbors} if method == "knn" else ({"fill_value": fill} if method == "constant" else {})
        result = _execute(st, terminal, "缺失值插补", lambda progress: (progress(0.1, "Fitting imputer"), data.impute(method, layer_key=impute_layer, **kwargs), progress(0.9, "Saving imputed layer"))[1], "preprocess")
        if result is not None: st.rerun()
    data.ensure_cell_quality_metrics()
    st.subheader("细胞异常点与可逆圈门")
    value, help_text = _parameter("preprocess.outlier_contamination", 0.05); contamination = st.number_input("Isolation Forest contamination", 0.001, 0.5, float(value), help=help_text)
    value, help_text = _parameter("preprocess.outlier_label", "isolation_forest"); isolation_key = st.text_input("Isolation Forest 标签名", value=str(value), help=help_text)
    if st.button("检测并标记异常细胞"):
        result = _execute(st, terminal, "异常细胞检测", lambda progress: (progress(0.15, "Fitting Isolation Forest"), data.remove_outlier(key=_safe_filter_suffix(isolation_key, "isolation_forest"), contamination=contamination, random_state=42), progress(0.9, "Writing uint8 obs label"))[1], "preprocess")
        if result is not None:
            column = f"cell_keep_{_safe_filter_suffix(isolation_key, 'isolation_forest')}"
            terminal.write(f"Tagged {(data.obs[column] == 0).sum()} cells in obs[{column!r}]")
            st.rerun()

    coordinate_options = _gate_coordinates(data)
    if coordinate_options:
        st.markdown("##### 二维圈门")
        labels = list(coordinate_options)
        default_x = next((name for name in labels if "X_pca" in name and "component 1" in name), labels[0])
        default_y = next((name for name in labels if "X_pca" in name and "component 2" in name), labels[min(1, len(labels) - 1)])
        a, b = st.columns(2)
        configured, x_help = _parameter("preprocess.gate_x", ""); x_label = a.selectbox("横坐标", labels, index=labels.index(configured) if configured in labels else labels.index(default_x), key="gate_x", help=x_help)
        configured, y_help = _parameter("preprocess.gate_y", ""); y_label = b.selectbox("纵坐标", labels, index=labels.index(configured) if configured in labels else labels.index(default_y), key="gate_y", help=y_help)
        x = _gate_values(data, coordinate_options[x_label])
        y = _gate_values(data, coordinate_options[y_label])
        gate_frame = pd.DataFrame(
            {
                "x": x,
                "y": y,
                "cell": data.obs_names.astype(str),
                "cell_position": np.arange(data.n_obs),
                "status": np.where(data.cell_keep_mask(), "active", "ignored"),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
        a, b = st.columns(2)
        color_modes = ["有效状态", "特征 / obs 数值", "二维点密度"]
        index, help_text = _choice("preprocess.gate_color_mode", color_modes)
        gate_color_mode = a.selectbox("圈门上色方式", color_modes, index=index, help=help_text, key="gate_color_mode")
        cmap_options = ["Viridis", "Turbo", "Plasma", "Inferno", "Cividis", "RdBu", "Spectral"]
        index, help_text = _choice("preprocess.gate_cmap", cmap_options)
        gate_cmap = b.selectbox("圈门 cmap", cmap_options, index=index, help=help_text, key="gate_cmap")
        color_column = "status"
        if gate_color_mode == "特征 / obs 数值":
            color_options = {name: descriptor for name, descriptor in coordinate_options.items() if descriptor[0] in {"feature", "obs"}}
            color_name = st.selectbox("圈门颜色数值", list(color_options), key="gate_color_value")
            gate_frame["color_value"] = _gate_values(data, color_options[color_name])[gate_frame["cell_position"].to_numpy()]
            color_column = "color_value"
        elif gate_color_mode == "二维点密度":
            from scMM.plot.diagnostics import point_density
            value, help_text = _parameter("preprocess.gate_density_smoothing", 2.0)
            smoothing = st.number_input("密度平滑带宽（网格单位）", .5, 10.0, float(value), .5, help=help_text, key="gate_density_smoothing")
            gate_frame["density"] = point_density(gate_frame.x, gate_frame.y, smoothing=smoothing)
            gate_frame = gate_frame.sort_values("density")
            color_column = "density"
            st.caption("颜色表示当前坐标平面上的二维点密度：二维直方图经高斯核平滑，再插值到每个细胞；高密度点最后绘制。")
        index, help_text = _choice("discovery.color_scale", ["linear", "log10", "symlog"])
        gate_scale = st.selectbox("圈门伪色 colorbar 尺度", ["linear", "log10", "symlog"], index=index,
                                 disabled=color_column == "status", help=help_text, key="gate_color_scale")
        gate_threshold = st.number_input("圈门对称对数阈值", min_value=1e-12, value=1.0, format="%.6g") if gate_scale == "symlog" and color_column != "status" else 1.0
        from scMM.plot.diagnostics import color_coordinates
        color_valid = np.ones(len(gate_frame), dtype=bool)
        plotted_color = color_column
        if color_column != "status" and gate_scale != "linear":
            gate_frame["__gate_scaled"] = color_coordinates(gate_frame[color_column].to_numpy(float), gate_scale, gate_threshold)
            color_valid = np.isfinite(gate_frame["__gate_scaled"].to_numpy())
            plotted_color = "__gate_scaled"
        gate_figure = px.scatter(
            gate_frame.loc[color_valid],
            x="x",
            y="y",
            color=plotted_color,
            color_continuous_scale=gate_cmap,
            hover_name="cell",
            custom_data=["cell_position"],
            color_discrete_map={"active": "#2563eb", "ignored": "#b8b8b8"},
            render_mode="webgl",
        )
        if color_column != "status":
            raw_color = gate_frame.loc[color_valid, color_column].to_numpy(float)
            for trace in gate_figure.data:
                trace.meta = {"scmm_color_values": raw_color.tolist()}
            if len(raw_color) and gate_scale != "linear":
                ticks = np.unique(np.geomspace(raw_color.min(), raw_color.max(), 5) if gate_scale == "log10" else np.linspace(raw_color.min(), raw_color.max(), 7))
                gate_figure.update_coloraxes(colorbar=dict(title=f"{color_column} ({gate_scale})",
                                                          tickvals=color_coordinates(ticks, gate_scale, gate_threshold), ticktext=[f"{value:.4g}" for value in ticks]))
            if not color_valid.all():
                gray = px.scatter(gate_frame.loc[~color_valid], x="x", y="y", hover_name="cell", custom_data=["cell_position"])
                for trace in gray.data:
                    trace.marker.color = "#b8b8b8"
                    trace.name = "非正值 / 缺失值"
                    trace.showlegend = True
                    gate_figure.add_trace(trace)
        gate_figure.update_layout(
            height=650,
            dragmode="lasso",
            xaxis_title=x_label,
            yaxis_title=y_label,
            meta={"scmm_cmap": gate_cmap, "scmm_color_scale": gate_scale if color_column != "status" else "linear",
                  "scmm_color_label": color_column, "scmm_linthresh": gate_threshold},
        )
        from scMM.plot.interaction import constrain_zoom
        constrain_zoom(gate_figure)
        selection = st.plotly_chart(
            gate_figure,
            use_container_width=True,
            key="cell_gate_plot",
            on_select="rerun",
            selection_mode=("lasso", "box", "points"),
        )
        _plotly_with_export(st, gate_figure, "cell_gate", render=False)
        selected_cells = _gate_selected_cells(
            selection,
            gate_frame["cell_position"].to_numpy(dtype=int),
        )
        st.caption(f"当前圈中 {len(selected_cells)} 个细胞。可使用套索、矩形框或点选。")
        a, b = st.columns(2)
        configured, gate_help = _parameter("preprocess.gate_label", "manual_gate"); gate_name = a.text_input("圈门标签名", value=str(configured), help=gate_help)
        mode_options = ["正选：圈中细胞标记为 0", "负选：圈外细胞标记为 0"]
        configured, mode_help = _parameter("preprocess.gate_mode", mode_options[0])
        gate_mode = b.radio(
            "圈门方向",
            mode_options,
            index=mode_options.index(configured) if configured in mode_options else 0,
            horizontal=False,
            help=mode_help,
        )
        if st.button("写入圈门标签", type="primary"):
            if not len(selected_cells):
                st.warning("请先在二维图中圈选细胞。")
            else:
                keep = np.ones(data.n_obs, dtype=np.uint8)
                if gate_mode.startswith("正选"):
                    keep[selected_cells] = 0
                else:
                    keep[:] = 0
                    keep[selected_cells] = 1
                column = data.set_cell_filter(
                    _safe_filter_suffix(gate_name, "manual_gate"), keep
                )
                terminal.write(f"Gate written to obs[{column!r}]")
                st.rerun()

    _cell_filter_controls(st, data)

    st.subheader("特征检出率筛选")
    a, b, c = st.columns(3)
    scope_options = ["全部细胞", "指定文件"]
    configured, scope_help = _parameter("preprocess.detection_scope", scope_options[0]); detection_scope = a.radio("检出率范围", scope_options, index=scope_options.index(configured) if configured in scope_options else 0, horizontal=True, help=scope_help)
    source_file = None
    if detection_scope == "指定文件":
        files = sorted(data.obs["source_file"].dropna().astype(str).unique()) if "source_file" in data.obs else []
        source_file = b.selectbox("文件", files) if files else None
    configured, threshold_help = _parameter("preprocess.detection_threshold", 0.1); threshold = c.number_input("最低检出率", 0.0, 1.0, float(configured), step=0.01, help=threshold_help)
    configured, label_help = _parameter("preprocess.detection_label", "detection_rate"); feature_key = st.text_input("特征筛选标签名", value=str(configured), help=label_help)
    if st.button("计算检出率并写入特征标签"):
        result = data.label_features_by_detection(
            threshold,
            source_file=source_file,
            key=_safe_filter_suffix(feature_key, "detection_rate"),
        )
        terminal.write(f"Feature detection filter written to var[{result[0]!r}]")
        st.rerun()
    _feature_filter_controls(st, data)

    st.subheader("交互式 QC")
    metrics = quality_metrics(data)["cell"]
    metric_options = [column for column in metrics if pd.api.types.is_numeric_dtype(metrics[column])]
    configured, help_text = _parameter("preprocess.qc_metric", "")
    metric = st.selectbox("指标", metric_options, index=metric_options.index(configured) if configured in metric_options else 0, help=help_text)
    color = "source_file" if "source_file" in metrics else None
    from scMM.plot.diagnostics import qc_distribution
    value, help_text = _parameter("preprocess.qc_bins", 40)
    bins = st.number_input("QC 直方图箱数", 5, 200, int(value), help=help_text, key="qc_bins")
    figure = qc_distribution(metrics, metric, bins=bins)
    _plotly_with_export(st, figure, "qc_histogram")
    _heatmap_ui(st, data, "file_qc")
    from scMM.group_sets_ui import render_group_sets
    render_group_sets(st, data, key="qc_sets", annotation=data.uns.get("annotation"))


def page_discovery():
    import streamlit as st

    st.header("降维、聚类与轨迹分析")
    _pipeline_status(st)
    terminal = _terminal(st)
    data = st.session_state.data
    if data is None:
        st.warning("请先加载数据。")
        return
    reduction_tab, cluster_tab, trajectory_tab, plot_tab = st.tabs(["降维", "聚类", "轨迹", "嵌入图"])
    with reduction_tab:
        method_options = ["pca", "umap", "tsne", "isomap", "lle"]
        index, help_text = _choice("discovery.reduction_method", method_options); method = st.selectbox("降维方法", method_options, index=index, help=help_text)
        reps = ["当前 X（所选 layer）", *data.obsm.keys()]
        configured, help_text = _parameter("discovery.reduction_rep", "当前 X（所选 layer）")
        fallback_index = 0 if method == "pca" else (reps.index("X_pca") if "X_pca" in reps else 0)
        use_rep_label = st.selectbox("输入表示", reps, index=reps.index(configured) if configured in reps else fallback_index, help=help_text)
        use_rep = None if use_rep_label == "当前 X（所选 layer）" else use_rep_label
        a, b, c = st.columns(3)
        value, help_text = _parameter("discovery.components", 2); components = a.number_input("输出维数", 1, 100, int(value), help=help_text)
        value, help_text = _parameter("discovery.scale", True); scale = b.checkbox("标准化输入", bool(value), help=help_text)
        value, help_text = _parameter("discovery.seed", 42); seed = c.number_input("随机种子", 0, 2_147_483_647, int(value), help=help_text)
        params = {}
        if method == "umap":
            upper = max(2, data.n_obs - 1); value, help_text = _bounded_default("discovery.neighbors", min(30, upper), 2, upper); params["n_neighbors"] = a.number_input("n_neighbors", 2, upper, int(value), help=help_text)
            value, help_text = _parameter("discovery.umap_min_dist", 0.3); params["min_dist"] = b.number_input("min_dist", 0.0, 1.0, float(value), help=help_text)
            options = ["euclidean", "cosine", "correlation", "manhattan"]; index, help_text = _choice("discovery.metric", options); params["metric"] = c.selectbox("metric", options, index=index, help=help_text)
        elif method == "tsne":
            upper = float(max(2, data.n_obs - 1)); value, help_text = _bounded_default("discovery.tsne_perplexity", min(30.0, upper), 1.0, upper); params["perplexity"] = a.number_input("perplexity", 1.0, upper, float(value), help=help_text)
            value, help_text = _parameter("discovery.tsne_exaggeration", 12.0); params["early_exaggeration"] = b.number_input("early_exaggeration", 1.0, 100.0, float(value), help=help_text)
            value, help_text = _parameter("discovery.tsne_learning_rate", 200.0); params["learning_rate"] = c.number_input("learning_rate", 1.0, 10000.0, float(value), help=help_text)
            options = ["pca", "random"]; index, help_text = _choice("discovery.tsne_init", options); params["init"] = a.selectbox("init", options, index=index, help=help_text)
            options = ["euclidean", "cosine", "correlation", "manhattan"]; index, help_text = _choice("discovery.metric", options); params["metric"] = b.selectbox("t-SNE metric", options, index=index, help=help_text)
            value, help_text = _parameter("discovery.tsne_iterations", 1000); params["max_iter"] = c.number_input("max_iter", 250, 10000, int(value), help=help_text)
        elif method == "isomap":
            upper = max(2, data.n_obs - 1); value, help_text = _bounded_default("discovery.neighbors", min(15, upper), 2, upper); params["n_neighbors"] = a.number_input("n_neighbors", 2, upper, int(value), key="isomap_neighbors", help=help_text)
            options = ["euclidean", "manhattan", "cosine"]; index, help_text = _choice("discovery.metric", options); params["metric"] = b.selectbox("Isomap metric", options, index=index, help=help_text)
            options = ["auto", "FW", "D"]; index, help_text = _choice("discovery.isomap_path", options); params["path_method"] = c.selectbox("path_method", options, index=index, help=help_text)
        elif method == "lle":
            upper = max(2, data.n_obs - 1); value, help_text = _bounded_default("discovery.neighbors", min(15, upper), 2, upper); params["n_neighbors"] = a.number_input("n_neighbors", 2, upper, int(value), key="lle_neighbors", help=help_text)
            options = ["standard", "modified", "hessian", "ltsa"]; index, help_text = _choice("discovery.lle_method", options); params["method"] = b.selectbox("LLE method", options, index=index, help=help_text)
            options = ["auto", "arpack", "dense"]; index, help_text = _choice("discovery.eigen_solver", options); params["eigen_solver"] = c.selectbox("eigen_solver", options, index=index, help=help_text)
        configured, help_text = _parameter("discovery.store_key", ""); store_key = st.text_input("写入 obsm 的键", value=str(configured or f"X_{method}"), key=f"store_{method}", help=help_text)
        if st.button("运行降维", type="primary"):
            result = _execute(
                st,
                terminal,
                f"{method.upper()} 降维",
                lambda progress: (
                    progress(0.1, "Preparing representation"),
                    _engine(st).reduce_dimension(method, use_rep=use_rep, n_components=components, scale=scale, random_state=seed, store_key=store_key, **params),
                    progress(0.92, f"Writing obsm[{store_key!r}]"),
                )[1],
                "embedding",
            )
            if result is not None: st.rerun()
    with cluster_tab:
        options = ["leiden", "louvain", "kmeans", "agglomerative", "dbscan"]; index, help_text = _choice("discovery.cluster_method", options); method = st.selectbox("聚类算法", options, index=index, help=help_text)
        rep_options = ["当前 X（所选 layer）", *data.obsm.keys()]; index, help_text = _choice("discovery.cluster_rep", rep_options); representation = st.selectbox("聚类输入", rep_options, index=index, key="cluster_rep", help=help_text)
        configured, help_text = _parameter("discovery.cluster_key", ""); key_added = st.text_input("写入 obs 的标签列名", value=str(configured or f"{method}_cluster"), help=help_text)
        a, b, c = st.columns(3)
        value, help_text = _parameter("discovery.cluster_count", 4); cluster_count = a.number_input("聚类数", 2, 200, int(value), help=help_text)
        upper = max(2, data.n_obs - 1); value, help_text = _bounded_default("discovery.graph_neighbors", min(15, upper), 2, upper); graph_neighbors = b.number_input("图邻居数", 2, upper, int(value), help=help_text)
        value, help_text = _parameter("discovery.resolution", 1.0); resolution = c.number_input("Leiden/Louvain resolution", 0.01, 20.0, float(value), help=help_text)
        value, help_text = _parameter("discovery.dbscan_eps", 0.5); eps = a.number_input("DBSCAN eps", 0.001, 100.0, float(value), help=help_text)
        value, help_text = _parameter("discovery.dbscan_min_samples", 5); min_samples = b.number_input("DBSCAN min_samples", 1, 1000, int(value), help=help_text)
        value, help_text = _parameter("discovery.cluster_seed", 42); seed = c.number_input("聚类随机种子", 0, 2_147_483_647, int(value), help=help_text)
        if st.button("运行聚类并写入 obs", type="primary"):
            kwargs = ({"n_clusters": cluster_count} if method in {"kmeans", "agglomerative"} else ({"eps": eps, "min_samples": min_samples} if method == "dbscan" else {"n_neighbors": graph_neighbors, "resolution": resolution}))
            use_rep = None if representation == "当前 X（所选 layer）" else representation
            result = _execute(st, terminal, f"{method} 聚类", lambda progress: (progress(0.1, "Preparing graph / matrix"), _engine(st).cluster_cells(method, use_rep=use_rep, key_added=key_added, random_state=seed, **kwargs), progress(0.9, f"Writing obs[{key_added!r}]"))[1], "cluster")
            if result is not None:
                terminal.write(f"Generated {result.nunique()} clusters in obs[{key_added!r}]")
                st.rerun()
    with trajectory_tab:
        methods = {
            "Diffusion pseudotime (DPT)": "diffusion_pseudotime",
            "kNN graph geodesic pseudotime": "graph_pseudotime",
            "PCA pseudotime": "pca_pseudotime",
            "PAGA-like cluster graph pseudotime": "paga_pseudotime",
        }
        method_options = list(methods); index, help_text = _choice("discovery.trajectory_method", method_options); method_label = st.selectbox("轨迹方法", method_options, index=index, help=help_text)
        method = methods[method_label]
        rep_options = ["当前 X（所选 layer）", *data.obsm.keys()]; index, help_text = _choice("discovery.trajectory_rep", rep_options); representation = st.selectbox("轨迹输入", rep_options, index=index, key="trajectory_rep", help=help_text)
        configured, help_text = _parameter("discovery.pseudotime_key", ""); key_added = st.text_input("写入 obs 的伪时间列", value=str(configured or f"{method}_value"), help=help_text)
        a, b, c = st.columns(3)
        upper = max(2, data.n_obs - 1); value, help_text = _bounded_default("discovery.trajectory_neighbors", min(15, upper), 2, upper); neighbors = a.number_input("轨迹图邻居数", 2, upper, int(value), help=help_text)
        value, help_text = _parameter("discovery.diffusion_components", 10); components = b.number_input("Diffusion components", 1, 50, int(value), help=help_text)
        cluster_columns = [column for column in data.obs if not pd.api.types.is_numeric_dtype(data.obs[column])]
        cluster_options = cluster_columns or data.obs.columns.tolist(); configured, help_text = _parameter("discovery.paga_cluster_key", ""); cluster_key = c.selectbox("PAGA 聚类列", cluster_options, index=cluster_options.index(configured) if configured in cluster_options else 0, help=help_text)
        root_options = ["第一细胞", "指定 cell ID", "按 obs 分组选代表细胞"]; index, help_text = _choice("discovery.root_mode", root_options); root_mode = st.radio("根细胞定义", root_options, index=index, horizontal=True, help=help_text)
        root_cell = None; root_obs = None; root_value = None
        if root_mode == "指定 cell ID":
            options = data.obs_names.astype(str).tolist(); configured, help_text = _parameter("discovery.root_cell", ""); root_cell = st.selectbox("Root cell", options, index=options.index(configured) if configured in options else 0, help=help_text)
        elif root_mode == "按 obs 分组选代表细胞":
            options = data.obs.columns.astype(str).tolist(); configured, help_text = _parameter("discovery.root_obs", ""); root_obs = st.selectbox("Root obs", options, index=options.index(configured) if configured in options else 0, help=help_text)
            options = sorted(data.obs[root_obs].dropna().astype(str).unique()); configured, help_text = _parameter("discovery.root_value", ""); root_value = st.selectbox("Root value", options, index=options.index(configured) if configured in options else 0, help=help_text)
        if st.button("运行轨迹并写入 obs", type="primary"):
            use_rep = None if representation == "当前 X（所选 layer）" else representation
            result = _execute(st, terminal, method_label, lambda progress: (progress(0.1, "Building trajectory graph"), _engine(st).trajectory(method, use_rep=use_rep, key_added=key_added, root_cell=root_cell, root_obs=root_obs, root_value=root_value, n_neighbors=neighbors, n_components=components, cluster_key=cluster_key), progress(0.92, f"Writing obs[{key_added!r}]"))[1], "trajectory")
            if result is not None:
                terminal.write(f"Pseudotime written to obs[{key_added!r}]")
                if method == "diffusion_pseudotime": terminal.write("Diffusion components written to obsm['X_diffmap']")
                st.rerun()
    with plot_tab:
        embeddings = list(data.obsm.keys())
        if not embeddings:
            st.info("请先运行降维。")
        else:
            configured, help_text = _parameter("discovery.embedding", ""); embedding = st.selectbox("嵌入", embeddings, index=embeddings.index(configured) if configured in embeddings else 0, help=help_text)
            options = ["none", "obs", "feature"]; index, help_text = _choice("discovery.color_mode", options); color_mode = st.radio("上色方式", options, index=index, horizontal=True, help=help_text)
            a, b = st.columns(2)
            cmap_options = ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "RdBu", "Spectral"]
            index, help_text = _choice("discovery.cmap", cmap_options)
            cmap = a.selectbox(
                "连续变量 cmap",
                cmap_options,
                index=index,
                help=help_text,
            )
            palette_options = ["Plotly", "D3", "G10", "Safe", "Dark24", "Alphabet"]
            index, help_text = _choice("discovery.palette", palette_options)
            categorical_palette = b.selectbox(
                "分类变量配色",
                palette_options,
                index=index,
                help=help_text,
            )
            color_key = None; feature = None
            if color_mode == "obs":
                options = data.obs.columns.astype(str).tolist(); configured, help_text = _parameter("discovery.color_obs", ""); color_key = st.selectbox("obs 项", options, index=options.index(configured) if configured in options else 0, help=help_text)
            elif color_mode == "feature":
                options = data.var_names.astype(str).tolist(); configured, help_text = _parameter("discovery.color_feature", ""); feature = st.selectbox("特征", options, index=options.index(configured) if configured in options else 0, help=help_text)
            continuous = color_mode == "feature" or (color_mode == "obs" and pd.api.types.is_numeric_dtype(data.obs[color_key]))
            index, scale_help = _choice("discovery.color_scale", ["linear", "log10", "symlog"])
            color_scale = st.selectbox("伪色 colorbar 尺度", ["linear", "log10", "symlog"], index=index, disabled=not continuous,
                                       format_func=lambda value: {"linear": "线性", "log10": "对数（log10）", "symlog": "对称对数（含负值）"}[value], help=scale_help)
            linthresh = st.number_input("对称对数线性区阈值", min_value=1e-12, value=1.0, format="%.6g") if continuous and color_scale == "symlog" else 1.0
            _plot_embedding(
                st,
                data,
                embedding,
                color_mode,
                color_key,
                feature,
                cmap,
                categorical_palette,
                color_scale=color_scale,
                linthresh=linthresh,
            )


def _bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(adjusted, 0.0, 1.0)


def _var_labels(data, feature_ids, key):
    identifiers = pd.Index(feature_ids).astype(str)
    if key == "feature_id":
        return identifiers.to_numpy()
    values = data.var.reindex(identifiers)[key]
    return np.asarray([str(identifier) if pd.isna(value) or not str(value).strip() else str(value)
                       for identifier, value in zip(identifiers, values)])


def _volcano_selection_changed(st, plot_key, feature_ids, visible, mode):
    from scMM.plot.volcano import selected_features
    state = st.session_state.volcano_annotation_state
    chosen = selected_features(st.session_state.get(plot_key, {}), feature_ids)
    previous = state["selection"]
    added = [feature for feature in chosen if feature not in previous]
    removed = [feature for feature in previous if feature not in chosen]
    state["selection"] = chosen
    if mode == "仅自动":
        return
    for feature in added or removed:
        state["overrides"].pop(feature, None)
        state["overrides"][feature] = feature not in visible


def _render_differential_plots(st, data):
    import plotly.express as px
    from scipy import stats

    result = st.session_state.differential_result
    context = st.session_state.differential_context
    if result is None or result.empty or not context:
        return
    st.divider()
    st.subheader("差异代谢物可视化")
    a, b, c = st.columns(3)
    fdr_default, fdr_help = _parameter("analysis.diff_fdr", 0.05)
    fold_default, fold_help = _parameter("analysis.diff_log2fc", 1.0)
    labels_default, labels_help = _parameter("analysis.diff_label_count", 10)
    fdr_limit = a.number_input("FDR 阈值", 0.0, 1.0, float(fdr_default), help=fdr_help)
    fold_limit = b.number_input("|log2 FC| 阈值", 0.0, 100.0, float(fold_default), help=fold_help)
    label_count = c.number_input("最大标注数量", 0, 500, int(labels_default), help=labels_help, key="volcano_label_count")
    label_options = ["feature_id", *data.var.columns.astype(str)]
    index, help_text = _choice("analysis.volcano_var_label", label_options)
    label_key = st.selectbox("火山图文字使用的 var 标签", label_options, index=index, help=help_text, key="volcano_var_label")
    a, b, c = st.columns(3)
    label_size, label_help = _parameter("analysis.volcano_label_size", 10.)
    label_size = a.number_input("标注字号（pt）", 6., 48., float(label_size), help=label_help, key="volcano_label_size")
    modes = ["自动＋手动", "仅手动", "仅自动"]
    index, help_text = _choice("analysis.volcano_label_mode", modes)
    label_mode = b.selectbox("标注方式", modes, index=index, help=help_text, key="volcano_label_mode")
    scope = (id(data), id(result))
    state = st.session_state.get("volcano_annotation_state", {})
    if state.get("scope") != scope:
        state = dict(scope=scope, overrides={}, selection=[])
        st.session_state.volcano_annotation_state = state
    c.button("重置手动标注", key="volcano_reset_labels", on_click=state["overrides"].clear)
    from scMM.analysis.statistics import differential_summary
    summary = differential_summary(result, fdr_limit=fdr_limit, log2fc_limit=fold_limit)
    summary.insert(1, "feature_label", _var_labels(data, summary["feature_id"], label_key))
    first_columns = ["feature_id", "feature_label", "mz", "direction", "regulation", "FC", "log2_fold_change", "FDR"]
    first_columns = [name for name in first_columns if name in summary]
    summary = summary[first_columns + [name for name in summary if name not in first_columns]]
    st.markdown("##### 非靶向差异分析明细")
    st.caption(f"以 {context['group_b']} 为参照：FC = {context['group_a']} / {context['group_b']}。"
               f"方向表示均值变化；差异判定要求 FDR ≤ {fdr_limit:g} 且 |log2 FC| ≥ {fold_limit:g}。"
               "样本不足或 FC 无法解释时显示“无法判定”。")
    st.dataframe(summary, hide_index=True, use_container_width=True, height=420,
                 column_config={"feature_id": "特征 ID", "feature_label": "特征标签", "direction": "变化方向", "regulation": "差异判定",
                                "FC": st.column_config.NumberColumn("FC (A/B)", format="%.4g"),
                                "FDR": st.column_config.NumberColumn("FDR", format="%.4g")})
    st.download_button("下载非靶向差异分析明细 CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                       "scmm_untargeted_differential.csv", "text/csv", key="untargeted_summary_csv")
    current = st.session_state.result_table
    if current is not None and "log2_fold_change" in current and "cluster" not in current:
        st.session_state.result_table = summary
    volcano = result.copy()
    significant = summary["regulation"].isin(["上调", "下调"])
    volcano["label"] = ""
    from scMM.plot.volcano import volcano_figure, labelled_features
    automatic = volcano.loc[significant].sort_values("fdr")["feature_id"].astype(str).tolist()
    visible = labelled_features(automatic, state["overrides"], int(label_count), label_mode)
    labelled = volcano["feature_id"].astype(str).isin(visible)
    volcano.loc[labelled, "label"] = _var_labels(data, volcano.loc[labelled, "feature_id"], label_key)
    figure = volcano_figure(volcano, volcano["label"], group_a=context["group_a"], group_b=context["group_b"], annotation_size=label_size)
    figure.update_layout(clickmode="event+select", dragmode="pan", uirevision=str(scope))
    figure.update_traces(selectedpoints=[], selected=dict(marker=dict(opacity=.75)), unselected=dict(marker=dict(opacity=.75)))
    from scMM.plot.interaction import constrain_zoom
    constrain_zoom(figure)
    plot_key = f"volcano_selection_{id(data)}_{id(result)}"
    plotted_ids = [str(row[0]) for row in figure.data[0].customdata]
    plot_signature = (tuple(zip(volcano.loc[labelled, "feature_id"].astype(str), volcano.loc[labelled, "label"])), label_size)
    if state.get("plot_signature") != plot_signature:
        state["selection"] = []
        state["plot_signature"] = plot_signature
    st.plotly_chart(figure, use_container_width=True, key=plot_key, selection_mode="points",
                    on_select=lambda: _volcano_selection_changed(st, plot_key, plotted_ids, visible, label_mode))
    _plotly_with_export(st, figure, "differential_volcano", render=False)
    st.caption("直接点击数据点切换标注，可连续点选多个点；Shift＋点击可调整选点。手动标注优先，超过上限时保留最近选择的标注。重绘与导出使用当前标注。")

    options = result["feature_id"].astype(str).tolist()
    selection_default, selection_help = _parameter("analysis.diff_features", [])
    selected_default = [value for value in selection_default if value in options] or options[: min(4, len(options))]
    selected = st.multiselect("用于分布图的代谢物", options, default=selected_default, help=selection_help)
    a, b, c, d = st.columns(4)
    plot_index, plot_help = _choice("analysis.diff_plot_type", ["小提琴图", "箱线图"])
    plot_type = a.selectbox("分布图类型", ["小提琴图", "箱线图"], index=plot_index, help=plot_help)
    points_default, points_help = _parameter("analysis.diff_show_points", True)
    show_points = b.checkbox("显示散点", bool(points_default), help=points_help)
    zeros_default, zeros_help = _parameter("analysis.diff_remove_zeros", False)
    remove_zeros = c.checkbox("删除零值", bool(zeros_default), help=zeros_help)
    test_index, test_help = _choice("analysis.diff_plot_test", ["mannwhitney", "welch"])
    plot_test = d.selectbox("图中统计检验", ["mannwhitney", "welch"], index=test_index, help=test_help)
    if not selected:
        return
    groups = data.obs[context["group_key"]].astype(str)
    keep = groups.isin([str(context["group_a"]), str(context["group_b"])]).to_numpy()
    if hasattr(data, "cell_keep_mask"):
        keep &= data.cell_keep_mask()
    indices = [data.var_names.get_loc(feature) for feature in selected]
    values = _dense(data.X)[keep][:, indices]
    wide = pd.DataFrame(values, columns=selected)
    wide["group"] = groups.to_numpy()[keep]
    long = wide.melt(id_vars="group", var_name="feature", value_name="intensity").dropna()
    if remove_zeros:
        long = long[long["intensity"] != 0]
    test_rows = []
    for feature in selected:
        left = long.loc[(long["feature"] == feature) & (long["group"] == str(context["group_a"])), "intensity"]
        right = long.loc[(long["feature"] == feature) & (long["group"] == str(context["group_b"])), "intensity"]
        if len(left) and len(right):
            if plot_test == "mannwhitney":
                statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
            else:
                statistic, p_value = stats.ttest_ind(left, right, equal_var=False, nan_policy="omit")
        else:
            statistic, p_value = np.nan, np.nan
        test_rows.append({"feature": feature, "n_A": len(left), "n_B": len(right), "statistic": statistic, "p_value": p_value})
    test_frame = pd.DataFrame(test_rows)
    finite = test_frame["p_value"].notna()
    test_frame["fdr"] = np.nan
    if finite.any():
        test_frame.loc[finite, "fdr"] = _bh_adjust(test_frame.loc[finite, "p_value"])
    points = "all" if show_points else False
    if plot_type == "小提琴图":
        distribution = px.violin(long, x="group", y="intensity", color="group", facet_col="feature", facet_col_wrap=3, box=True, points=points, color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        distribution = px.box(long, x="group", y="intensity", color="group", facet_col="feature", facet_col_wrap=3, points=points, color_discrete_sequence=px.colors.qualitative.Plotly)
    _match_distribution_colors(distribution)
    distribution.update_yaxes(matches=None)
    distribution.update_layout(height=max(460, 300 * ((len(selected) + 2) // 3)), showlegend=False)
    _plotly_with_export(st, distribution, "differential_distribution")
    st.caption("统计检验针对当前图中数据重新计算；选择“删除零值”时样本量会随特征变化。")
    st.dataframe(test_frame, use_container_width=True, hide_index=True)


def _file_label_tool(st, data):
    if "source_file" not in data.obs:
        st.info("文件打标需要 obs['source_file']。")
        return
    with st.expander("按质谱文件赋予实验标签", expanded=False):
        column = st.text_input("写入 obs 的列名", value="experimental_group", key="file_label_column")
        files = sorted(data.obs["source_file"].dropna().astype(str).unique())
        rows = []
        for source in files:
            values = data.obs.loc[data.obs["source_file"].astype(str).eq(source), column].dropna().astype(str).unique() if column in data.obs else []
            rows.append({"source_file": source, "label": values[0] if len(values) == 1 else ""})
        labels = st.data_editor(pd.DataFrame(rows), disabled=["source_file"], hide_index=True,
                                use_container_width=True, key=f"file_labels_{column}")
        if st.button("将文件标签写入对应细胞", key="apply_file_labels"):
            if not column.strip() or column in {"source_file", "frame_id"} or column.startswith("cell_keep_"):
                st.error("请填写实验标签列名，例如 group、diagnosis 或 treatment。")
            else:
                mapping = labels.set_index("source_file")["label"].replace("", pd.NA).dropna().to_dict()
                mapped = data.obs["source_file"].astype(str).map(mapping)
                existing = data.obs[column].astype(object) if column in data.obs else pd.Series(pd.NA, index=data.obs_names)
                data.obs[column] = mapped.where(mapped.notna(), existing)
                for source, value in mapping.items():
                    data.uns.setdefault("files", {}).setdefault(source, {})[column] = str(value)
                st.success(f"已写入 obs[{column!r}]，对应 {int(mapped.notna().sum())} 个细胞。")


def _match_distribution_colors(figure):
    for trace in figure.data:
        if trace.type in {"violin", "box"}:
            color = trace.line.color or trace.marker.color
            if isinstance(color, str):
                trace.line.color = color
                trace.marker.color = color
                trace.fillcolor = color
    return figure


def _targeted_analysis(st, data, group_key):
    import plotly.express as px
    from scMM.analysis.statistics import compare_groups

    available = data.var_names[data.feature_keep_mask()].astype(str).tolist()
    selected = st.multiselect("靶向特征", available, default=available[:min(3, len(available))], key="target_features")
    group_values = sorted(data.obs[group_key].dropna().astype(str).unique())
    groups = st.multiselect("展示的实验组", group_values, default=group_values, key="target_groups")
    a, b, c = st.columns(3)
    include_zeros = a.checkbox("靶向分析使用零值", value=True)
    show_points = b.checkbox("靶向图显示散点", value=True)
    method = c.selectbox("靶向统计检验", ["ANOVA", "Kruskal-Wallis", "Welch", "Mann-Whitney"])
    pairwise = method in {"Welch", "Mann-Whitney"}
    if pairwise:
        pair = st.multiselect("两组比较（选择两个实验组）", groups, default=groups[:2], max_selections=2, key="target_pair")
    else:
        pair = groups
    if not selected or len(groups) < 2:
        st.info("请选择特征及至少两个实验组。")
        return
    keep = data.cell_keep_mask() & data.obs[group_key].notna().to_numpy() & data.obs[group_key].astype(str).isin(groups).to_numpy()
    positions = data.var_names.get_indexer(selected)
    matrix = _dense(data.X[np.flatnonzero(keep)][:, positions])
    frame = pd.DataFrame(matrix, columns=selected)
    frame["group"] = data.obs.loc[keep, group_key].astype(str).to_numpy()
    long = frame.melt(id_vars="group", var_name="feature", value_name="intensity").dropna()
    if not include_zeros:
        long = long[long.intensity != 0]

    st.markdown("##### 两分布比较")
    pending = st.session_state.pop("target_pending_pairs", None)
    if pending:
        for suffix, (feature, group) in zip(["a", "b"], pending):
            st.session_state[f"target_compare_feature_{suffix}"] = feature
            st.session_state[f"target_compare_group_{suffix}"] = group
    defaults = {
        "target_compare_feature_a": (selected, selected[0]),
        "target_compare_feature_b": (selected, selected[0]),
        "target_compare_group_a": (groups, groups[0]),
        "target_compare_group_b": (groups, groups[1]),
    }
    for key, (options, fallback) in defaults.items():
        if st.session_state.get(key) not in options:
            st.session_state[key] = fallback
    a, b = st.columns(2)
    feature_a = a.selectbox("A：特征", selected, key="target_compare_feature_a")
    group_a = a.selectbox("A：实验组", groups, key="target_compare_group_a")
    feature_b = b.selectbox("B：特征", selected, key="target_compare_feature_b")
    group_b = b.selectbox("B：实验组", groups, key="target_compare_group_b")
    compare_method = st.selectbox("两分布检验方法", ["Welch", "Mann-Whitney"], key="target_compare_method")
    signature = (feature_a, group_a, feature_b, group_b, compare_method, include_zeros, group_key)
    identical = (feature_a, group_a) == (feature_b, group_b)
    if st.button("运行两分布比较", key="target_compare_run", disabled=identical):
        st.session_state.target_comparison = signature
    if identical:
        st.caption("请为 A、B 选择不同的特征或实验组。")
    comparison = None
    if st.session_state.get("target_comparison") == signature and not identical:
        sample_a = long.loc[(long.feature == feature_a) & (long.group == group_a), "intensity"]
        sample_b = long.loc[(long.feature == feature_b) & (long.group == group_b), "intensity"]
        statistic, pvalue = compare_groups([sample_a, sample_b], compare_method)
        comparison = dict(feature_a=feature_a, group_a=group_a, feature_b=feature_b, group_b=group_b,
                          n_a=len(sample_a), n_b=len(sample_b), test=compare_method, statistic=statistic, p_value=pvalue)
        if np.isfinite(pvalue):
            st.success(f"A: {feature_a} / {group_a} (n={len(sample_a)})；B: {feature_b} / {group_b} (n={len(sample_b)})；{compare_method} p={pvalue:.4g}")
        else:
            st.info("当前两个分布的有效数据不足以计算 p 值，请检查组别、零值设置和样本数。")
    rows = []
    for feature in selected:
        samples = [long.loc[(long.feature == feature) & (long.group == group), "intensity"] for group in pair]
        statistic, pvalue = compare_groups(samples, method) if len(pair) >= 2 else (np.nan, np.nan)
        rows.append({"feature": feature, "test": method, "groups": " / ".join(pair), "statistic": statistic, "p_value": pvalue})
    result = pd.DataFrame(rows)
    result["fdr"] = np.nan
    finite = result.p_value.notna()
    if finite.any():
        result.loc[finite, "fdr"] = _bh_adjust(result.loc[finite, "p_value"])
    figure = px.violin(long, x="group", y="intensity", color="group", facet_col="feature", facet_col_wrap=3,
                       box=True, points=False,
                       color_discrete_sequence=px.colors.qualitative.Plotly,
                       category_orders={"group": groups, "feature": selected}, custom_data=["feature", "group"])
    _match_distribution_colors(figure)
    if show_points:
        for trace in list(figure.data):
            figure.add_scatter(x=trace.x, y=trace.y, xaxis=trace.xaxis, yaxis=trace.yaxis,
                               mode="markers", marker=dict(size=4, opacity=.65, color=trace.line.color),
                               line=dict(color=trace.line.color), legendgroup=trace.legendgroup,
                               customdata=trace.customdata, name=trace.name, showlegend=False,
                               hovertemplate="%{customdata[0]} · %{customdata[1]}<br>%{y}<extra></extra>")
    figure.update_yaxes(matches=None)
    for feature, pvalue in zip(selected, result.p_value):
        traces = [trace for trace in figure.data if trace.customdata is not None and len(trace.customdata) and trace.customdata[0][0] == feature]
        if traces:
            trace = traces[0]
            figure.add_annotation(x=.5, y=1.02, xref=f"{trace.xaxis or 'x'} domain", yref=f"{trace.yaxis or 'y'} domain",
                                  text=f"{method}: p={pvalue:.3g}", showarrow=False)
    if comparison is not None:
        for feature in dict.fromkeys([feature_a, feature_b]):
            trace = next((trace for trace in figure.data if trace.customdata is not None and len(trace.customdata) and trace.customdata[0][0] == feature), None)
            if trace is not None:
                figure.add_annotation(x=.5, y=.94, xref=f"{trace.xaxis or 'x'} domain", yref=f"{trace.yaxis or 'y'} domain",
                                      text=f"{compare_method}: A vs B, p={comparison['p_value']:.3g}", showarrow=False,
                                      bgcolor="rgba(255,255,255,.8)")
    figure.update_layout(title="靶向特征分组比较", height=max(520, 350 * ((len(selected) + 2) // 3)), showlegend=True)
    # Point selection identifies feature/group combinations directly on the plot.
    from scMM.plot.interaction import constrain_zoom
    constrain_zoom(figure)
    event = st.plotly_chart(figure, key="targeted_selection", use_container_width=True,
                            on_select="rerun", selection_mode=("points", "box", "lasso"))
    _plotly_with_export(st, figure, "targeted_distribution", render=False)
    st.dataframe(result, hide_index=True, use_container_width=True)
    if comparison is not None:
        st.dataframe(pd.DataFrame([comparison]), hide_index=True, use_container_width=True)
    state = event.get("selection", {}) if hasattr(event, "get") else {}
    chosen = []
    for point in state.get("points", []):
        custom = point.get("customdata", [])
        if hasattr(custom, "get"):
            values = list(custom.values())
            feature = custom.get("feature", custom.get("0", values[0] if values else None))
            group = custom.get("group", custom.get("1", values[1] if len(values) > 1 else None))
        else:
            feature, group = custom[:2] if len(custom) >= 2 else (None, None)
        if feature in selected and group in groups and (feature, group) not in chosen:
            chosen.append((feature, group))
    if chosen:
        st.caption("图中已选分布：" + "；".join(f"{feature} / {group}" for feature, group in chosen))
    if st.button("将图选的两个分布填入比较栏", disabled=len(chosen) != 2, key="target_apply_selection"):
        st.session_state.target_pending_pairs = chosen
        st.rerun()
    st.caption(
        "使用方法：① 在上方选择要展示的特征和实验组。② 在“两分布比较”中分别指定 A、B 的特征与实验组，"
        "选择 Welch 或 Mann–Whitney 后点击“运行两分布比较”。同一特征配两个实验组可比较组间差异；"
        "两个特征配相同实验组可比较该组中两个特征的强度。③ 图选方式：打开散点，按住 Shift 连续点选两个分布中的"
        "任意细胞点，或使用框选/套索圈出两个分布，再点击“将图选的两个分布填入比较栏”，随后运行比较。"
        "检验使用所选分布的全部有效细胞；零值是否纳入由“靶向分析使用零值”决定。跨特征比较请使用可比较的强度尺度。"
    )


def _heatmap_ui(st, data, kind):
    from scMM.analysis.heatmaps import expression_matrix, heatmap_figure

    titles = {"differential": "差异特征表达热图", "correlation": "特征相关性热图", "file_qc": "测量文件 pseudobulk 质控热图"}
    with st.expander(titles[kind], expanded=False):
        available = data.var_names[data.feature_keep_mask()].astype(str).tolist()
        defaults = available[:min(30, len(available))]
        if kind == "differential":
            differential = st.session_state.differential_result
            fdr_limit = st.number_input("热图差异特征 FDR 上限", 0.0, 1.0, .05, key=f"{kind}_fdr")
            if differential is not None and not differential.empty:
                defaults = [name for name in differential.loc[differential.fdr <= fdr_limit, "feature_id"].astype(str) if name in available][:50]
            else:
                defaults = [name for name in st.session_state.get("target_features", []) if name in available]
            st.caption("默认选择当前差异分析中满足 FDR 阈值的特征；也可手动选择靶向特征。")
        features = st.multiselect("热图特征", available, default=defaults, key=f"{kind}_features")
        a, b, c = st.columns(3)
        sample_mode = a.selectbox("样本单位", ["单个细胞", "质谱文件 pseudobulk"], index=1 if kind == "file_qc" else 0,
                                  disabled=kind == "file_qc", key=f"{kind}_sample_mode")
        value, help_text = _parameter("heatmap.cell_limit", 500)
        cell_limit = st.number_input("最多展示细胞数（0=全部）", 0, 1000000, int(value), step=100, help=help_text, key=f"{kind}_cell_limit") if kind == "differential" and sample_mode == "单个细胞" else 0
        index, help_text = _choice("heatmap.aggregate", ["mean", "sum", "median"])
        aggregate = b.selectbox("文件聚合", ["mean", "sum", "median"], index=index, help=help_text, key=f"{kind}_aggregate")
        cmap_options = ["Viridis", "RdBu", "Spectral", "Turbo", "Plasma", "Cividis", "Inferno"]
        index, help_text = _choice("heatmap.cmap", cmap_options)
        cmap = c.selectbox("热图 cmap", cmap_options, index=index, help=help_text, key=f"{kind}_cmap")
        a, b = st.columns(2)
        obs_key = a.selectbox("样本/组别文字使用的 obs", ["cell/file ID", *data.obs.columns.astype(str)], key=f"{kind}_obs_label")
        var_key = b.selectbox("特征文字使用的 var", ["feature_id", *data.var.columns.astype(str)], key=f"{kind}_var_label")
        show_features = a.checkbox("显示特征文字", value=True, key=f"{kind}_show_features")
        show_samples = b.checkbox("显示样本/组别文字", value=kind == "file_qc", key=f"{kind}_show_samples")
        cluster_rows = a.checkbox("行层次聚类", value=True, key=f"{kind}_cluster_rows")
        cluster_columns = b.checkbox("列层次聚类", value=True, key=f"{kind}_cluster_columns")
        methods = ["average", "complete", "single", "ward"]
        index, help_text = _choice("heatmap.linkage", methods)
        method = a.selectbox("Linkage 方法", methods, index=index, help=help_text, key=f"{kind}_linkage")
        metrics = ["euclidean", "correlation", "cosine"]
        index, help_text = _choice("heatmap.distance", metrics)
        metric = b.selectbox("聚类距离", metrics, index=index, help=help_text, disabled=method == "ward", key=f"{kind}_distance")
        transforms = ["none", "log1p", "feature z-score"]
        index, help_text = _choice("heatmap.transform", transforms)
        transform = a.selectbox("表达值变换", transforms, index=index, help=help_text, key=f"{kind}_transform")
        correlation_method = b.selectbox("相关系数", ["pearson", "spearman", "kendall"], disabled=kind != "correlation", key=f"{kind}_correlation")
        annotate = st.checkbox("在格内显示 R²" if kind == "file_qc" else "在格内显示数值", value=kind == "file_qc", key=f"{kind}_annotate")
        if kind == "file_qc":
            st.caption("按来源文件聚合有效细胞，用聚合后的特征强度向量计算文件间 Pearson r，并显示 R²=r²。不同实验标签混在同一文件时，标签显示 mixed。")
        if st.button("绘制热图", key=f"{kind}_run"):
            try:
                if not features:
                    raise ValueError("请至少选择一个特征。")
                matrix, sample_labels = expression_matrix(data, features, pseudobulk=sample_mode != "单个细胞" or kind == "file_qc",
                                                          aggregate=aggregate, obs_key=None if obs_key == "cell/file ID" else obs_key)
                if cell_limit and len(matrix) > cell_limit:
                    indices = np.sort(np.random.default_rng(42).choice(len(matrix), size=cell_limit, replace=False))
                    matrix = matrix.iloc[indices]
                    sample_labels = sample_labels.reindex(matrix.index)
                    st.caption(f"热图使用固定种子抽取的 {cell_limit} 个有效细胞；完整数据保持原样。")
                if transform == "log1p":
                    if (matrix.to_numpy() < 0).any():
                        raise ValueError("log1p 表达热图使用非负强度，请选择 none 或 feature z-score。")
                    matrix = np.log1p(matrix)
                elif transform == "feature z-score":
                    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
                feature_labels = _var_labels(data, matrix.columns, var_key)
                if kind == "differential":
                    values = matrix.T
                    row_labels, column_labels = feature_labels, sample_labels.to_numpy()
                    show_rows, show_columns = show_features, show_samples
                    value_range = None
                elif kind == "correlation":
                    values = matrix.corr(method=correlation_method)
                    row_labels = column_labels = feature_labels
                    show_rows = show_columns = show_features
                    value_range = (-1, 1)
                else:
                    values = matrix.T.corr(method="pearson") ** 2
                    row_labels = column_labels = sample_labels.reindex(values.index).to_numpy()
                    show_rows = show_columns = show_samples
                    value_range = (0, 1)
                figure = heatmap_figure(values, title=titles[kind] + (" · R²" if kind == "file_qc" else ""), row_labels=row_labels, column_labels=column_labels,
                                        show_rows=show_rows, show_columns=show_columns,
                                        cluster_rows=cluster_rows, cluster_columns=cluster_columns,
                                        method=method, metric="euclidean" if method == "ward" else metric,
                                        cmap=cmap, annotate=annotate, value_range=value_range)
                figure.update_layout(meta={"scmm_cmap": cmap})
                st.session_state[f"{kind}_heatmap_result"] = (figure, values)
            except ValueError as exc:
                st.error(str(exc))
        result = st.session_state.get(f"{kind}_heatmap_result")
        if result is not None:
            figure, values = result
            _plotly_with_export(st, figure, f"{kind}_heatmap")
            st.download_button("下载热图数值 CSV", values.to_csv().encode("utf-8-sig"), f"{kind}_heatmap.csv", key=f"{kind}_csv")


def _network_plot(st, data):
    import networkx as nx
    import plotly.graph_objects as go

    network = st.session_state.network
    if network is None:
        return
    graph, _ = network
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    ex, ey = [], []
    for left, right in graph.edges:
        ex.extend([positions[left][0], positions[right][0], None])
        ey.extend([positions[left][1], positions[right][1], None])
    nodes = list(graph.nodes)
    a, b = st.columns(2)
    color_options = ["节点度数", *data.var.columns.astype(str)]
    index, help_text = _choice("analysis.network_var_color", color_options)
    color_key = a.selectbox("网络节点颜色", color_options, index=index, help=help_text, key="network_var_color")
    cmap_options = ["Viridis", "Turbo", "Plasma", "RdBu", "Spectral"]
    index, help_text = _choice("analysis.network_cmap", cmap_options)
    cmap = b.selectbox("网络 cmap", cmap_options, index=index, help=help_text, key="network_cmap")
    values = pd.Series([graph.degree(node) for node in nodes], index=nodes) if color_key == "节点度数" else data.var.reindex(nodes)[color_key]
    figure = go.Figure([
        go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=.7, color="#999"), hoverinfo="skip"),
    ])
    if pd.api.types.is_numeric_dtype(values):
        figure.add_scatter(x=[positions[node][0] for node in nodes], y=[positions[node][1] for node in nodes],
                           mode="markers", text=nodes, name=color_key,
                           marker=dict(size=8, color=values.to_numpy(dtype=float), colorscale=cmap, showscale=True, colorbar=dict(title=color_key)))
    else:
        import plotly.express as px
        categories = values.astype(object).where(values.notna(), "missing").astype(str)
        for index, category in enumerate(categories.unique()):
            selected_nodes = categories.index[categories == category]
            color = px.colors.qualitative.Plotly[index % len(px.colors.qualitative.Plotly)]
            figure.add_scatter(x=[positions[node][0] for node in selected_nodes], y=[positions[node][1] for node in selected_nodes],
                               mode="markers", text=list(selected_nodes), name=category,
                               marker=dict(size=8, color=color), line=dict(color=color))
    figure.update_layout(title=f"Feature correlation network · {color_key}", height=600, showlegend=not pd.api.types.is_numeric_dtype(values), meta={"scmm_cmap": cmap})
    _plotly_with_export(st, figure, "feature_network")


def _association_ui(st, data):
    import plotly.graph_objects as go
    from scMM.analysis.statistics import numeric_association

    st.subheader("两变量相关性：特征 / 数值型 obs")
    active_features = set(data.var_names[data.feature_keep_mask()].astype(str))
    coordinates = {name: descriptor for name, descriptor in _gate_coordinates(data).items()
                   if descriptor[0] == "obs" or (descriptor[0] == "feature" and descriptor[1] in active_features)}
    options = list(coordinates)
    if len(options) < 2:
        return
    a, b, c = st.columns(3)
    xname = a.selectbox("相关性 X", options)
    yname = b.selectbox("相关性 Y", options, index=1)
    permutations = c.number_input("Permutation 次数", 99, 99999, 999, step=100)
    include_zeros = a.checkbox("相关性使用零值", value=True)
    confidence = b.number_input("拟合均值置信水平", .8, .999, .95)
    seed = c.number_input("相关性随机种子", 0, 2147483647, 42)
    if st.button("计算相关性与置换检验"):
        x, y = _gate_values(data, coordinates[xname]), _gate_values(data, coordinates[yname])
        mask = data.cell_keep_mask()
        if not include_zeros:
            mask &= (x != 0) & (y != 0)
        try:
            result = numeric_association(x[mask], y[mask], permutations=permutations, confidence=confidence, seed=seed)
            st.session_state.association_result = (xname, yname, result)
        except ValueError as exc:
            st.error(str(exc))
    stored = st.session_state.get("association_result")
    if stored is None:
        return
    x_label, y_label, result = stored
    line, points = result["line"], result["points"]
    figure = go.Figure()
    figure.add_scatter(x=line.x, y=line.lower, mode="lines", line=dict(width=0), name="CI lower", showlegend=False)
    figure.add_scatter(x=line.x, y=line.upper, mode="lines", fill="tonexty", line=dict(width=0), name=f"{result['confidence']:.0%} CI", fillcolor="rgba(37,99,235,.18)")
    figure.add_scatter(x=points.x, y=points.y, mode="markers", name="cells", marker=dict(size=5, opacity=.6))
    figure.add_scatter(x=line.x, y=line.fit, mode="lines", name="OLS fit")
    figure.update_layout(title=f"R²={result['r2']:.4f}; permutation p={result['permutation_p']:.4g}; n={result['n']}",
                         xaxis_title=x_label, yaxis_title=y_label, height=600)
    _plotly_with_export(st, figure, "numeric_association")
    st.caption(f"Pearson r 双侧置换检验，{result['permutations']} 次置换；阴影为 OLS 拟合均值的置信区间。")


def _render_shap_plots(st, analyzer):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if not hasattr(analyzer, "shap_explanation_"):
        return
    st.subheader("SHAP 可视化")
    outputs = analyzer.shap_output_names()
    a, b = st.columns(2)
    output_index = a.selectbox("SHAP 解释类别", range(len(outputs)),
                                index=1 if len(outputs) == 2 else 0,
                                format_func=lambda index: outputs[index], key="shap_output")
    kinds = {"特征重要性条形图": "bar", "蜂群图": "beeswarm", "特征依赖图": "dependence"}
    index, help_text = _choice("supervised.shap_plot", list(kinds))
    plot_label = b.selectbox("SHAP 图形", list(kinds), index=index, help=help_text, key="shap_plot_kind")
    value, help_text = _parameter("supervised.shap_top", 20)
    top = st.number_input("SHAP Top N", 1, len(analyzer.feature_names), min(int(value), len(analyzer.feature_names)), help=help_text)
    feature = color_feature = None
    if kinds[plot_label] == "dependence":
        a, b = st.columns(2)
        feature = a.selectbox("依赖图特征", analyzer.feature_names, key="shap_dependence_feature")
        color = b.selectbox("依赖图颜色特征", ["同一特征", *analyzer.feature_names], key="shap_dependence_color")
        color_feature = feature if color == "同一特征" else color
    with st.expander("SHAP 绘图与导出参数"):
        a, b, c = st.columns(3)
        width = a.number_input("SHAP 宽度（inch）", 3.0, 30.0, 9.0, .5)
        height = b.number_input("SHAP 高度（inch）", 3.0, 30.0, max(5., min(15., top * .3 + 2)), .5)
        dpi = c.number_input("SHAP DPI", 72, 600, 200)
        point_size = a.number_input("SHAP 点面积（pt²）", .1, 500., 16., 1.)
        alpha = b.number_input("SHAP 透明度", .05, 1., .8, .05)
        font_size = c.number_input("SHAP 字号", 6, 30, 10)
        cmap = a.selectbox("SHAP cmap", ["coolwarm", "viridis", "plasma", "RdBu_r", "Spectral"])
    fig = None
    try:
        fig = analyzer.shap_figure(plot_type=kinds[plot_label], class_index=output_index, max_display=top,
                                   feature=feature, color_feature=color_feature, width=width, height=height,
                                   point_size=point_size, alpha=alpha, cmap=cmap, font_size=font_size)
        st.pyplot(fig, use_container_width=True)
        a, b = st.columns(2)
        with mpl.rc_context({"svg.fonttype": "none"}):
            for target, file_format in [(a, "svg"), (b, "png")]:
                buffer = io.BytesIO()
                fig.savefig(buffer, format=file_format, dpi=dpi, bbox_inches="tight")
                if target.button(f"下载 SHAP {file_format.upper()}", key=f"shap_download_{file_format}"):
                    description = f"{plot_label}_{analyzer.model_name_}_{outputs[output_index]}_{feature or 'Top'+str(top)}"
                    _save_image(st, buffer.getvalue(), _image_filename("SHAP", file_format, title=description))
        values = np.asarray(analyzer.shap_explanation_.values)
        selected_values = values[..., output_index] if values.ndim == 3 else values
        table = pd.DataFrame({"feature": analyzer.feature_names, "mean_abs_shap": np.mean(np.abs(selected_values), axis=0)}).sort_values("mean_abs_shap", ascending=False)
        st.dataframe(table.head(int(top)), hide_index=True, use_container_width=True)
        st.caption(
            f"解释输出：{outputs[output_index]}；使用已计算的 {len(values)} 个测试集细胞。"
            "条形图展示平均绝对 SHAP 值；蜂群图展示影响方向和特征值；依赖图展示特征值与 SHAP 值的关系。"
            "横轴/颜色中的特征值采用解释器保存的模型输入尺度，含标准化步骤的模型可能显示标准化值。SVG 保留文字对象。"
        )
    except Exception as exc:
        st.error(f"SHAP 绘图失败：{exc}")
    finally:
        if fig is not None:
            plt.close(fig)


def _render_supervised_plots(st, analyzer):
    import plotly.express as px
    import plotly.graph_objects as go

    result = analyzer.result_
    qc = result["model_qc"]
    columns = st.columns(4)
    for column, (name, value) in zip(columns, [("R²", qc["r2"]), ("Q²", qc["q2"]), ("PR-AUC", qc["pr_auc_macro"]), ("Brier", qc["brier_macro"])]):
        column.metric(name, f"{value:.3f}")
    st.caption(qc["definition"])

    roc = go.Figure()
    for name, values in analyzer.roc_curves().items():
        roc.add_scatter(x=values["fpr"], y=values["tpr"], mode="lines", name=f"{name} (AUC={values['auc']:.3f})")
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="random")
    roc.update_layout(title="ROC", xaxis_title="False positive rate", yaxis_title="True positive rate", height=500)
    _plotly_with_export(st, roc, "supervised_roc")

    if str(analyzer.model_name_).lower() in {"lda", "plsda", "pls-da"}:
        key = "X_lda" if str(analyzer.model_name_).lower() == "lda" else "X_plsda"
        scores = np.asarray(analyzer.data.obsm[key])[analyzer.valid_mask]
        latent = pd.DataFrame({"component 1": scores[:, 0], "component 2": scores[:, 1], "class": analyzer.y, "cell": analyzer.obs_names.astype(str)})
        figure = px.scatter(latent, x="component 1", y="component 2", color="class", hover_name="cell", title=f"{analyzer.model_name_.upper()} 潜变量得分图")
        figure.update_layout(height=540)
        _plotly_with_export(st, figure, "supervised_latent")

    left, right = st.columns(2)
    pr_figure = go.Figure()
    for name, values in analyzer.precision_recall_curves().items():
        pr_figure.add_scatter(x=values["recall"], y=values["precision"], mode="lines", name=f"{name} (AP={values['average_precision']:.3f})")
    pr_figure.update_layout(title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision", height=470)
    _plotly_with_export(st, pr_figure, "precision_recall", container=left)
    calibration_figure = go.Figure()
    calibration_figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="ideal")
    for name, values in analyzer.calibration_curves().items():
        calibration_figure.add_scatter(x=values["mean_predicted_probability"], y=values["fraction_positive"], mode="lines+markers", name=name)
    calibration_figure.update_layout(title="校正曲线", xaxis_title="Mean predicted probability", yaxis_title="Fraction positive", height=470)
    _plotly_with_export(st, calibration_figure, "calibration", container=right)
    confusion = px.imshow(result["confusion_matrix"], x=analyzer.classes_.astype(str), y=analyzer.classes_.astype(str), text_auto=True, labels={"x": "Predicted", "y": "True"}, title="混淆矩阵")
    confusion.update_layout(height=470)
    _plotly_with_export(st, confusion, "confusion_matrix")


def page_analysis():
    import streamlit as st

    from scMM.analysis import SupervisedAnalyzer

    st.header("统计分析与监督学习")
    _pipeline_status(st)
    terminal = _terminal(st)
    data = st.session_state.data
    if data is None:
        st.warning("请先加载数据。")
        return
    _file_label_tool(st, data)
    stats_tab, network_tab, supervised_tab = st.tabs(["差异与 Marker", "相关网络", "监督学习 / SHAP"])
    with stats_tab:
        group_options = data.obs.columns.astype(str).tolist()
        group_index, group_help = _choice("analysis.group_key", group_options)
        group_key = st.selectbox("分组列", group_options, index=group_index, help=group_help)
        mode = st.radio("差异分析模式", ["非靶向差异分析", "靶向差异分析"], horizontal=True)
        include_zeros = st.checkbox("非靶向检验使用零值", value=True, disabled=mode == "靶向差异分析")
        values = sorted(data.obs[group_key].dropna().astype(str).unique())
        a, b = st.columns(2)
        configured, help_text = _parameter("analysis.group_a", ""); group_a = a.selectbox("Group A", values, index=values.index(configured) if configured in values else 0, help=help_text)
        configured, help_text = _parameter("analysis.group_b", ""); group_b = b.selectbox("Group B", values, index=values.index(configured) if configured in values else min(1, len(values) - 1), help=help_text)
        test_index, test_help = _choice("analysis.diff_test", ["mannwhitney", "welch"])
        test = st.selectbox("检验", ["mannwhitney", "welch"], index=test_index, help=test_help)
        if st.button("差异分析（BH-FDR）", type="primary", disabled=mode == "靶向差异分析"):
            result = _execute(st, terminal, "差异分析", lambda progress: (progress(0.1, "Testing features"), _engine(st).differential_features(group_key, group_a, group_b, method=test, include_zeros=include_zeros), progress(0.92, "Applying BH-FDR"))[1], "analysis")
            if result is not None:
                st.session_state.result_table = result
                st.session_state.differential_result = result
                st.session_state.differential_context = {"group_key": group_key, "group_a": group_a, "group_b": group_b, "test": test, "include_zeros": include_zeros}
                st.rerun()
        if st.button("聚类 Marker 分析", disabled=mode == "靶向差异分析"):
            result = _execute(st, terminal, "Marker 分析", lambda progress: (progress(0.1, "Comparing clusters"), _engine(st).marker_features(group_key, method=test, include_zeros=include_zeros), progress(0.92, "Combining marker tables"))[1], "analysis")
            if result is not None: st.session_state.result_table = result; st.rerun()
        if mode == "靶向差异分析":
            _targeted_analysis(st, data, group_key)
        else:
            _render_differential_plots(st, data)
        _heatmap_ui(st, data, "differential")
    with network_tab:
        a, b = st.columns(2)
        threshold_default, threshold_help = _parameter("analysis.correlation_threshold", 0.7)
        threshold = a.number_input("|相关系数|阈值", 0.0, 1.0, float(threshold_default), help=threshold_help)
        top_default, top_help = _bounded_default("analysis.top_features", min(100, data.n_vars), 2, min(5000, data.n_vars))
        top_features = b.number_input("Top variable features", 2, min(5000, data.n_vars), int(top_default), help=top_help)
        method_index, method_help = _choice("analysis.correlation_method", ["pearson", "spearman", "kendall"])
        method = st.selectbox("相关方法", ["pearson", "spearman", "kendall"], index=method_index, help=method_help)
        if st.button("构建相关网络", type="primary"):
            result = _execute(st, terminal, "相关网络", lambda progress: (progress(0.1, "Computing correlation matrix"), _engine(st).feature_correlation_network(method=method, min_abs_correlation=threshold, top_features=top_features), progress(0.92, "Building network"))[1], "analysis")
            if result is not None: st.session_state.network = result; st.session_state.result_table = result[1]; st.rerun()
        _network_plot(st, data)
        _heatmap_ui(st, data, "correlation")
        _association_ui(st, data)
    with supervised_tab:
        label_options = data.obs.columns.astype(str).tolist()
        label_index, label_help = _choice("supervised.label", label_options)
        label = st.selectbox("标签列", label_options, index=label_index, help=label_help)
        model_index, model_help = _choice("supervised.model", ["logistic", "random_forest", "lda", "plsda"])
        model = st.selectbox("模型", ["logistic", "random_forest", "lda", "plsda"], index=model_index, help=model_help)
        a, b, c = st.columns(3)
        cv_default, cv_help = _parameter("supervised.cv", 5)
        cv = a.number_input("CV folds", 2, 20, int(cv_default), help=cv_help)
        split_default, split_help = _parameter("supervised.test_size", 0.2)
        test_size = b.number_input("测试集比例", 0.05, 0.5, float(split_default), help=split_help)
        smote_default, smote_help = _parameter("supervised.smote", False)
        smote = c.checkbox("SMOTE", bool(smote_default), help=smote_help)
        tree_default, tree_help = _parameter("supervised.rf_trees", 500)
        trees = a.number_input("RF trees", 10, 5000, int(tree_default), help=tree_help)
        seed_default, seed_help = _parameter("supervised.seed", 42)
        seed = b.number_input("随机种子", 0, 2_147_483_647, int(seed_default), help=seed_help)
        bins_default, bins_help = _parameter("supervised.calibration_bins", 10)
        calibration_bins = c.number_input("校正曲线分箱数", 2, 50, int(bins_default), help=bins_help)
        model_params = {}
        if model == "plsda":
            pls_default, pls_help = _parameter("supervised.pls_components", 2)
            model_params["n_components"] = a.number_input("PLS components", 1, min(100, data.n_vars), min(int(pls_default), data.n_vars), help=pls_help)
        elif model == "lda":
            solver_index, solver_help = _choice("supervised.lda_solver", ["svd", "lsqr", "eigen"])
            solver = a.selectbox("LDA solver", ["svd", "lsqr", "eigen"], index=solver_index, help=solver_help)
            model_params["solver"] = solver
            if solver in {"lsqr", "eigen"}:
                shrink_index, shrink_help = _choice("supervised.lda_shrinkage", ["none", "auto"])
                shrinkage = b.selectbox("LDA shrinkage", ["none", "auto"], index=shrink_index, help=shrink_help)
                model_params["shrinkage"] = None if shrinkage == "none" else shrinkage
        if st.button("训练、交叉验证并生成质控", type="primary"):
            def train(progress):
                progress(0.05, "Preparing labels and matrix")
                analyzer = SupervisedAnalyzer(data, label, random_state=seed)
                params = dict(model_params)
                if model == "random_forest": params["n_estimators"] = trees
                progress(0.2, "Cross-validation and fitting")
                result = analyzer.evaluate(model, cv=cv, test_size=test_size, smote=smote, model_params=params, calibration_bins=calibration_bins)
                if model in {"lda", "plsda"}:
                    key, columns = analyzer.store_latent_scores()
                    progress(0.82, f"Writing obsm[{key!r}] and obs[{', '.join(columns)}]")
                st.session_state.supervised = analyzer
                progress(0.92, "Generating QC, PR and calibration reports")
                test_metrics = pd.DataFrame([result["test_metrics"]], index=["holdout"])
                qc_metrics = pd.DataFrame([{key: value for key, value in result["model_qc"].items() if isinstance(value, (int, float))}], index=["model_qc"])
                return pd.concat([test_metrics, qc_metrics, pd.DataFrame(result["classification_report"]).T])
            result = _execute(st, terminal, "监督学习", train, "analysis")
            if result is not None: st.session_state.result_table = result; st.rerun()
        shap_default, shap_help = _parameter("supervised.shap_samples", 100)
        shap_samples = st.number_input("SHAP 最大样本数", 1, 1000, int(shap_default), help=shap_help)
        if st.button("计算 SHAP", disabled=st.session_state.supervised is None):
            result = _execute(st, terminal, "SHAP", lambda progress: (progress(0.1, "Selecting background and samples"), st.session_state.supervised.explain_shap(max_samples=shap_samples), progress(0.92, "Aggregating SHAP values"))[1], "analysis")
            if result is not None:
                values = np.asarray(result.values)
                values = np.mean(np.abs(values), axis=(0, 2)) if values.ndim == 3 else np.mean(np.abs(values), axis=0)
                st.session_state.result_table = pd.DataFrame({"feature": st.session_state.supervised.feature_names, "mean_abs_shap": values}).sort_values("mean_abs_shap", ascending=False)
                st.rerun()
        analyzer = st.session_state.supervised
        if analyzer is not None and hasattr(analyzer, "model_"):
            _render_shap_plots(st, analyzer)
            _render_supervised_plots(st, analyzer)


def page_results():
    import streamlit as st

    st.header("结果表、报告与导出")
    data = st.session_state.data
    if data is None:
        st.warning("暂无数据。")
        return
    st.metric("Cells × features", f"{data.n_obs:,} × {data.n_vars:,}")
    if hasattr(data, "cell_keep_mask"):
        st.caption(
            f"当前后续分析使用 {int(data.cell_keep_mask().sum()):,} 个细胞 × "
            f"{int(data.feature_keep_mask().sum()):,} 个特征。"
        )
    options = ["X", "obs", "var"]; index, help_text = _choice("results.table", options); table_name = st.radio("AnnData 表", options, index=index, horizontal=True, help=help_text)
    if table_name == "X":
        configured, enhanced_help = _parameter("results.enhanced_table", True); enhanced = st.checkbox("显示并冻结 obs 列和 var 行", value=bool(configured), help=enhanced_help)
        if enhanced:
            a, b = st.columns(2)
            obs_key = a.selectbox("最左列显示的 obs", data.obs.columns.astype(str).tolist())
            var_key = b.selectbox("最上行显示的 var", data.var.columns.astype(str).tolist())
            _sticky_matrix_table(st, data, obs_key, var_key)
        else:
            _sticky_matrix_table(st, data, None, None)
    else:
        st.dataframe(getattr(data, table_name), use_container_width=True, height=520)
    st.subheader("分析结果")
    result = st.session_state.result_table
    if result is not None and not result.empty:
        st.dataframe(result, use_container_width=True, height=480)
        st.download_button("下载分析结果 CSV", result.to_csv(index=True).encode("utf-8-sig"), "scmm_analysis_result.csv", "text/csv")
    else:
        st.info("尚无统计或模型结果。")
    st.subheader("导出完整数据")
    a, b = st.columns(2)
    if a.button("生成 H5AD 下载文件", use_container_width=True):
        with st.spinner("正在生成完整 H5AD…"):
            payload = _h5ad_bytes(data)
        a.download_button("下载 H5AD", payload, "scmm_result.h5ad", "application/octet-stream", on_click="ignore", use_container_width=True)
    if b.button("生成 CSV 下载包", use_container_width=True):
        with st.spinner("正在生成完整 CSV 包…"):
            payload = _csv_zip(data)
        b.download_button("下载 X/obs/var CSV 包", payload, "scmm_tables.zip", "application/zip", on_click="ignore", use_container_width=True)


def page_terminal():
    import streamlit as st

    st.header("任务进度与运行报告")
    terminal = _terminal(st)
    a, b = st.columns(2)
    if a.button("清空终端日志"):
        st.session_state.logs = [f"[{_now()}] Terminal cleared"]
        st.rerun()
    if b.button("清空任务报告"):
        st.session_state.tasks = []
        st.rerun()
    terminal.render()


def page_annotation():
    from scMM.annotation_ui import render_annotation_page
    render_annotation_page()


def render_app():
    import streamlit as st

    st.set_page_config(page_title="SCMM CyESI Workbench", page_icon="🧬", layout="wide")
    _state(st)
    pages = [
        st.Page(page_data, title="数据导入与峰提取", icon="📥", default=True),
        st.Page(page_preview, title="TIC / EIC / 质谱预览", icon="📈"),
        st.Page(page_preprocess, title="预处理与质量控制", icon="🧹"),
        st.Page(page_discovery, title="降维、聚类与轨迹", icon="🧭"),
        st.Page(page_analysis, title="统计与监督学习", icon="🧪"),
        st.Page(page_annotation, title="物质注释", icon="🔎"),
        st.Page(page_results, title="结果与导出", icon="📦"),
        st.Page(page_terminal, title="任务终端", icon="🖥️"),
    ]
    navigation = st.navigation(pages, position="sidebar", expanded=True)
    st.sidebar.markdown("**SCMM CyESI Workbench**")
    data = st.session_state.data
    if data is not None:
        _layer_selector(st, data)
        st.sidebar.success(f"{data.n_obs:,} cells × {data.n_vars:,} features")
        if hasattr(data, "cell_keep_mask"):
            st.sidebar.caption(
                f"Active: {int(data.cell_keep_mask().sum()):,} cells × "
                f"{int(data.feature_keep_mask().sum()):,} features"
            )
    navigation.run()


def main():
    target = str(Path(__file__).resolve())
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        target,
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    render_app()
