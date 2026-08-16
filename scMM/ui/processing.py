"""Guided processing controls backed by persistent server-side tasks."""

from __future__ import annotations

from html import escape
from pathlib import Path

import panel as pn
import plotly.graph_objects as go

from scMM.application import (
    OutputCatalog,
    OutputRoot,
    ProcessingParameters,
    ProcessingPlanner,
    ProcessingRequest,
    ProcessingTaskManager,
    StorageCatalog,
    load_quality_report,
)

_PLOT_CONFIG = {"displaylogo": False, "responsive": True, "scrollZoom": True}

_PRESETS = {
    "均衡（推荐起点）": "balanced",
    "灵敏（保留更多弱信号）": "sensitive",
    "严格（减少噪声特征）": "strict",
}
_STATUS_LABELS = {
    "queued": "等待启动",
    "running": "处理中",
    "succeeded": "已完成",
    "failed": "失败",
}


class GuidedProcessingPanel:
    """Collect, preflight, submit, and recover processing tasks."""

    def __init__(self, storage: StorageCatalog, output_roots: tuple[OutputRoot, ...]) -> None:
        self.outputs = OutputCatalog(output_roots)
        self.planner = ProcessingPlanner(storage, self.outputs)
        self.tasks = ProcessingTaskManager(self.planner, self.outputs.roots[0].path / ".scmm-tasks")
        self._storage_label: str | None = None
        self._input_path: str | None = None
        self._request: ProcessingRequest | None = None
        self._active_task_id: str | None = None
        self._periodic = None

        self.input_text = pn.pane.Markdown(
            "尚未选择输入文件。请先在左侧选择并预览原始数据。",
            css_classes=["scmm-card"],
        )
        self.preset = pn.widgets.Select(
            label="参数预设",
            options=_PRESETS,
            value="balanced",
            width=210,
            sizing_mode=None,
        )
        self.output_select = pn.widgets.Select(
            label="结果存储",
            options=[root.label for root in self.outputs.roots],
            width=180,
            sizing_mode=None,
        )
        self.result_name = pn.widgets.TextInput(
            label="结果名称", placeholder="默认使用原始文件名", width=220, sizing_mode=None
        )
        self.ref_mz = _float_input("参考离子 m/z", 100.0, step=0.0001)
        self.ppm_tol = _float_input("对齐容差 ppm", 10.0, step=0.5)
        self.resolution = _float_input("m/z 200 分辨率", 35_000.0, step=1_000)
        self.cell_snr = _float_input("细胞 SNR", 5.0, step=0.5)
        self.peak_snr = _float_input("特征 SNR", 3.0, step=0.5)
        self.n_jobs = pn.widgets.IntInput(
            label="CPU 任务数", value=1, start=-1, end=64, width=140, sizing_mode=None
        )
        self.ms_peak_snr = _float_input("总谱 SNR", 10.0, step=1.0)
        self.points_per_fwhm = _float_input("每 FWHM 采样点", 5.0, step=0.5)
        self.baseline_size = pn.widgets.IntInput(
            label="基线窗口", value=50, start=1, end=100_000, width=140, sizing_mode=None
        )
        self.max_zero_frac = _float_input("最大零值比例", 0.9, step=0.01)
        self.advanced_toggle = pn.widgets.Toggle(
            label="显示高级参数", icon="adjustments", width=150, height=36, sizing_mode="fixed"
        )
        self.advanced = pn.FlexBox(
            self.ms_peak_snr,
            self.points_per_fwhm,
            self.baseline_size,
            self.max_zero_frac,
            gap="10px",
            visible=False,
            sizing_mode="stretch_width",
        )
        self.overwrite = pn.widgets.Checkbox(label="允许写入已有同名结果", value=False)
        self.preflight_button = pn.widgets.Button(
            label="检查参数与存储",
            icon="checklist",
            color="primary",
            width=170,
            height=38,
            sizing_mode="fixed",
        )
        self.confirm = pn.widgets.Checkbox(
            label="我已核对参考离子、参数和输出位置",
            value=False,
            disabled=True,
        )
        self.submit_button = pn.widgets.Button(
            label="开始后台处理",
            icon="player-play",
            color="success",
            width=160,
            height=38,
            sizing_mode="fixed",
            disabled=True,
        )
        self.preflight_text = pn.pane.Markdown("修改参数后先执行预检。", css_classes=["scmm-card"])

        self.task_select = pn.widgets.Select(
            label="任务记录", options={"暂无任务": None}, value=None, sizing_mode="stretch_width"
        )
        self.refresh_button = pn.widgets.Button(
            label="刷新状态",
            icon="refresh",
            width=120,
            height=36,
            sizing_mode="fixed",
        )
        self.status_text = pn.pane.Markdown("暂无任务。", css_classes=["scmm-card"])
        self.log_text = pn.widgets.TextAreaInput(
            label="任务日志（最新 100 KB）",
            value="",
            disabled=True,
            height=220,
            sizing_mode="stretch_width",
        )
        self.quality_summary = pn.pane.Markdown(
            "任务完成后显示质量检查。", css_classes=["scmm-card"]
        )
        self.intensity_plot = pn.pane.Plotly(
            _empty_figure("每细胞总强度"),
            config=_PLOT_CONFIG,
            height=280,
            styles={"flex": "1 1 420px", "min-width": "0"},
        )
        self.detection_plot = pn.pane.Plotly(
            _empty_figure("特征检出率"),
            config=_PLOT_CONFIG,
            height=280,
            styles={"flex": "1 1 420px", "min-width": "0"},
        )
        self.pca_plot = pn.pane.Plotly(
            _empty_figure("PCA"),
            config=_PLOT_CONFIG,
            height=320,
            styles={"flex": "1 1 420px", "min-width": "0"},
        )
        self.umap_plot = pn.pane.Plotly(
            _empty_figure("UMAP"),
            config=_PLOT_CONFIG,
            height=320,
            styles={"flex": "1 1 420px", "min-width": "0"},
        )
        self.cell_table = pn.pane.DataFrame(height=260, sizing_mode="stretch_width", index=False)
        self.feature_table = pn.pane.DataFrame(height=260, sizing_mode="stretch_width", index=False)
        artifact_names = (
            ("处理矩阵 CSV", "data.csv"),
            ("细胞元数据 CSV", "peak_meta.csv"),
            ("特征元数据 CSV", "feature_meta.csv"),
            ("细胞质量 CSV", "cell-quality.csv"),
            ("特征质量 CSV", "feature-quality.csv"),
            ("PCA/UMAP CSV", "embedding.csv"),
            ("运行清单 JSON", "scmm-manifest.json"),
        )
        self.artifact_downloads = {
            filename: pn.widgets.FileDownload(
                label=label,
                icon="download",
                file=None,
                disabled=True,
                width=145,
                height=36,
                sizing_mode="fixed",
            )
            for label, filename in artifact_names
        }
        self.quality_section = pn.Column(
            "### 4. 质量检查与结果下载",
            self.quality_summary,
            pn.FlexBox(
                self.intensity_plot,
                self.detection_plot,
                gap="12px",
                sizing_mode="stretch_width",
            ),
            pn.FlexBox(
                self.pca_plot,
                self.umap_plot,
                gap="12px",
                sizing_mode="stretch_width",
            ),
            pn.Tabs(
                ("细胞质量（前 200 行）", self.cell_table),
                ("特征质量（前 200 行）", self.feature_table),
                dynamic=False,
                sizing_mode="stretch_width",
            ),
            pn.FlexBox(
                *self.artifact_downloads.values(),
                gap="8px",
                sizing_mode="stretch_width",
            ),
            visible=False,
            sizing_mode="stretch_width",
        )
        self._quality_task_id: str | None = None

        self._wire_callbacks()
        self._reload_tasks()

    def panel(self):
        """Return the processing and task-status page."""
        primary = pn.FlexBox(
            self.preset,
            self.ref_mz,
            self.ppm_tol,
            self.resolution,
            self.cell_snr,
            self.peak_snr,
            self.n_jobs,
            gap="10px",
            align_items="flex-end",
            sizing_mode="stretch_width",
        )
        destinations = pn.FlexBox(
            self.output_select,
            self.result_name,
            self.overwrite,
            gap="10px",
            align_items="flex-end",
            sizing_mode="stretch_width",
        )
        actions = pn.FlexBox(
            self.preflight_button,
            self.confirm,
            self.submit_button,
            gap="12px",
            align_items="center",
            sizing_mode="stretch_width",
        )
        task_header = pn.FlexBox(
            self.task_select,
            self.refresh_button,
            gap="10px",
            align_items="flex-end",
            sizing_mode="stretch_width",
        )
        return pn.Column(
            "## ③ 处理与结果",
            "按顺序完成参数设置、预检、明确确认和后台提交。关闭页面不会中止已提交任务。",
            self.input_text,
            "### 1. 参数与输出",
            primary,
            self.advanced_toggle,
            self.advanced,
            destinations,
            "### 2. 提交前检查",
            self.preflight_text,
            actions,
            pn.layout.Divider(),
            "### 3. 任务状态",
            task_header,
            self.status_text,
            self.log_text,
            self.quality_section,
            sizing_mode="stretch_both",
        )

    def set_input(self, storage_label: str | None, input_path: str | None) -> None:
        """Use a raw-file selection from the preview workflow."""
        self._storage_label = storage_label
        self._input_path = input_path
        if storage_label is None or input_path is None:
            self.input_text.object = "尚未选择输入文件。请先在左侧选择并预览原始数据。"
        else:
            self.result_name.value = Path(input_path).stem
            self.input_text.object = (
                f"**输入存储：** {escape(storage_label)}  \n**原始文件：** `{escape(input_path)}`"
            )
        self._invalidate_preflight()

    def start_polling(self) -> None:
        """Start session-local status polling after the Bokeh document loads."""
        if self._periodic is None:
            self._periodic = pn.state.add_periodic_callback(self.poll, period=2_000)

    def poll(self) -> None:
        """Refresh the selected task status and bounded log tail."""
        if self._active_task_id is None:
            return
        try:
            task = self.tasks.get(self._active_task_id)
        except KeyError:
            self.status_text.object = "所选任务记录已不存在。"
            return
        status = _STATUS_LABELS[task.status]
        details = [
            f"**状态：** {status}",
            f"**任务 ID：** `{task.task_id}`",
            f"**输入：** `{escape(task.input_path)}`",
            f"**结果：** `{escape(task.result_path)}`",
        ]
        if task.error:
            details.append(f"**错误：** {escape(task.error)}")
        self.status_text.object = "  \n".join(details)
        self.log_text.value = self.tasks.read_log(task.task_id)
        if task.status == "succeeded" and self._quality_task_id != task.task_id:
            self._load_quality(task)
        elif task.status != "succeeded" and self._quality_task_id != task.task_id:
            self.quality_section.visible = False

    def _wire_callbacks(self) -> None:
        self.preset.param.watch(self._apply_preset, "value")
        self.advanced_toggle.param.watch(
            lambda event: setattr(self.advanced, "visible", event.new), "value"
        )
        parameters = (
            self.output_select,
            self.result_name,
            self.ref_mz,
            self.ppm_tol,
            self.resolution,
            self.cell_snr,
            self.peak_snr,
            self.n_jobs,
            self.ms_peak_snr,
            self.points_per_fwhm,
            self.baseline_size,
            self.max_zero_frac,
            self.overwrite,
        )
        for widget in parameters:
            widget.param.watch(self._invalidate_preflight, "value")
        self.confirm.param.watch(self._update_submit_state, "value")
        self.preflight_button.on_click(self._preflight)
        self.submit_button.on_click(self._submit)
        self.refresh_button.on_click(lambda _event: self._refresh_tasks())
        self.task_select.param.watch(self._select_task, "value")

    def _apply_preset(self, event) -> None:
        params = ProcessingParameters.from_preset(event.new, max(self.ref_mz.value, 1.0))
        self.ms_peak_snr.value = params.ms_peak_snr_threshold
        self.cell_snr.value = params.cell_snr
        self.peak_snr.value = params.peak_snr
        self.max_zero_frac.value = params.max_zero_frac
        self._invalidate_preflight()

    def _parameters(self) -> ProcessingParameters:
        return ProcessingParameters(
            ref_mz=self.ref_mz.value,
            ppm_tol=self.ppm_tol.value,
            resolution=self.resolution.value,
            resample_points_per_fwhm=self.points_per_fwhm.value,
            ms_peak_snr_threshold=self.ms_peak_snr.value,
            cell_snr=self.cell_snr.value,
            peak_snr=self.peak_snr.value,
            baseline_filter_size=self.baseline_size.value,
            max_zero_frac=self.max_zero_frac.value,
            n_jobs=self.n_jobs.value,
        )

    def _build_request(self) -> ProcessingRequest:
        if self._storage_label is None or self._input_path is None:
            raise ValueError("请先选择一个原始数据文件")
        result_name = self.result_name.value.strip() or Path(self._input_path).stem
        return ProcessingRequest(
            storage_label=self._storage_label,
            input_path=self._input_path,
            output_label=self.output_select.value,
            result_name=result_name,
            overwrite=self.overwrite.value,
            parameters=self._parameters(),
        )

    def _preflight(self, _event=None) -> None:
        try:
            request = self._build_request()
            plan = self.planner.preflight(request)
        except Exception as exc:
            self._request = None
            self.preflight_text.object = f"❌ **预检未通过：** {escape(str(exc))}"
            self.confirm.disabled = True
            self.submit_button.disabled = True
            return
        self._request = request
        warnings = ""
        if plan.warnings:
            warnings = "  \n**提示：** " + "；".join(escape(item) for item in plan.warnings)
        self.preflight_text.object = (
            "✅ **预检通过**  \n"
            f"输入大小：{_human_size(plan.input_size_bytes)}　"
            f"输出可用：{_human_size(plan.free_bytes)}  \n"
            f"结果目录：`{escape(str(plan.result_path))}`{warnings}"
        )
        self.confirm.disabled = False
        self.confirm.value = False
        self._update_submit_state()

    def _submit(self, _event=None) -> None:
        if self._request is None or not self.confirm.value:
            return
        self.submit_button.loading = True
        try:
            task = self.tasks.submit(self._request)
        except Exception as exc:  # Panel callbacks must report task/preflight errors in-session.
            self.status_text.object = f"❌ **无法提交：** {escape(str(exc))}"
            return
        finally:
            self.submit_button.loading = False
        self._active_task_id = task.task_id
        self._reload_tasks(select_id=task.task_id)
        self.poll()
        self._invalidate_preflight()

    def _invalidate_preflight(self, _event=None) -> None:
        self._request = None
        self.confirm.value = False
        self.confirm.disabled = True
        self.submit_button.disabled = True
        self.preflight_text.object = "输入或参数已变化，请重新执行预检。"

    def _update_submit_state(self, _event=None) -> None:
        self.submit_button.disabled = self._request is None or not self.confirm.value

    def _refresh_tasks(self) -> None:
        self._reload_tasks(select_id=self._active_task_id)
        self.poll()

    def _reload_tasks(self, select_id: str | None = None) -> None:
        tasks = self.tasks.list()
        if not tasks:
            self.task_select.param.update(options={"暂无任务": None}, value=None)
            return
        options = {
            f"{task.created_at[:19]} · {_STATUS_LABELS[task.status]} · {Path(task.input_path).name}": task.task_id
            for task in tasks
        }
        value = select_id if select_id in options.values() else tasks[0].task_id
        self.task_select.param.update(options=options, value=value)
        self._active_task_id = value
        self.poll()

    def _select_task(self, event) -> None:
        self._active_task_id = event.new
        self.poll()

    def _load_quality(self, task) -> None:
        try:
            result_path = self._safe_result_path(task.result_path)
            report = load_quality_report(result_path)
        except Exception as exc:
            self.quality_section.visible = True
            self.quality_summary.object = f"质量检查文件暂不可用：{escape(str(exc))}"
            return
        summary = report.summary
        warning_text = ""
        if summary.embedding_warnings:
            warning_text = "  \n**嵌入提示：** " + "；".join(
                escape(item) for item in summary.embedding_warnings
            )
        self.quality_summary.object = (
            f"**细胞事件：** {summary.cell_count:,}　"
            f"**特征：** {summary.feature_count:,}　"
            f"**整体零值：** {summary.zero_fraction:.1%}　"
            f"**中位总强度：** {summary.median_total_intensity:.4g}　"
            f"**中位检出特征：** {summary.median_detected_features:.0f}{warning_text}"
        )
        self.intensity_plot.object = _histogram_figure(
            report.cells.get("total_intensity", []), "每细胞总强度", "总强度"
        )
        self.detection_plot.object = _detection_figure(report.features)
        self.pca_plot.object = _embedding_figure(report, "PCA1", "PCA2", "PCA")
        self.umap_plot.object = _embedding_figure(report, "UMAP1", "UMAP2", "UMAP")
        self.cell_table.object = report.cells.head(200)
        self.feature_table.object = report.features.head(200)
        for filename, download in self.artifact_downloads.items():
            path = result_path / filename
            download.file = str(path) if path.is_file() else None
            download.filename = f"{result_path.name}_{filename}"
            download.disabled = not path.is_file()
        self._quality_task_id = task.task_id
        self.quality_section.visible = True

    def _safe_result_path(self, value: str) -> Path:
        result = Path(value).resolve(strict=True)
        if not result.is_dir() or not (result / ".meta").is_file():
            raise FileNotFoundError(f"不是完整的 scMM 结果目录：{result}")
        if not any(result.parent == root.path for root in self.outputs.roots):
            raise PermissionError("任务结果不在已配置的输出根目录内")
        return result


def _float_input(label: str, value: float, *, step: float):
    return pn.widgets.FloatInput(
        label=label,
        value=value,
        step=step,
        width=150,
        sizing_mode=None,
    )


def _human_size(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title)
    return _style_figure(figure)


def _histogram_figure(values, title: str, x_title: str) -> go.Figure:
    figure = go.Figure(go.Histogram(x=values, marker_color="#0F766E"))
    figure.update_layout(title=title, xaxis_title=x_title, yaxis_title="细胞数")
    return _style_figure(figure)


def _detection_figure(features) -> go.Figure:
    figure = go.Figure(
        go.Scattergl(
            x=features.get("mz", []),
            y=features.get("detection_rate", []),
            mode="markers",
            marker={"color": "#C2410C", "size": 6, "opacity": 0.7},
        )
    )
    figure.update_layout(title="特征检出率", xaxis_title="m/z", yaxis_title="检出率")
    figure.update_yaxes(range=[0, 1.02])
    return _style_figure(figure)


def _embedding_figure(report, x_name: str, y_name: str, title: str) -> go.Figure:
    if x_name not in report.embedding or y_name not in report.embedding:
        return _empty_figure(f"{title}（当前数据不可用）")
    totals = report.cells.set_index("cell_index")["total_intensity"]
    colors = report.embedding["cell_index"].map(totals)
    figure = go.Figure(
        go.Scattergl(
            x=report.embedding[x_name],
            y=report.embedding[y_name],
            mode="markers",
            marker={
                "color": colors,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "总强度"},
                "size": 6,
                "opacity": 0.75,
            },
            text=report.embedding["cell_index"],
            hovertemplate="cell=%{text}<br>x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>",
        )
    )
    figure.update_layout(title=title, xaxis_title=x_name, yaxis_title=y_name)
    return _style_figure(figure)


def _style_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        margin={"l": 55, "r": 25, "t": 55, "b": 48},
        hovermode="closest",
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return figure


__all__ = ["GuidedProcessingPanel"]
