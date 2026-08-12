"""Guided Panel application for selecting and previewing raw MS files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from scMM.application import RawFilePreview, RawPreviewService, StorageCatalog, StorageRoot

pn.extension("plotly", notifications=True, sizing_mode="stretch_width")

_ACCENT = "#0F766E"
_PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "svg", "filename": "scmm-preview"},
}
_STYLESHEET = """
:root { --scmm-accent: #0f766e; }
.scmm-card {
  background: var(--panel-background-color, white);
  border: 1px solid color-mix(in srgb, var(--scmm-accent) 18%, #d1d5db);
  border-radius: 12px;
  box-shadow: 0 3px 14px rgba(15, 118, 110, 0.06);
  padding: 12px 16px;
  overflow-wrap: anywhere;
}
.scmm-hint { color: #64748b; font-size: 0.92rem; }
"""


class PreviewWorkspace:
    """State and callbacks for one browser session."""

    def __init__(self, roots: tuple[StorageRoot, ...]) -> None:
        self.catalog = StorageCatalog(roots)
        self.service = RawPreviewService(self.catalog)
        self.preview: RawFilePreview | None = None
        self.tic = pd.DataFrame()
        self.eic = pd.DataFrame()
        self.spectrum = pd.DataFrame()
        self._browser_directory = Path(".")
        self._directory_values: set[str] = set()
        self._selected_path: str | None = None

        labels = [root.label for root in self.catalog.roots]
        self.root_select = pn.widgets.Select(
            label="数据存储",
            options=labels,
            value=labels[0],
            width=None,
            min_width=120,
            max_width=180,
            sizing_mode="stretch_width",
        )
        self.directory_text = pn.pane.Markdown("目录：`/`", width=180, sizing_mode=None)
        self.file_select = pn.widgets.Select(
            label="文件夹 / 原始数据",
            options={"正在读取…": None},
            value=None,
            width=180,
            sizing_mode=None,
        )
        self.up_button = pn.widgets.Button(
            label="上一级",
            icon="arrow-back-up",
            width=84,
            height=34,
            sizing_mode="fixed",
        )
        self.reload_button = pn.widgets.Button(
            label="刷新",
            icon="refresh",
            width=84,
            height=34,
            sizing_mode="fixed",
        )
        browser_actions = pn.Row(
            self.up_button,
            self.reload_button,
            width=180,
            sizing_mode=None,
        )
        self.selector_area = pn.Column(
            self.directory_text,
            self.file_select,
            browser_actions,
            width=180,
            sizing_mode=None,
        )
        self.selection_text = pn.pane.Markdown(
            "<span class='scmm-hint'>请选择一个 mzML 或 mzXML 文件。</span>"
        )
        self.load_button = pn.widgets.Button(
            label="打开并预览",
            color="primary",
            icon="database-search",
            disabled=True,
        )

        self.summary = pn.pane.Markdown("尚未加载数据。", css_classes=["scmm-card"])
        self.ms_level = pn.widgets.Select(
            label="MS",
            options=[1],
            value=1,
            disabled=True,
            width=96,
            height=54,
            sizing_mode="fixed",
        )
        self.target_mz = pn.widgets.FloatInput(
            label="目标 m/z",
            value=100.0,
            step=0.0001,
            disabled=True,
            width=96,
            height=54,
            sizing_mode="fixed",
        )
        self.ppm = pn.widgets.FloatInput(
            label="ppm",
            value=5.0,
            step=0.5,
            disabled=True,
            width=96,
            height=54,
            sizing_mode="fixed",
        )
        self.rt_range = pn.widgets.RangeSlider(
            label="保留时间范围（秒）",
            start=0.0,
            end=1.0,
            value=(0.0, 1.0),
            step=0.1,
            disabled=True,
            width=180,
            sizing_mode=None,
        )
        self.mz_min = pn.widgets.FloatInput(
            label="m/z 下限",
            value=100.0,
            disabled=True,
            width=96,
            height=54,
            sizing_mode="fixed",
        )
        self.mz_max = pn.widgets.FloatInput(
            label="m/z 上限",
            value=1000.0,
            disabled=True,
            width=96,
            height=54,
            sizing_mode="fixed",
        )
        self.average_spectrum = pn.widgets.Checkbox(label="显示平均谱", value=False, disabled=True)
        self.refresh_button = pn.widgets.Button(
            label="应用范围并刷新",
            color="primary",
            icon="refresh",
            disabled=True,
        )

        self.tic_pane = pn.pane.Plotly(
            _empty_figure("总离子流图（TIC）"),
            config=_PLOT_CONFIG,
            height=280,
            styles={"flex": "1 1 420px", "min-width": "0"},
        )
        self.eic_pane = pn.pane.Plotly(
            _empty_figure("提取离子流图（EIC）"),
            config=_PLOT_CONFIG,
            height=280,
            styles={"flex": "1 1 420px", "min-width": "0"},
        )
        self.spectrum_pane = pn.pane.Plotly(
            _empty_figure("合并谱"),
            config=_PLOT_CONFIG,
            height=340,
            styles={"min-width": "0"},
        )

        self.tic_download = pn.widgets.FileDownload(
            label="下载 TIC CSV",
            icon="download",
            callback=lambda: _csv_buffer(self.tic),
            disabled=True,
            width=128,
            sizing_mode="fixed",
        )
        self.eic_download = pn.widgets.FileDownload(
            label="下载 EIC CSV",
            icon="download",
            callback=lambda: _csv_buffer(self.eic),
            disabled=True,
            width=128,
            sizing_mode="fixed",
        )
        self.spectrum_download = pn.widgets.FileDownload(
            label="下载谱图 CSV",
            icon="download",
            callback=lambda: _csv_buffer(self.spectrum),
            disabled=True,
            width=128,
            sizing_mode="fixed",
        )

        self.tabs = pn.Tabs(dynamic=True, sizing_mode="stretch_both")
        self._build_tabs()
        self._refresh_browser()
        self.root_select.param.watch(self._on_root_change, "value")
        self.file_select.param.watch(self._on_browser_selection, "value")
        self.up_button.on_click(self._go_to_parent)
        self.reload_button.on_click(self._reload_browser)
        self.load_button.on_click(self._load_selected)
        self.refresh_button.on_click(self._refresh_all)
        self.spectrum_pane.param.watch(self._use_clicked_mz, "click_data")

    def _build_tabs(self) -> None:
        selection_page = pn.Column(
            pn.pane.Markdown(
                """## ① 选择服务器上的原始数据

从左侧选择已挂载的数据存储和一个原始谱文件。浏览器只显示启动时明确开放的目录；
文件在服务器端直接读取，不会先上传到浏览器。"""
            ),
            self.selection_text,
            self.summary,
            sizing_mode="stretch_width",
        )
        preview_header = pn.FlexBox(
            pn.pane.Markdown(
                "## ② 初步查看\n缩放、框选或悬停检查信号；点击合并谱上的点可填入 EIC 目标。",
                sizing_mode="stretch_width",
                styles={"flex": "1 1 420px"},
            ),
            self.tic_download,
            self.eic_download,
            self.spectrum_download,
            align_items="flex-end",
            gap="8px",
            sizing_mode="stretch_width",
        )
        chromatograms = pn.FlexBox(
            self.tic_pane,
            self.eic_pane,
            gap="12px",
            sizing_mode="stretch_width",
        )
        preview_page = pn.Column(
            preview_header,
            self.summary,
            chromatograms,
            self.spectrum_pane,
            sizing_mode="stretch_both",
        )
        processing_page = pn.Column(
            pn.pane.Markdown(
                """## ③ 处理与结果

该阶段将复用当前预览选择，按“参数预检 → 后台处理 → 质量检查 → 结果转存”引导操作。
首个可运行版本先固定原始数据选择、TIC/EIC/合并谱预览和 CSV 下载边界。""",
                css_classes=["scmm-card"],
            )
        )
        self.tabs.extend(
            [
                ("① 数据选择", selection_page),
                ("② 原始数据预览", preview_page),
                ("③ 处理与结果", processing_page),
            ]
        )

    def sidebar(self) -> pn.Column:
        """Build the guided control column."""
        preview_inputs = pn.FlexBox(
            self.ms_level,
            self.mz_min,
            self.mz_max,
            self.average_spectrum,
            gap="8px",
            align_items="flex-end",
            sizing_mode="stretch_width",
        )
        eic_inputs = pn.FlexBox(
            self.target_mz,
            self.ppm,
            gap="8px",
            align_items="flex-end",
            sizing_mode="stretch_width",
        )
        return pn.Column(
            "### 1. 选择数据",
            self.root_select,
            self.selector_area,
            self.load_button,
            pn.layout.Divider(),
            "### 2. 查看范围",
            self.rt_range,
            preview_inputs,
            pn.layout.Divider(),
            "### 3. 提取离子流",
            eic_inputs,
            self.refresh_button,
            sizing_mode="stretch_width",
        )

    def _refresh_browser(self) -> None:
        entries = tuple(
            entry
            for entry in self.catalog.list_entries(self.root_select.value, self._browser_directory)
            if not entry.name.startswith(".")
        )
        self._directory_values = {
            entry.relative_path.as_posix() for entry in entries if entry.is_directory
        }
        options: dict[str, str | None] = {"请选择…": None}
        for entry in entries:
            prefix = "📁 " if entry.is_directory else "📄 "
            options[f"{prefix}{entry.name}"] = entry.relative_path.as_posix()
        if len(options) == 1:
            options = {"当前目录没有可浏览的数据": None}

        relative = self._browser_directory.as_posix()
        self.directory_text.object = f"目录：`/{'' if relative == '.' else relative}`"
        self.up_button.disabled = self._browser_directory == Path(".")
        self.file_select.param.update(options=options, value=None)

    def _on_root_change(self, _event) -> None:
        self._browser_directory = Path(".")
        self._selected_path = None
        self._refresh_browser()
        self._update_file_selection()

    def _on_browser_selection(self, event) -> None:
        selected = event.new
        if selected is None:
            return
        if selected in self._directory_values:
            self._browser_directory = Path(selected)
            self._selected_path = None
            self._refresh_browser()
        else:
            self._selected_path = selected
        self._update_file_selection()

    def _go_to_parent(self, _event) -> None:
        if self._browser_directory == Path("."):
            return
        parent = self._browser_directory.parent
        self._browser_directory = Path(".") if parent == Path(".") else parent
        self._selected_path = None
        self._refresh_browser()
        self._update_file_selection()

    def _reload_browser(self, _event) -> None:
        self._selected_path = None
        self._refresh_browser()
        self._update_file_selection()

    def _update_file_selection(self) -> None:
        self.load_button.disabled = self._selected_path is None
        if self._selected_path is None:
            self.selection_text.object = "请选择一个 mzML 或 mzXML 文件。"
        else:
            self.selection_text.object = f"已选择：`{self._selected_path}`"

    def _load_selected(self, _event) -> None:
        if self._selected_path is None:
            return
        self.load_button.loading = True
        try:
            preview = self.service.open(self.root_select.value, self._selected_path)
            self.preview = preview
            self._configure_controls(preview)
            self._calculate_all()
            self.tabs.active = 1
            pn.state.notifications.success(f"已加载 {preview.path.name}", duration=3000)
        except Exception as exc:  # Panel callbacks need to report errors in the session.
            pn.state.notifications.error(f"加载失败：{exc}", duration=8000)
        finally:
            self.load_button.loading = False

    def _configure_controls(self, preview: RawFilePreview) -> None:
        summary = preview.summary
        levels = list(preview.ms_levels) or [1]
        self.ms_level.options = levels
        self.ms_level.value = 1 if 1 in levels else levels[0]
        rt_min = summary.rt_min_seconds if summary.rt_min_seconds is not None else 0.0
        rt_max = summary.rt_max_seconds if summary.rt_max_seconds is not None else rt_min + 1.0
        if rt_max <= rt_min:
            rt_max = rt_min + 1.0
        self.rt_range.param.update(start=rt_min, end=rt_max, value=(rt_min, rt_max))
        mz_min = summary.mz_min if summary.mz_min is not None else 100.0
        mz_max = summary.mz_max if summary.mz_max is not None else mz_min + 1.0
        if mz_max <= mz_min:
            mz_max = mz_min + max(abs(mz_min) * 1e-6, 0.001)
        self.mz_min.value = mz_min
        self.mz_max.value = mz_max
        self.target_mz.value = (mz_min + mz_max) / 2
        for widget in (
            self.ms_level,
            self.target_mz,
            self.ppm,
            self.rt_range,
            self.mz_min,
            self.mz_max,
            self.average_spectrum,
            self.refresh_button,
        ):
            widget.disabled = False
        self._update_summary()

    def _refresh_all(self, _event) -> None:
        self.refresh_button.loading = True
        try:
            self._calculate_all()
        except Exception as exc:  # Panel callbacks need to report errors in the session.
            pn.state.notifications.error(f"刷新失败：{exc}", duration=8000)
        finally:
            self.refresh_button.loading = False

    def _calculate_all(self) -> None:
        if self.preview is None:
            return
        rt_range = tuple(self.rt_range.value)
        ms_level = int(self.ms_level.value)
        self.tic = self.preview.total_ion_chromatogram(ms_level=ms_level, rt_range=rt_range)
        self.eic = self.preview.extracted_ion_chromatogram(
            self.target_mz.value,
            ppm_tolerance=self.ppm.value,
            ms_level=ms_level,
            rt_range=rt_range,
        )
        self.spectrum = self.preview.binned_spectrum(
            mz_range=(self.mz_min.value, self.mz_max.value),
            bins=20_000,
            ms_level=ms_level,
            rt_range=rt_range,
            normalize=self.average_spectrum.value,
        )
        self.tic_pane.object = _chromatogram_figure(self.tic, "总离子流图（TIC）", _ACCENT)
        self.eic_pane.object = _chromatogram_figure(
            self.eic,
            f"提取离子流图（{self.target_mz.value:.5f} ± {self.ppm.value:g} ppm）",
            "#C2410C",
        )
        spectrum_title = "平均谱" if self.average_spectrum.value else "合并谱"
        self.spectrum_pane.object = _spectrum_figure(self.spectrum, spectrum_title)
        stem = self.preview.path.stem
        self.tic_download.filename = f"{stem}_tic.csv"
        self.eic_download.filename = f"{stem}_eic_{self.target_mz.value:.5f}.csv"
        self.spectrum_download.filename = f"{stem}_spectrum.csv"
        self.tic_download.disabled = self.tic.empty
        self.eic_download.disabled = self.eic.empty
        self.spectrum_download.disabled = self.spectrum.empty
        self._update_summary()

    def _use_clicked_mz(self, event) -> None:
        data = event.new
        if not data or not data.get("points"):
            return
        point = data["points"][0]
        if "x" not in point:
            return
        self.target_mz.value = float(point["x"])
        if self.preview is not None:
            self.eic = self.preview.extracted_ion_chromatogram(
                self.target_mz.value,
                ppm_tolerance=self.ppm.value,
                ms_level=int(self.ms_level.value),
                rt_range=tuple(self.rt_range.value),
            )
            self.eic_pane.object = _chromatogram_figure(
                self.eic,
                f"提取离子流图（{self.target_mz.value:.5f} ± {self.ppm.value:g} ppm）",
                "#C2410C",
            )

    def _update_summary(self) -> None:
        if self.preview is None:
            return
        summary = self.preview.summary
        level_text = "，".join(f"MS{level}: {count}" for level, count in summary.scans_by_ms_level)
        rt_text = _range_text(summary.rt_min_seconds, summary.rt_max_seconds, "s")
        mz_text = _range_text(summary.mz_min, summary.mz_max, "")
        instrument = summary.instrument or "未记录"
        self.summary.object = (
            f"### {summary.name}\n"
            f"`{summary.path}`  \n"
            f"**扫描：** {summary.scan_count:,}（{level_text}）　"
            f"**RT：** {rt_text}　**m/z：** {mz_text}　"
            f"**大小：** {_human_size(summary.size_bytes)}　**仪器：** {instrument}"
        )


def create_app(roots: tuple[StorageRoot, ...]):
    """Create an isolated guided UI session for configured storage roots."""
    workspace = PreviewWorkspace(tuple(roots))
    template = pn.template.FastListTemplate(
        title="scMM 数据查看",
        accent_base_color=_ACCENT,
        header_background=_ACCENT,
        sidebar=[workspace.sidebar()],
        main=[workspace.tabs],
        sidebar_width=390,
        theme_toggle=True,
        raw_css=[_STYLESHEET],
    )
    return template


def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(title=title)
    return _style_figure(figure)


def _chromatogram_figure(frame: pd.DataFrame, title: str, color: str) -> go.Figure:
    display = _peak_preserving_downsample(frame, 30_000, "intensity")
    figure = go.Figure(
        go.Scattergl(
            x=display.get("rt_seconds", []),
            y=display.get("intensity", []),
            mode="lines",
            line={"color": color, "width": 1.3},
            hovertemplate="RT %{x:.3f} s<br>强度 %{y:.4g}<extra></extra>",
        )
    )
    figure.update_layout(title=title, xaxis_title="保留时间（秒）", yaxis_title="强度")
    return _style_figure(figure)


def _spectrum_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    display = _peak_preserving_downsample(frame, 40_000, "intensity")
    figure = go.Figure(
        go.Scattergl(
            x=display.get("mz", []),
            y=display.get("intensity", []),
            mode="lines",
            line={"color": "#334155", "width": 1},
            hovertemplate="m/z %{x:.6f}<br>强度 %{y:.4g}<extra></extra>",
        )
    )
    figure.update_layout(title=title, xaxis_title="m/z", yaxis_title="强度")
    return _style_figure(figure)


def _style_figure(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        autosize=True,
        margin={"l": 58, "r": 18, "t": 52, "b": 48},
        hovermode="closest",
        dragmode="zoom",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", rangemode="tozero")
    return figure


def _peak_preserving_downsample(
    frame: pd.DataFrame,
    max_points: int,
    intensity_column: str,
) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    groups = np.arange(len(frame)) * max_points // len(frame)
    positions = frame.groupby(groups, sort=True)[intensity_column].idxmax()
    return frame.loc[positions].sort_index()


def _csv_buffer(frame: pd.DataFrame) -> BytesIO:
    return BytesIO(frame.to_csv(index=False).encode("utf-8"))


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    raise AssertionError("unreachable")


def _range_text(lower: float | None, upper: float | None, unit: str) -> str:
    if lower is None or upper is None:
        return "无"
    suffix = f" {unit}" if unit else ""
    return f"{lower:.3f}–{upper:.3f}{suffix}"
