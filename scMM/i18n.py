"""Language packs and presentation-only localization for the web interface."""
from functools import lru_cache
import json
import os
from pathlib import Path
import re

PACK_DIRECTORY = Path(__file__).with_name("locales")


def languages():
    paths = list(PACK_DIRECTORY.glob("*.json"))
    if os.environ.get("SCMM_LANGUAGE_DIR"):
        paths.extend(Path(os.environ["SCMM_LANGUAGE_DIR"]).glob("*.json"))
    return {pack["code"]: pack for path in paths
            for pack in [json.loads(path.read_text(encoding="utf-8-sig"))]}


def current_language():
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is None:
        return "zh_CN"
    return st.session_state.get("ui_language", "zh_CN")


@lru_cache(maxsize=16)
def _catalog(language):
    pack = languages().get(language, languages()["zh_CN"])
    messages = pack["messages"]
    patterns = []
    for source, target in messages.items():
        if re.search(r"\{\d+\}", source):
            parts = re.split(r"(\{\d+\})", source)
            expression = "".join("(.*?)" if re.fullmatch(r"\{\d+\}", part) else re.escape(part) for part in parts)
            patterns.append((re.compile("^" + expression + "$", re.DOTALL), target))
    return messages, patterns


@lru_cache(maxsize=8192)
def translate(text, language="zh_CN"):
    if not isinstance(text, str):
        return text
    messages, patterns = _catalog(language)
    if text in messages:
        return messages[text]
    for pattern, target in patterns:
        match = pattern.fullmatch(text)
        if match:
            values = [messages.get(value, value) for value in match.groups()]
            return target.format(*values)
    if "**" in text:
        return re.sub(r"\*\*(.+?)\*\*", lambda match: "**" + translate(match[1], language) + "**", text)
    return text


def tr(text):
    return translate(text, current_language())


def language_selector(st):
    packs = languages()
    st.session_state.setdefault("ui_language", "zh_CN")
    st.sidebar.selectbox("Language / 语言", list(packs), key="ui_language",
                         format_func=lambda code: packs[code]["name"])


def localize_figure(figure):
    import plotly.graph_objects as go
    result = go.Figure(figure)
    if result.layout.title.text:
        result.layout.title.text = tr(result.layout.title.text)
    for key in result.layout:
        if key.startswith(("xaxis", "yaxis")):
            axis = result.layout[key]
            if axis.title.text:
                axis.title.text = tr(axis.title.text)
    for annotation in result.layout.annotations or []:
        annotation.text = tr(annotation.text)
    for trace in result.data:
        if trace.name:
            trace.name = tr(trace.name)
        marker = getattr(trace, "marker", None)
        if marker is not None and hasattr(marker, "colorbar") and marker.colorbar.title.text:
            marker.colorbar.title.text = tr(marker.colorbar.title.text)
    return result


_TEXT_METHODS = {"header", "subheader", "title", "caption", "markdown", "text", "write", "info", "error", "warning", "success",
                 "button", "download_button", "checkbox", "toggle", "number_input", "text_input", "text_area", "selectbox", "radio",
                 "multiselect", "slider", "select_slider", "color_picker", "file_uploader", "expander", "dialog", "spinner", "metric"}
_CHOICES = {"selectbox", "radio", "multiselect", "select_slider", "pills", "segmented_control"}
_WIDGETS = {"button", "download_button", "checkbox", "toggle", "number_input", "text_input", "text_area", "selectbox", "radio",
            "multiselect", "slider", "select_slider", "color_picker", "file_uploader"}


class LocalizedUI:
    """Translate presentation text while preserving widget values and callbacks."""
    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.target.__enter__()
        return self

    def __exit__(self, *args):
        return self.target.__exit__(*args)

    def __getattr__(self, name):
        target = getattr(self.target, name)
        if name in {"sidebar", "column_config"}:
            return LocalizedUI(target)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            import inspect
            original = args[0] if args else kwargs.get("label", "")
            if name in _WIDGETS and not kwargs.get("key"):
                caller = inspect.currentframe().f_back
                # Use the source label for a stable widget identity across languages.
                kwargs["key"] = f"ui_{Path(caller.f_code.co_filename).stem}_{caller.f_lineno}_{name}_{original}"
            if name in _TEXT_METHODS or name.endswith("Column"):
                if args and isinstance(args[0], str):
                    args = (tr(args[0]), *args[1:])
                for field in ("label", "help", "text", "body"):
                    if isinstance(kwargs.get(field), str):
                        kwargs[field] = tr(kwargs[field])
            if name in _CHOICES:
                formatter = kwargs.get("format_func", str)
                language = current_language()
                kwargs["format_func"] = lambda value: translate(str(formatter(value)), language)
            if name == "tabs":
                args = ([tr(value) for value in args[0]], *args[1:])
            if name == "Page" and "title" in kwargs:
                kwargs["title"] = tr(kwargs["title"])
            if name == "progress" and "text" in kwargs:
                kwargs["text"] = tr(kwargs["text"])
            if name == "code" and args and isinstance(args[0], str):
                lines = []
                for line in args[0].splitlines():
                    match = re.match(r"(\[[^\]]+\]\s*(?:[^|]+\|\s*)?)(.*)", line)
                    lines.append(match[1] + tr(match[2]) if match else tr(line))
                args = ("\n".join(lines), *args[1:])
            if name in {"dataframe", "data_editor"} and args and hasattr(args[0], "columns"):
                columns = {key: tr(key) for key in args[0].columns if isinstance(key, str) and tr(key) != key}
                columns.update(kwargs.get("column_config") or {})
                if columns:
                    kwargs["column_config"] = columns
            if name == "plotly_chart":
                args = (localize_figure(args[0]), *args[1:])
            result = target(*args, **kwargs)
            from streamlit.delta_generator import DeltaGenerator
            if isinstance(result, DeltaGenerator):
                return LocalizedUI(result)
            if isinstance(result, (list, tuple)) and all(isinstance(item, DeltaGenerator) for item in result):
                return [LocalizedUI(item) for item in result]
            return result
        return call


def localize(st):
    return st if isinstance(st, LocalizedUI) else LocalizedUI(st)
