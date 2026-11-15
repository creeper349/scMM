"""Per-file frame selection for the data import workflow."""
from pathlib import Path


def render_trim_tool(st, paths, terminal):
    from scMM.file.acquisition import resolve_ms_files
    from scMM.file.trim import frame_extent
    from scMM.webapp import _execute, _parameter

    files = resolve_ms_files(paths)
    saved = st.session_state.setdefault("file_crop_settings", {})
    extents = st.session_state.setdefault("file_crop_extents", {})
    with st.expander("文件裁剪（逐文件时间 / 帧序号）", expanded=False):
        st.caption("先选择输入文件，再加载滑块。拖动两个端点选择保留区间，各文件独立设置，随“开始峰提取”生效。")
        if not files:
            st.info("选择 mzML、mzXML 或 RAW 文件后可配置裁剪。")
            return {}
        signatures = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        if st.button("加载各文件裁剪滑块", key="trim_load"):
            def read(progress):
                result = dict(extents)
                for index, path in enumerate(files):
                    key = str(path)
                    if result.get(key, {}).get("signature") != signatures[key]:
                        result[key] = dict(frame_extent(path), signature=signatures[key])
                    progress((index + 1) / len(files), f"读取时间与帧范围: {index + 1}/{len(files)} files")
                return result
            result = _execute(st, terminal, "读取裁剪范围", read,
                              on_success=lambda result: st.session_state.update(file_crop_extents=result))
            if result is not None:
                extents = result
        selected = {}
        for path in files:
            key = str(path)
            extent = extents.get(key, {})
            if extent.get("signature") != signatures[key]:
                st.caption(f"{path.name} · 点击“加载各文件裁剪滑块”读取范围。")
                continue
            settings = saved.setdefault(key, {})
            if settings.get("signature") != signatures[key]:
                settings.clear()
                settings["signature"] = signatures[key]
            token = f"trim_{key}_{signatures[key]}"
            st.markdown(f"**{path.name}**")
            st.caption(f"{extent['frames']} 帧，{extent['rt_min'] / 60:.3f}–{extent['rt_max'] / 60:.3f} 分钟")
            options = ["全部帧", "按时间", "按帧序号", "时间与帧序号"]
            mode_key = token + "_mode"
            if mode_key not in st.session_state:
                st.session_state[mode_key] = settings.get("mode", "全部帧")
            mode = st.radio("裁剪方式", options, key=mode_key, horizontal=True)
            bounds = {}
            if mode in {"按时间", "时间与帧序号"}:
                lower, upper = extent["rt_min"] / 60, extent["rt_max"] / 60
                slider_key = token + "_rt"
                if lower < upper:
                    if slider_key not in st.session_state:
                        st.session_state[slider_key] = tuple(settings.get("rt_minutes", (lower, upper)))
                    window = st.slider("保留时间区间（分钟）", lower, upper,
                                       step=max((upper - lower) / 10000, .000001),
                                       format="%.4f", key=slider_key,
                                       help=_parameter("trim.rt_range", None)[1])
                else:
                    window = (lower, upper)
                    st.caption(f"时间范围为 {lower:.4f} 分钟。")
                settings["rt_minutes"] = list(window)
                bounds["rt_range"] = [window[0] * 60, window[1] * 60]
            if mode in {"按帧序号", "时间与帧序号"}:
                last = extent["frames"] - 1
                slider_key = token + "_frames"
                if last > 0:
                    if slider_key not in st.session_state:
                        st.session_state[slider_key] = tuple(settings.get("frames", (0, last)))
                    window = st.slider("保留原始帧区间（从 0 开始）", 0, last, step=1, key=slider_key,
                                       help=_parameter("trim.frame_range", None)[1])
                else:
                    window = (0, 0)
                    st.caption("文件仅包含第 0 帧。")
                settings["frames"] = list(window)
                bounds["frame_range"] = list(window)
            settings["mode"] = mode
            if bounds:
                selected[key] = bounds
            st.divider()
        st.caption("区间包含两个端点；时间与帧序号同时启用时取交集。帧序号对应原始文件。")
    return selected
