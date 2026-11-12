"""Shared grouped-set controls for quality control and compound annotation."""
import io

import matplotlib as mpl
import numpy as np
import pandas as pd

from scMM.analysis.group_sets import feature_membership, compound_records, annotation_membership, intersection_counts, annotation_isotope_mask
from scMM.plot.group_sets import group_set_figure


def _figure_export(st, builder, key, *, context=(), description=""):
    from scMM.webapp import _save_image, _image_filename
    if not st.button("Matplotlib 绘图参数与导出", key=f"{key}_export"):
        return

    @st.dialog("绘图参数、预览与保存", width="large")
    def show():
        controls, preview = st.columns([1, 1.2])
        with controls:
            a, b = st.columns(2)
            fmt = a.selectbox("格式", ["svg", "png"], key=f"{key}_format")
            width = a.number_input("宽度（inch）", 3., 30., 9., key=f"{key}_width")
            height = b.number_input("高度（inch）", 3., 30., 6., key=f"{key}_height")
            size = a.number_input("字号", 6., 30., 10., key=f"{key}_font_size")
            family = b.selectbox("字体", ["Microsoft YaHei", "Arial", "SimHei", "Times New Roman"], key=f"{key}_family")
            cmap = a.selectbox("cmap", ["tab10", "tab20", "Set2", "viridis", "plasma"], key=f"{key}_cmap")
            alpha = b.number_input("透明度", .05, 1., .6, key=f"{key}_alpha")
            line = a.number_input("线宽（pt）", .1, 10., 1., key=f"{key}_line")
            dpi = b.number_input("DPI", 72, 1200, 300, key=f"{key}_dpi")
            parameters = dict(width=width, height=height, font_size=size, font_family=family, cmap=cmap,
                              alpha=alpha, line_width=line)
            signature = repr((parameters, dpi, fmt, context))
            if st.button("预览", key=f"{key}_preview"):
                figure = builder(**parameters)
                with mpl.rc_context({"svg.fonttype": "none"}):
                    png, payload = io.BytesIO(), io.BytesIO()
                    figure.savefig(png, format="png", dpi=dpi, bbox_inches="tight")
                    figure.savefig(payload, format=fmt, dpi=dpi, bbox_inches="tight")
                st.session_state[f"{key}_preview_data"] = dict(signature=signature, png=png.getvalue(), payload=payload.getvalue())
            cache = st.session_state.get(f"{key}_preview_data", {})
            current = cache.get("signature") == signature
            if st.button("下载", disabled=not current, key=f"{key}_download"):
                _save_image(st, cache["payload"], _image_filename(key, fmt, title=description))
        if current:
            preview.image(cache["png"], width="stretch")
    show()


def render_group_sets(st, data, *, key, annotation=None, composition=False):
    from scMM.webapp import _parameter, _choice
    st.subheader("分组集合关系与物质组成" if composition else "分组特征 / 注释集合关系")
    if not len(data.obs.columns):
        st.info("添加 obs 分组信息后可绘制集合关系图。")
        return
    a, b, c = st.columns(3)
    keys = data.obs.columns.tolist()
    default = keys.index("source_file") if "source_file" in keys else 0
    group_key = a.selectbox("obs 分组键", keys, index=default, key=f"{key}_group")
    groups = data.obs[group_key].astype("string").dropna().unique().tolist()
    selected = b.multiselect("展示组别", groups, default=groups[:3], key=f"{key}_groups")
    index, help_text = _choice("sets.kind", ["UpSet", "Venn"])
    kind = c.selectbox("集合关系图", ["UpSet", "Venn"], index=index, help=help_text, key=f"{key}_kind")
    mode = a.selectbox("集合内容", ["特征", "注释物质"], index=1 if composition and annotation is not None else 0, key=f"{key}_content")
    layers = list(dict.fromkeys(["X", *data.layers.keys()]))
    layer = b.selectbox("检出判定数据层", layers, index=layers.index("raw") if "raw" in layers else 0, key=f"{key}_layer",
                        help="默认按原始 raw 强度判定检出，使用当前有效细胞和有效特征。")
    value, help_text = _parameter("sets.intensity", 0.)
    intensity = c.number_input("检出强度下限", 0., 1.e15, float(value), help=help_text, key=f"{key}_intensity")
    value, help_text = _parameter("sets.detection_rate", 0.)
    rate = a.number_input("组内最低检出率", 0., 1., float(value), step=.01, help=help_text, key=f"{key}_rate")
    value, help_text = _parameter("sets.top", 30)
    top = b.number_input("UpSet 交集排名上限", 1, 500, int(value), help=help_text, key=f"{key}_top")
    source = name_key = category_key = None
    accepted, split = True, True
    var_keys = data.var.columns.tolist()
    available = annotation is not None or bool(var_keys)
    default_composition = annotation is not None or any(name in var_keys for name in ["annotation", "manual_annotation", "name"])
    show_composition = composition and st.checkbox("展示各组物质种类组成", value=default_composition, disabled=not available, key=f"{key}_show_composition")
    if mode == "注释物质" or show_composition:
        sources = ["var 注释列"] if var_keys else []
        if annotation is not None:
            sources.append("注释候选表")
        if not sources:
            st.info("先将注释结果写入 var，可在此选择对应键名进行比较。")
            return
        preferred = ["annotation", "manual_annotation", "identification", "compound_name", "name"]
        source_index = sources.index("注释候选表") if annotation is not None and not any(name in var_keys for name in preferred) else 0
        source = a.selectbox("注释数据来源", sources, index=source_index, key=f"{key}_annotation_source",
                             help="选择 var 注释列后，可指定表示注释结果的任意 var 键。")
        if source == "注释候选表":
            accepted = b.checkbox("仅统计已接受注释", True, key=f"{key}_accepted")
        else:
            default_name = next((name for name in preferred if name in var_keys), var_keys[0]) if var_keys else None
            if not var_keys:
                st.info("先将注释名称写入 var，可在此按分组展示物质集合。")
                return
            name_key = b.selectbox("表示注释结果的 var 键", var_keys, index=var_keys.index(default_name), key=f"{key}_name",
                                   help="使用所选 var 列的值作为物质名称或标识，同名结果在每组中去重，空值跳过。")
            category_key = c.selectbox("物质类别 var 键", ["未指定", *var_keys], key=f"{key}_category")
            category_key = None if category_key == "未指定" else category_key
            split = b.checkbox("按 / 拆分同一特征的多个名称", True, key=f"{key}_split")
    labels = pd.Categorical(data.obs[group_key].astype("string"), categories=selected).codes.tobytes()
    isotope_flags = annotation_isotope_mask(data)
    masks = (np.packbits(data.cell_keep_mask()).tobytes(), np.packbits(data.feature_keep_mask()).tobytes(),
             np.packbits(isotope_flags.astype(bool)).tobytes())
    use_table = source == "注释候选表"
    accepted_flags = np.packbits(annotation["table"]["accepted"]).tobytes() if use_table else b""
    settings = (id(data), group_key, tuple(selected), layer, intensity, rate, mode, source, name_key, category_key,
                accepted, split, show_composition, labels, masks, accepted_flags,
                id(annotation["table"]) if use_table else None)
    if st.button("计算分组集合", key=f"{key}_run", disabled=not selected):
        with st.spinner("统计各组检出特征与物质…"):
            features = feature_membership(data, group_key, selected, layer=layer, intensity_threshold=intensity, detection_rate=rate)
            members, counts = features, pd.DataFrame()
            if source is not None:
                records = compound_records(data, table=annotation["table"] if use_table else None,
                                           name_key=name_key, category_key=category_key, accepted_only=accepted, split_names=split)
                compounds, counts = annotation_membership(features, records)
                if mode == "注释物质":
                    members = compounds
            generation = st.session_state.get(f"{key}_generation", 0) + 1
            st.session_state[f"{key}_generation"] = generation
            st.session_state[f"{key}_result"] = dict(settings=settings, membership=members, counts=counts, generation=generation)
    result = st.session_state.get(f"{key}_result", {})
    if result.get("settings") != settings:
        return
    membership, counts = result["membership"], result["counts"]
    if membership.empty:
        st.info("当前条件下没有检出的集合成员，可调整检出阈值或注释来源。")
    elif kind == "Venn" and len(selected) not in (2, 3):
        st.info("Venn 图支持 2 或 3 组，请调整展示组别。")
    elif kind == "UpSet" and len(selected) < 2:
        st.info("请选择至少两个组别以绘制 UpSet 集合关系。")
    else:
        builder = lambda **params: group_set_figure(membership, kind=kind, top=top, **params)
        st.pyplot(builder(), use_container_width=True)
        _figure_export(st, builder, f"{key}_{kind}", context=(result["generation"], top), description=f"{group_key}_{mode}_{'-'.join(selected)}")
        st.dataframe(pd.DataFrame({"组别": membership.columns, "集合大小": membership.sum().to_numpy()}), hide_index=True)
        st.download_button("下载集合成员 CSV", membership.to_csv().encode("utf-8-sig"), f"SCMM_{key}_membership.csv", key=f"{key}_members_csv")
        st.dataframe(intersection_counts(membership).head(top), hide_index=True, use_container_width=True)
    if show_composition and not counts.empty:
        percent = st.checkbox("物质种类组成显示百分比", True, key=f"{key}_percent")
        builder = lambda **params: group_set_figure(membership, kind="composition", composition=counts, percentages=percent, **params)
        st.pyplot(builder(), use_container_width=True)
        _figure_export(st, builder, f"{key}_composition", context=(result["generation"], percent), description=f"{group_key}_{'-'.join(selected)}")
        st.dataframe(counts, use_container_width=True)
        st.download_button("下载分组物质类别 CSV", counts.to_csv().encode("utf-8-sig"), f"SCMM_{key}_categories.csv", key=f"{key}_categories_csv")
        st.caption("每组按物质名称去重，同一物质的不同类别归属分别计数。")
