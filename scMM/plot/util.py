import os
import re
import glob
import math
import colorsys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def _parse_cluster_id(path: str) -> str:

    base = os.path.basename(path)
    m = re.search(r"(clust\d+)", base, flags=re.IGNORECASE)
    return m.group(1).lower() if m else os.path.splitext(base)[0]


def _ensure_rgb01(color):

    if isinstance(color, str):
        h = color.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    c = tuple(color)
    if max(c) > 1.0:
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
    return c[:3]


def _auto_color_map(cluster_ids):
    cmap = plt.get_cmap("tab10") 
    colors = list(cmap.colors)
    return {cid: colors[i % 10] for i, cid in enumerate(sorted(cluster_ids))}


def load_lion_enrichment_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    col_term = None
    col_name = None
    col_q = None
    col_sig = None
    col_exp = None

    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("term id", "term_id", "termid"):
            col_term = c
        if cl in ("discription", "description", "term", "name", "term name"):
            col_name = c
        if "q-value" in cl or "q value" in cl or cl in ("fdr", "q", "qval", "q_value"):
            col_q = c
        if cl in ("significant", "sig", "hits"):
            col_sig = c
        if cl in ("expected", "exp"):
            col_exp = c

    if col_name is None or col_q is None:
        raise ValueError(f"CSV columns not recognized in {path}. Got: {df.columns.tolist()}")

    out = pd.DataFrame({
        "term_id": df[col_term] if col_term else "",
        "term_name": df[col_name].astype(str),
        "q": pd.to_numeric(df[col_q], errors="coerce"),
    })

    if col_sig is not None:
        out["significant"] = pd.to_numeric(df[col_sig], errors="coerce")
    if col_exp is not None:
        out["expected"] = pd.to_numeric(df[col_exp], errors="coerce")

    out = out.dropna(subset=["q"])
    return out


def build_word_frequencies(
    df: pd.DataFrame,
    top_k: int = 30,
    q_thresh: float = 0.05,
    weight_mode: str = "neglog10q", 
    drop_bracket_code: bool = True, 
):
    
    d = df.copy()

    d = d[d["q"] <= q_thresh].sort_values("q", ascending=True)
    if top_k is not None and top_k > 0:
        d = d.head(top_k)

    if len(d) == 0:
        return {}, {}, d

    def clean_name(s: str) -> str:
        s = str(s)
        if drop_bracket_code:
            s = re.sub(r"\s*\[[^\]]+\]\s*$", "", s) 
        return s.strip()

    d["word"] = d["term_name"].map(clean_name)

    if weight_mode == "neglog10q":
        eps = 1e-300
        w = -np.log10(np.clip(d["q"].to_numpy(dtype=float), eps, 1.0))
    elif weight_mode == "enrich_ratio":
        if "expected" not in d.columns or "significant" not in d.columns:
            raise ValueError("enrich_ratio needs 'Significant' and 'Expected' columns.")
        w = (d["significant"].to_numpy(dtype=float) + 1e-12) / (d["expected"].to_numpy(dtype=float) + 1e-12)
    elif weight_mode == "significant":
        if "significant" not in d.columns:
            raise ValueError("significant mode needs 'Significant' column.")
        w = d["significant"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    if np.allclose(w, w[0]):
        w = w + np.linspace(0, 1e-6, len(w))

    freqs = {wd: float(wt) for wd, wt in zip(d["word"], w)}
    q_map = {wd: float(qv) for wd, qv in zip(d["word"], d["q"])}
    return freqs, q_map, d


def make_q_fade_color_func(base_rgb01, q_map, q_thresh):
    base_rgb01 = _ensure_rgb01(base_rgb01)
    h, l, s = colorsys.rgb_to_hls(*base_rgb01)

    l_dark = 0.35
    l_light = 0.88

    def color_func(word, *args, **kwargs):
        q = q_map.get(word, q_thresh)
        t = min(max(q / max(q_thresh, 1e-12), 0.0), 1.0)  # 0..1
        l2 = l_dark + (l_light - l_dark) * t
        r, g, b = colorsys.hls_to_rgb(h, l2, s)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    return color_func


def render_cluster_wordcloud_svg(
    freqs: dict,
    color_func,
    out_svg: str,
    font_path: str = None,
    width: int = 1600,
    height: int = 1200,
    background_color: str = "white",
    max_words: int = 200,
    random_state: int = 0,
    **kwargs
):
    if not freqs:
        return False

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        prefer_horizontal=0.9,
        random_state=random_state,
        font_path=font_path,
    ).generate_from_frequencies(freqs)
    for spine in plt.gca().spines.values():
        spine.set_visible(kwargs.get("spines_visible", True))

    wc = wc.recolor(color_func=color_func, random_state=random_state)

    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(wc.to_svg(embed_font=True))
    return True

def batch_lion_wordclouds(
    csv_glob: str,
    out_dir: str,
    top_k: int = 30,
    q_thresh: float = 0.05,
    weight_mode: str = "neglog10q",
    cluster_color_map: dict = None, 
    font_path: str = None,
    width: int = 1600,
    height: int = 1200,
    random_state: int = 0,
    save_filtered_tables: bool = True,
    **kwargs
):
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {csv_glob}")

    cluster_ids = [_parse_cluster_id(p) for p in paths]
    if cluster_color_map is None:
        cluster_color_map = _auto_color_map(cluster_ids)
    else:
        cluster_color_map = {k.lower(): _ensure_rgb01(v) for k, v in cluster_color_map.items()}

    summary = []

    for path in paths:
        cid = _parse_cluster_id(path)
        df = load_lion_enrichment_csv(path)

        freqs, q_map, used = build_word_frequencies(
            df, top_k=top_k, q_thresh=q_thresh, weight_mode=weight_mode
        )

        base = cluster_color_map.get(cid, (0.2, 0.2, 0.2))
        color_func = make_q_fade_color_func(base, q_map, q_thresh)

        out_svg = os.path.join(out_dir, f"{cid}_lion_wordcloud.svg")
        ok = render_cluster_wordcloud_svg(
            freqs, color_func, out_svg,
            font_path=font_path, width=width, height=height,
            random_state=random_state,
            **kwargs
        )

        if save_filtered_tables:
            out_csv = os.path.join(out_dir, f"{cid}_lion_filtered_top{top_k}_q{q_thresh}.csv")
            os.makedirs(out_dir, exist_ok=True)
            used.to_csv(out_csv, index=False)

        summary.append({
            "cluster": cid,
            "input_file": os.path.basename(path),
            "n_terms_after_filter": int(len(used)),
            "svg_written": bool(ok),
            "svg_path": out_svg,
        })

    return pd.DataFrame(summary)