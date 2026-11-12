"""Feature detection and annotated-compound membership across obs groups."""
import numpy as np
import pandas as pd
from scipy import sparse


def feature_membership(data, group_key, groups, *, layer="raw", intensity_threshold=0., detection_rate=0.):
    matrix = data.X if layer == "X" else data.layers.get(layer, data.X)
    labels = data.obs[group_key].astype("string")
    cells, features = data.cell_keep_mask(), data.feature_keep_mask()
    result = pd.DataFrame(False, index=data.var_names[features], columns=list(groups))
    for group in groups:
        mask = cells & labels.eq(group).fillna(False).to_numpy(dtype=bool)
        values = matrix[np.ix_(mask, features)]
        if not values.shape[0]:
            continue
        if sparse.issparse(values):
            values.data[~np.isfinite(values.data)] = 0
            counts = np.asarray((values > intensity_threshold).sum(axis=0)).ravel()
        else:
            values = np.asarray(values)
            counts = np.count_nonzero(np.isfinite(values) & (values > intensity_threshold), axis=0)
        result[group] = (counts > 0) & (counts / values.shape[0] >= detection_rate)
    return result.loc[result.any(axis=1)]


def annotation_isotope_mask(data):
    if "annotation_isotope" in data.var:
        flags = data.var["annotation_isotope"]
        return (pd.to_numeric(flags, errors="coerce").fillna(0).ne(0)
                | flags.astype("string").str.strip().str.casefold().isin(["true", "isotope"]))
    if "deisotope_role" in data.var:
        return data.var["deisotope_role"].astype("string").str.casefold().eq("isotope").fillna(False)
    return pd.to_numeric(data.var.get("isotope_order", pd.Series(0, index=data.var_names)), errors="coerce").fillna(0).gt(0)


def compound_records(data, *, table=None, name_key=None, category_key=None, accepted_only=True, split_names=True):
    if table is not None:
        records = table.loc[~table["is_isotope"].astype(bool) & table["match_level"].isin(["MS1", "MS/MS"])].copy()
        if accepted_only:
            records = records.loc[records["accepted"].astype(bool)]
        records = records[["feature_id", "name", "category"]]
    else:
        records = pd.DataFrame({"feature_id": data.var_names, "name": data.var[name_key].astype("string").fillna("").to_numpy(),
                                "category": data.var[category_key].astype("string").fillna("").to_numpy() if category_key else ""})
        if split_names:
            records["name"] = records["name"].str.split("/")
            records = records.explode("name")
    flags = annotation_isotope_mask(data)
    records = records.loc[~records["feature_id"].map(flags).fillna(False)].copy()
    records["name"] = records["name"].fillna("").astype(str).str.strip()
    records["category"] = records["category"].fillna("").astype(str).str.strip().replace("", "Unclassified")
    empty_names = {"", "nan", "none", "null", "<na>", "n/a", "-", "unknown", "unidentified", "unannotated",
                   "未注释", "未鉴定", "未匹配", "isotope", "同位素"}
    return records.loc[~records["name"].str.casefold().isin(empty_names)].drop_duplicates()


def annotation_membership(features, records):
    records = records.loc[records["feature_id"].isin(features.index)]
    names = pd.Index(records["name"].unique(), name="compound")
    membership = pd.DataFrame(False, index=names, columns=features.columns)
    composition = {}
    for group in features:
        detected = features.index[features[group]]
        present = records.loc[records["feature_id"].isin(detected)]
        membership.loc[present["name"].unique(), group] = True
        composition[group] = present.drop_duplicates(["name", "category"])["category"].value_counts()
    counts = pd.DataFrame(composition).fillna(0).astype(int).T
    return membership.loc[membership.any(axis=1)], counts


def intersection_counts(membership):
    counts = membership.value_counts(sort=True).rename("count")
    return counts.reset_index()
