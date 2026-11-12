"""MS1 candidates, isotope evidence and precursor-selected MS2 library matching."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms
from scMM.util.lipids import normalize_lipid_name, report_lipid_name

from scMM.util.annotation import (
    ADDUCT_ALIASES, DEFAULT_ADDUCTS_NEG, DEFAULT_ADDUCTS_POS, load_lipidmaps_sdf,
)

ADDUCTS = {**DEFAULT_ADDUCTS_POS, **DEFAULT_ADDUCTS_NEG}
ION_FORMULAS = {
    "[M]+": ("", ""), "[M+H]+": ("H", ""), "[M+Na]+": ("Na", ""),
    "[M+K]+": ("K", ""), "[M+NH4]+": ("NH4", ""),
    "[M-H2O+H]+": ("H", "H2O"), "[M+Li]+": ("Li", ""),
    "[M-H]-": ("", "H"), "[M+HCOO]-": ("CHO2", ""),
    "[M+CH3COO]-": ("C2H3O2", ""), "[M+Cl]-": ("Cl", ""),
}


def canonical_adduct(value):
    value = str(value).replace(" ", "")
    if value in ADDUCT_ALIASES:
        return ADDUCT_ALIASES[value]
    if value in ADDUCTS:
        return value
    for name in ADDUCTS:
        if value == name.replace("[", "").replace("]", "")[:-1]:
            return name
    return value


def _key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _empirical_formula(formula):
    """Keep fixed-isotope counts when translating library formulas to OpenMS notation."""
    formula = re.sub(r"\[(\d+)([A-Z][a-z]?)\]", r"(\1)\2", str(formula).strip())
    return oms.EmpiricalFormula(formula)


def _records_from_spectra(path):
    records, record, peaks = [], {}, []

    def finish():
        if record:
            records.append({**record, "peaks": list(peaks)})
        record.clear()
        peaks.clear()

    for raw in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() in {"END IONS", "BEGIN IONS"}:
            if line.upper() == "END IONS":
                finish()
            continue
        if line.lower().startswith("name:") and record:
            finish()
        peak = re.match(r"^([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s+([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)", line)
        if peak:
            peaks.append([float(peak[1]), float(peak[2])])
        else:
            parts = re.split(r"[:=]", line, maxsplit=1)
            if len(parts) == 2:
                record[parts[0].strip()] = parts[1].strip().strip('"')
    finish()
    return records


def _peaks(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty((0, 2))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [re.split(r"[:\s]+", part.strip())[:2] for part in value.split(";") if part.strip()]
    array = np.asarray(value, dtype=float).reshape(-1, 2)
    array = array[np.isfinite(array).all(axis=1) & (array[:, 0] > 0) & (array[:, 1] > 0)]
    return array[np.argsort(array[:, 0])]


def load_annotation_database(path):
    """Read neutral-mass CSV/TSV/SDF or precursor-mass MSP/MGF/JSON libraries."""
    suffix = Path(path).suffix.lower()
    if suffix == ".sdf":
        records = load_lipidmaps_sdf(str(path)).to_dict("records")
    elif suffix in {".msp", ".mgf"}:
        records = _records_from_spectra(path)
    elif suffix == ".json":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        records = payload.get("compounds", payload.get("spectra", [])) if isinstance(payload, dict) else payload
    else:
        records = pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",").fillna("").to_dict("records")
    rows, spectra = [], {}
    for i, source in enumerate(records):
        fields = {_key(k): v for k, v in source.items()}

        def take(*names, default=""):
            return next((fields[_key(n)] for n in names if _key(n) in fields and str(fields[_key(n)]) not in {"", "nan"}), default)

        formula = str(take("formula", "molecular formula", "sum formula"))
        mass = take("neutral_mass", "exact_mass", "monoisotopic_mass")
        precursor = take("precursor_mz", "precursormz", "pepmass", "theoretical_mz")
        adduct = canonical_adduct(take("adduct", "precursortype", "iontype"))
        mass = float(mass) if mass != "" else (_empirical_formula(formula).getMonoWeight() if formula else np.nan)
        precursor = float(str(precursor).split()[0]) if precursor != "" else np.nan
        if np.isfinite(precursor) and not adduct:
            raise ValueError(f"{Path(path).name} 第 {i + 1} 条记录指定 precursor_mz 时需同时指定 adduct。")
        if not np.isfinite(mass) and not np.isfinite(precursor):
            raise ValueError(f"{Path(path).name} 第 {i + 1} 条记录需要 formula、neutral_mass 或 precursor_mz。")
        library_id = f"lib{i}"
        spectra[library_id] = _peaks(take("peaks", "spectrum", "msms", default=[]))
        polarity = str(take("polarity", "ionmode", "mode")).lower()
        name = str(take("name", "common_name", "abbreviation", "title", "record_title", default=f"compound_{i+1}"))
        embedded_id = re.search(r"\bLM[A-Z]{2}\d+\b", name)
        rows.append(dict(library_id=library_id, name=str(take("name", "common_name", "abbreviation", "title", "record_title", default=f"compound_{i+1}")),
                         source_library=Path(path).name,
                         abbreviation=str(take("abbreviation")),
                         formula=formula, neutral_mass=mass, precursor_mz=precursor, adduct=adduct,
                         category=str(take("category", "compoundclass", "class", "main_class")),
                         polarity="neg" if polarity.startswith(("neg", "-")) else "pos" if polarity.startswith(("pos", "+")) else "",
                         database_id=str(take("id", "lm_id", "database_id", default=embedded_id[0] if embedded_id else library_id))))
    return pd.DataFrame(rows), spectra


def deisotope_features(data, **params):
    """Run the existing CyESIData implementation on m/z-labelled features, retaining X."""
    from scMM.file.data import CyESIData
    mz = data.var["mz"].to_numpy(dtype=float)
    matrix = data.layers.get("raw", data.X)[data.cell_keep_mask()]
    working = CyESIData(X=matrix, var=pd.DataFrame(index=mz.astype(str)))
    result = working.deisotope(remove=False, inplace=False, merge_mode="keep_parent", **params)
    metadata = result["feature_meta"].copy()
    mapping = dict(zip(mz.astype(str), data.var_names.astype(str)))
    for column in metadata:
        if column in {"mz"}:
            continue
        values = metadata[column].to_numpy()
        if column == "isotope_parent" or re.fullmatch(r"M\d+_feature", column):
            values = [mapping.get(str(v), "") for v in values]
        elif column == "isotope_order":
            values = pd.to_numeric(metadata[column]).fillna(0).astype(np.uint8).to_numpy()
        elif metadata[column].dtype == object:
            values = metadata[column].fillna("").astype(str).to_numpy()
        data.var[column] = values
    data.var["annotation_isotope"] = (metadata["deisotope_role"].to_numpy() == "isotope").astype(np.uint8)
    if "feature_keep_annotation_isotope" in data.var:
        data.var["feature_keep_annotation_isotope"] = (1 - data.var["annotation_isotope"]).astype(np.uint8)
    data.uns["annotation_deisotope_params"] = result["params"]
    sync_isotope_flags(data)
    return result["final_table"]


def sync_isotope_flags(data):
    """Reflect isotope changes immediately; restored features are searched on refresh."""
    if "annotation" not in data.uns:
        return
    table = data.uns["annotation"]["table"]
    flags = table["feature_id"].map(data.var["annotation_isotope"].astype(bool)).fillna(False)
    restored = table["is_isotope"] & ~flags
    table["is_isotope"] = flags.to_numpy(dtype=bool)
    table.loc[flags | restored, "accepted"] = False
    table.loc[flags, "match_level"] = "isotope"
    table.loc[restored, "match_level"] = "unmatched"
    data.uns["annotation"]["table"] = table.sort_values(
        ["is_isotope", "observed_mz", "feature_id", "combined_score", "abs_ppm_error"],
        ascending=[True, True, True, False, True], kind="stable")


@lru_cache(maxsize=4096)
def theoretical_isotopes(formula, adduct, count):
    if not formula:
        return None
    empirical = _empirical_formula(formula)
    add, subtract = ION_FORMULAS[adduct]
    if add:
        empirical += oms.EmpiricalFormula(add)
    if subtract:
        empirical -= oms.EmpiricalFormula(subtract)
    return np.array([p.getIntensity() for p in empirical.getIsotopeDistribution(oms.CoarseIsotopePatternGenerator(count)).getContainer()])


def isotope_similarity(data, position, formula, adduct):
    if not formula or "isotope_parent" not in data.var:
        return np.nan
    children = data.var[(data.var["isotope_parent"].astype(str) == str(data.var_names[position])) & data.var["annotation_isotope"].astype(bool)]
    if children.empty:
        return np.nan
    count = max(int(children["isotope_order"].max()) + 1,
                int(data.uns.get("annotation_deisotope_params", {}).get("max_isotope_order", 3)) + 1)
    observed = np.zeros(count)
    observed[0] = 1
    for _, row in children.iterrows():
        observed[int(row["isotope_order"])] += float(row["isotope_slope_A"])
    expected = theoretical_isotopes(formula, adduct, count)
    return float(np.dot(observed, expected) / (np.linalg.norm(observed) * np.linalg.norm(expected)))


def merge_centroids(peaks, ppm):
    peaks = _peaks(peaks)
    if not len(peaks):
        return peaks
    mz, intensity, total, output = peaks[0, 0], peaks[0, 1], peaks[0, 1], []
    weighted = mz * intensity
    for m, y in peaks[1:]:
        if (m - mz) / mz * 1e6 <= ppm:
            weighted += m * y
            total += y
            mz = weighted / total
        else:
            output.append((mz, total))
            mz, total, weighted = m, y, m * y
    output.append((mz, total))
    return np.asarray(output)


def collect_msms(paths, targets, *, precursor_ppm=10., fragment_ppm=20., profile_step=.005,
                 baseline_width=1., snr=0., min_relative_intensity=.01, mz_min=30., mz_max=2000., progress=None):
    """Accumulate only MS2 scans with a selected precursor; retain source-native scan IDs."""
    from scMM.file.io import load_single_file
    report = progress or (lambda *_: None)
    target_ids = list(targets)
    masses = np.array([targets[k] for k in target_ids])
    order = np.argsort(masses)
    masses = masses[order]
    target_ids = [target_ids[i] for i in order]
    grid = np.arange(mz_min, mz_max + profile_step / 2, profile_step)
    accumulated, skipped, total = {}, 0, 0
    for file_i, path in enumerate(paths):
        report(.85 * file_i / max(1, len(paths)), f"读取 MS/MS 文件 {file_i+1}/{len(paths)}: {Path(path).name}")
        experiment, _ = load_single_file(str(path), format="auto", centroid_ms2=True)
        count = experiment.getNrSpectra()
        for scan_i, spectrum in enumerate(experiment):
            if spectrum.getMSLevel() != 2:
                continue
            total += 1
            precursors = spectrum.getPrecursors()
            if not precursors:
                skipped += 1
                continue
            selected = set()
            for precursor in precursors:
                q = precursor.getMZ()
                lo = np.searchsorted(masses, q / (1 + precursor_ppm * 1e-6), side="left")
                hi = np.searchsorted(masses, q / (1 - precursor_ppm * 1e-6), side="right")
                selected.update(target_ids[lo:hi])
            if not selected:
                continue
            mz, intensity = spectrum.get_peaks()
            good = np.isfinite(mz) & np.isfinite(intensity) & (mz >= mz_min) & (mz <= mz_max)
            mz, intensity = mz[good], np.maximum(intensity[good], 0)
            native_id = spectrum.getNativeID() or str(scan_i + 1)
            scan_name = f"{Path(path).name}:{native_id}"
            spectrum_type = spectrum.getType()
            if spectrum_type == oms.SpectrumSettings.SpectrumType.UNKNOWN:
                probe = oms.MSSpectrum()
                probe.set_peaks((mz, intensity - intensity.min() if len(intensity) else intensity))
                spectrum_type = oms.PeakTypeEstimator().estimateType(probe)
            profile = spectrum_type == oms.SpectrumSettings.SpectrumType.PROFILE
            for feature_id in selected:
                entry = accumulated.setdefault(feature_id, {"centroid": np.empty((0, 2)), "profile": None, "scans": []})
                entry["scans"].append(scan_name)
                if profile and len(mz) > 1:
                    if entry["profile"] is None:
                        entry["profile"] = np.zeros(len(grid), dtype=np.float32)
                    entry["profile"] += np.interp(grid, mz, intensity, left=0, right=0).astype(np.float32)
                elif len(mz):
                    entry["centroid"] = merge_centroids(np.vstack([entry["centroid"], np.column_stack([mz, intensity])]), fragment_ppm)
            if scan_i % 100 == 0:
                report(.85 * (file_i + (scan_i + 1) / max(1, count)) / len(paths), f"匹配前体扫描 {scan_i+1}/{count}")
        del experiment
    output = {}
    for i, (feature_id, entry) in enumerate(accumulated.items()):
        peaks = entry["centroid"]
        if entry["profile"] is not None:
            summed = oms.MSSpectrum()
            summed.setMSLevel(2)
            summed.set_peaks((grid, entry.pop("profile")))
            if baseline_width > 0:
                baseline = oms.MorphologicalFilter()
                params = baseline.getParameters()
                params.setValue("struc_elem_length", float(baseline_width))
                baseline.setParameters(params)
                baseline.filter(summed)
            picker = oms.PeakPickerHiRes()
            params = picker.getParameters()
            params.setValue("signal_to_noise", float(snr))
            picker.setParameters(params)
            picked = oms.MSSpectrum()
            picker.pick(summed, picked)
            peaks = merge_centroids(np.vstack([peaks, np.column_stack(picked.get_peaks())]), fragment_ppm)
        if len(peaks):
            peaks = peaks[peaks[:, 1] >= peaks[:, 1].max() * min_relative_intensity]
        output[feature_id] = {"peaks": peaks, "scans": entry["scans"]}
        report(.85 + .15 * (i + 1) / max(1, len(accumulated)), f"叠加、去基线与质心化 {i+1}/{len(accumulated)}")
    report(1., f"MS/MS 扫描完成：MS2={total}，缺少前体信息={skipped}，命中特征={len(output)}")
    return output


def match_fragments(observed, reference, ppm=20.):
    """One-to-one nearest mass matches; square-root-intensity cosine including unmatched ions."""
    observed, reference = _peaks(observed), _peaks(reference)
    if not len(observed) or not len(reference):
        return np.nan, []
    pairs = []
    for j, (mz, _) in enumerate(reference):
        lo, hi = np.searchsorted(observed[:, 0], [mz * (1 - ppm * 1e-6), mz * (1 + ppm * 1e-6)])
        for i in range(lo, hi):
            pairs.append((abs(observed[i, 0] - mz), i, j))
    used_i, used_j, matched = set(), set(), []
    for _, i, j in sorted(pairs):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            matched.append(dict(observed_mz=float(observed[i, 0]), reference_mz=float(reference[j, 0]),
                                ppm_error=float((observed[i, 0] / reference[j, 0] - 1) * 1e6),
                                observed_intensity=float(observed[i, 1]), reference_intensity=float(reference[j, 1])))
    score = sum(np.sqrt(m["observed_intensity"] * m["reference_intensity"]) for m in matched)
    score /= np.sqrt(observed[:, 1].sum() * reference[:, 1].sum())
    return float(score), matched


def rank_annotations(table):
    return table.sort_values(["is_isotope", "observed_mz", "feature_id", "combined_score", "abs_ppm_error"],
                             ascending=[True, True, True, False, True], kind="stable").reset_index(drop=True)


def annotate_features(data, database, reference_spectra, *, adducts, mode="pos", ppm=5.,
                      measured=None, fragment_ppm=20., min_msms_score=.5, min_matches=3,
                      isotope_weight=.5, msms_weight=.5, progress=None):
    report = progress or (lambda *_: None)
    measured = measured or {}
    iso_mask = data.var.get("annotation_isotope", pd.Series(0, index=data.var_names)).astype(bool).to_numpy()
    # An isotope filter created on this page remains reversible during a refresh.
    other_filters = [k for k in data.uns.get("active_feature_filters", data.feature_filter_columns()) if k != "feature_keep_annotation_isotope"]
    eligible = np.ones(data.n_vars, dtype=bool)
    for key in other_filters:
        eligible &= data.var[key].to_numpy(dtype=bool)
    selected = [a for a in adducts if mode == "both" or a in (DEFAULT_ADDUCTS_POS if mode == "pos" else DEFAULT_ADDUCTS_NEG)]
    database = database.copy()
    if "source_library" not in database:
        database["source_library"] = ""
    if "abbreviation" not in database:
        database["abbreviation"] = ""
    # Connect MGF names containing LM IDs to the curated LMSD sum abbreviations.
    descriptions = database[database["abbreviation"].fillna("").ne("")].drop_duplicates("database_id").set_index("database_id")
    for column in ("abbreviation", "category"):
        mapped = database["database_id"].map(descriptions[column])
        database[column] = database[column].mask(database[column].fillna("").eq(""), mapped).fillna("")
    expanded = []
    for _, record in database.iterrows():
        if record["polarity"] and mode != "both" and record["polarity"] != mode:
            continue
        for adduct in ([record["adduct"]] if record["adduct"] else selected):
            if adduct not in selected:
                continue
            if record["polarity"] and adduct not in (DEFAULT_ADDUCTS_POS if record["polarity"] == "pos" else DEFAULT_ADDUCTS_NEG):
                continue
            theoretical = (record["neutral_mass"] + ADDUCTS[adduct]["mass_shift"]
                           if np.isfinite(record["neutral_mass"]) else record["precursor_mz"])
            expanded.append({**record, "adduct": adduct, "theoretical_mz": theoretical})
    ions = pd.DataFrame(expanded)
    if not ions.empty:
        ions = ions.sort_values("theoretical_mz").reset_index(drop=True)
    ion_masses = ions["theoretical_mz"].to_numpy() if not ions.empty else np.array([])
    output, cache = [], {"observed": {}, "reference": {}}
    for position, (feature_id, feature) in enumerate(data.var.iterrows()):
        feature_id, mz = str(feature_id), float(feature["mz"])
        observation = measured.get(feature_id, {"peaks": np.empty((0, 2)), "scans": []})
        observed = observation["peaks"]
        spectrum_key = f"f{position}"
        cache["observed"][spectrum_key] = observed
        base = dict(feature_id=feature_id, observed_mz=mz, theoretical_mz=np.nan, ppm_error=np.nan,
                    abs_ppm_error=np.nan, library_precursor_mz=np.nan, adduct="", name="", formula="", category="", database_id="", abbreviation="", source_library="",
                    isotope_similarity=np.nan, has_msms=bool(observation["scans"]),
                    msms_scan_number="; ".join(observation["scans"]), matched_fragments="[]",
                    msms_similarity=np.nan, n_matched_fragments=0, combined_score=0.,
                    reference_msms_available=False, msms_status="无参考 MS/MS",
                    match_level="isotope" if iso_mask[position] else "unmatched" if eligible[position] else "filtered",
                    is_isotope=bool(iso_mask[position]), accepted=False, library_id="", spectrum_key=spectrum_key)
        candidates = []
        if not iso_mask[position] and eligible[position]:
            lo = np.searchsorted(ion_masses, mz / (1 + ppm * 1e-6))
            hi = np.searchsorted(ion_masses, mz / (1 - ppm * 1e-6), side="right")
            for _, ion in ions.iloc[lo:hi].iterrows():
                library_id = ion["library_id"]
                reference = reference_spectra.get(library_id, np.empty((0, 2)))
                score, fragments = match_fragments(observed, reference, fragment_ppm)
                iso_score = isotope_similarity(data, position, ion["formula"], ion["adduct"])
                weights = [(iso_score, isotope_weight), (score, msms_weight)]
                denominator = sum(w for s, w in weights if np.isfinite(s))
                combined = sum(s*w for s, w in weights if np.isfinite(s)) / denominator if denominator else 0.
                row = {**base, **{key: ion[key] for key in ["name", "formula", "category", "adduct", "database_id", "theoretical_mz", "library_id", "abbreviation", "source_library"]},
                       "isotope_similarity": iso_score, "msms_similarity": score,
                       "library_precursor_mz": ion["precursor_mz"],
                       "reference_msms_available": bool(len(reference)),
                       "msms_status": "无参考 MS/MS" if not len(reference) else "无实测 MS/MS" if not len(observed) else "匹配未达阈值",
                       "n_matched_fragments": len(fragments), "matched_fragments": json.dumps(fragments),
                       "combined_score": combined, "match_level": "MS/MS" if np.isfinite(score) and score >= min_msms_score and len(fragments) >= min_matches else "MS1"}
                # Spectral-only entries need fragment evidence to become candidates.
                if not np.isfinite(ion["neutral_mass"]) and row["match_level"] != "MS/MS":
                    continue
                if row["match_level"] == "MS/MS":
                    row["msms_status"] = "MS/MS 匹配"
                row["ppm_error"] = (mz / ion["theoretical_mz"] - 1) * 1e6
                row["abs_ppm_error"] = abs(row["ppm_error"])
                cache["reference"][library_id] = reference
                candidates.append(row)
        output.extend(candidates or [base])
        if position % 25 == 0 or position + 1 == data.n_vars:
            report((position + 1) / data.n_vars, f"物质注释 {position+1}/{data.n_vars}")
    table = rank_annotations(pd.DataFrame(output))
    table.index = pd.Index([f"a{i}" for i in range(len(table))], name="annotation_id")
    table = standardize_lipid_annotations(table)
    data.uns["annotation"] = {"table": table, "spectra": cache, "name_policy": "regular_lipids_v2",
                              "parameters": dict(mode=mode, ppm=ppm, adducts=list(selected), fragment_ppm=fragment_ppm,
                                                 min_msms_score=min_msms_score, min_matches=min_matches,
                                                 isotope_weight=isotope_weight, msms_weight=msms_weight)}
    return table


def accept_annotations(table, first=False):
    table["accepted"] = False
    valid = table["name"].str.strip().ne("") & ~table["is_isotope"] & table["match_level"].isin(["MS1", "MS/MS"])
    eligible = table.loc[valid]
    if first:
        eligible = eligible.drop_duplicates("feature_id")
    table.loc[eligible.index, "accepted"] = True


@lru_cache(maxsize=65536)
def _formula_identity(formula):
    if not formula:
        return ""
    try:
        return _empirical_formula(formula).toString()
    except RuntimeError:
        return formula


@lru_cache(maxsize=65536)
def _fixed_isotope_suffix(formula):
    try:
        composition = _empirical_formula(formula).getElementalComposition()
    except RuntimeError:
        return ""
    tokens = []
    for element, count in composition.items():
        match = re.fullmatch(r"\((\d+)\)([A-Z][a-z]?)", element.decode() if isinstance(element, bytes) else element)
        if match and count:
            tokens.append(f"[{match[1]}]{match[2]}{count if count > 1 else ''}")
    return "[M" + "".join(sorted(tokens)) + "]" if tokens else ""


def _spectral_rank(row):
    score = row.get("msms_similarity", np.nan)
    return (float(score) if pd.notna(score) else -1., int(row.get("n_matched_fragments", 0)),
            float(row.get("combined_score", 0)), -float(row.get("abs_ppm_error", np.inf)))


def standardize_lipid_annotations(table):
    """Collapse indistinguishable species while retaining the best single reference spectrum."""
    from scMM.util.lipids import _regular_species
    restored, restored_ids = [], []
    for row_id, series in table.iterrows():
        row = series.to_dict()
        old_level = row.get("report_level")
        evidence = row.get("candidate_metrics", "")
        expand = old_level == "class_formula" or (old_level == "species" and not _regular_species(row["name"]))
        if expand and isinstance(evidence, str) and evidence:
            for i, member in enumerate(json.loads(evidence)):
                member.update(name=member.get("original_name") or member["name"], report_level="unchanged",
                              candidate_metrics="", accepted=bool(row["accepted"]))
                restored.append(member)
                restored_ids.append(f"{row_id}_r{i}")
        else:
            restored.append(row)
            restored_ids.append(row_id)
    table = pd.DataFrame(restored, index=pd.Index(restored_ids, name="annotation_id")) if restored else table
    groups = {}
    for row_id, series in table.iterrows():
        row = series.to_dict()
        name = str(row["name"])
        abbreviation = row.get("abbreviation", "")
        row["abbreviation"] = abbreviation = abbreviation if pd.notna(abbreviation) else ""
        if name and not row["is_isotope"]:
            if row.get("report_level") == "manual":
                abbreviation = ""
            label, level = report_lipid_name(name, abbreviation, row["category"], row["formula"])
            if level == "species":
                isotopes = _fixed_isotope_suffix(str(row["formula"]))
                if isotopes:
                    label = label.split("[M", 1)[0] + isotopes
        else:
            label, level = name, row.get("report_level", "unchanged")
        row.update(name=label, report_level=level)
        key = (row["feature_id"], label, _formula_identity(str(row["formula"])), row["adduct"], row["is_isotope"])
        groups.setdefault(key, []).append((row_id, row, name))
    output, indices = [], []
    for entries in groups.values():
        row_id, best, original = max(entries, key=lambda item: _spectral_rank(item[1]))
        best = dict(best)
        if not best["category"]:
            best["category"] = next((r["category"] for _, r, _ in entries if r["category"]), "")
        members = []
        for member_id, member, original_name in entries:
            evidence = member.get("candidate_metrics", "")
            if isinstance(evidence, str) and evidence:
                members.extend(json.loads(evidence))
            else:
                members.append({**member, "name": original_name, "candidate_id": str(member_id)})
        original_names = list(dict.fromkeys(m.get("original_name", m["name"]) for m in members))
        best.update(original_name=best.get("original_name", original),
                    original_names=json.dumps(original_names, ensure_ascii=False), isomer_count=len(original_names),
                    accepted=any(row["accepted"] for _, row, _ in entries),
                    candidate_metrics=pd.DataFrame(members).to_json(orient="records", force_ascii=False, double_precision=15))
        output.append(best)
        indices.append(row_id)
    if not output:
        return table.copy()
    result = pd.DataFrame(output, index=pd.Index(indices, name="annotation_id"))
    return result.sort_values(["is_isotope", "observed_mz", "feature_id", "combined_score", "abs_ppm_error"],
                              ascending=[True, True, True, False, True], kind="stable")


def write_annotations(data, table, key="annotation", *, mapping=None):
    from .annotation_tools import validate_var_mapping
    mapping = validate_var_mapping({"name": key} if mapping is None else mapping)
    table = standardize_lipid_annotations(table)
    isotope_flags = data.var.get("annotation_isotope", pd.Series(False, index=data.var_names))
    flags = table["feature_id"].map(isotope_flags.astype(bool)).fillna(False)
    selected = table[table["accepted"] & ~flags & table["name"].str.strip().ne("")]
    updated = data.var.copy()
    for field, target in mapping.items():
        grouped = selected.groupby("feature_id", sort=False)[field].agg(lambda values: list(dict.fromkeys(values.dropna())))
        numeric = pd.api.types.is_numeric_dtype(table[field])
        if numeric and all(len(values) <= 1 for values in grouped):
            values = pd.to_numeric(grouped.map(lambda items: items[0] if items else np.nan).reindex(data.var_names))
        else:
            values = grouped.map(lambda items: "/".join(str(item) for item in items)).reindex(data.var_names).fillna("")
        updated[target] = values
    data.var = updated
    return int(selected["feature_id"].nunique())


def export_annotations(table):
    """Merge same-formula isomers; preserve each candidate's measurements as JSON."""
    table = standardize_lipid_annotations(table[~table["is_isotope"] & table["match_level"].ne("filtered")])
    table = table.copy()
    for index, row in table.iterrows():
        members = json.loads(row["candidate_metrics"])
        for member in members:
            member["accepted"] = bool(row["accepted"])
        table.at[index, "candidate_metrics"] = json.dumps(members, ensure_ascii=False)
    return table.reset_index(drop=True)


def annotation_report(data, table):
    eligible = table[~table["is_isotope"] & table["match_level"].ne("filtered")]
    levels = eligible.groupby("feature_id")["match_level"].agg(lambda x: "MS/MS" if "MS/MS" in x.values else "MS1" if "MS1" in x.values else "unmatched")
    accepted = eligible[eligible["accepted"]].drop_duplicates(["feature_id", "category"])
    counts = accepted.groupby("category")["feature_id"].nunique()
    return levels.value_counts().to_dict(), counts.to_dict()
