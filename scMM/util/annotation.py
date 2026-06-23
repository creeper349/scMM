import re
from typing import Iterable

import numpy as np
import pandas as pd


PROTON = 1.007276466812
NA = 22.989218
K = 38.963158
NH4 = 18.033823
CL = 34.969402
FA = 44.998201


DEFAULT_ADDUCTS_POS = {
    "[M+H]+": {"mass_shift": PROTON, "charge": 1},
    "[M+Na]+": {"mass_shift": NA, "charge": 1},
    "[M+K]+": {"mass_shift": K, "charge": 1},
    "[M+NH4]+": {"mass_shift": NH4, "charge": 1},
}

DEFAULT_ADDUCTS_NEG = {
    "[M-H]-": {"mass_shift": -PROTON, "charge": 1},
    "[M+Cl]-": {"mass_shift": CL, "charge": 1},
    "[M+FA-H]-": {"mass_shift": FA - PROTON, "charge": 1},
}


def parse_sdf_record(record: str) -> dict:
    lines = record.splitlines()
    result = {}

    if lines:
        result["record_title"] = lines[0].strip()

    pattern = re.compile(r"^> <(.+?)>$")

    i = 0
    while i < len(lines):
        m = pattern.match(lines[i].strip())
        if m:
            key = m.group(1).strip()
            i += 1

            values = []
            while i < len(lines):
                line = lines[i]
                if line.strip() == "":
                    break
                if pattern.match(line.strip()):
                    i -= 1
                    break
                values.append(line.rstrip())
                i += 1

            result[key] = "\n".join(values).strip()

        i += 1

    return result


def load_lipidmaps_sdf(sdf_path: str) -> pd.DataFrame:
    with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    records = [r.strip() for r in text.split("$$$$") if r.strip()]
    parsed = [parse_sdf_record(r) for r in records]

    df = pd.DataFrame(parsed)

    if "EXACT_MASS" not in df.columns:
        raise ValueError("SDF does not contain EXACT_MASS field.")

    df["EXACT_MASS"] = pd.to_numeric(df["EXACT_MASS"], errors="coerce")
    df = df.dropna(subset=["EXACT_MASS"]).reset_index(drop=True)

    return df


def theoretical_mz(neutral_mass: float, adduct_info: dict) -> float:
    return (neutral_mass + adduct_info["mass_shift"]) / abs(adduct_info["charge"])


def neutral_mass_from_mz(mz: float, adduct_info: dict) -> float:
    return mz * abs(adduct_info["charge"]) - adduct_info["mass_shift"]


def _is_iterable_mz(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index))


class SDFMzSearcher:
    def __init__(
        self,
        sdf_path: str,
        adducts_pos: dict | None = None,
        adducts_neg: dict | None = None,
    ):
        self.db = load_lipidmaps_sdf(sdf_path)
        self.adducts_pos = adducts_pos or DEFAULT_ADDUCTS_POS
        self.adducts_neg = adducts_neg or DEFAULT_ADDUCTS_NEG

    def _get_adducts(self, mode: str) -> dict:
        if mode not in {"pos", "neg", "both"}:
            raise ValueError("mode must be 'pos', 'neg', or 'both'.")

        if mode == "pos":
            return self.adducts_pos
        if mode == "neg":
            return self.adducts_neg

        return {**self.adducts_pos, **self.adducts_neg}

    def search_one(
        self,
        mz: float,
        ppm_tol: float = 5.0,
        mode: str = "both",
        max_results: int | None = None,
    ) -> pd.DataFrame:
        adducts = self._get_adducts(mode)
        masses = self.db["EXACT_MASS"].to_numpy(dtype=float)

        results = []

        for adduct_name, adduct_info in adducts.items():
            theo_mz = theoretical_mz(masses, adduct_info)
            err_ppm = (mz - theo_mz) / theo_mz * 1e6

            hit_idx = np.where(np.abs(err_ppm) <= ppm_tol)[0]

            for idx in hit_idx:
                row = self.db.iloc[idx]

                results.append({
                    "query_mz": mz,
                    "adduct": adduct_name,
                    "theoretical_mz": theo_mz[idx],
                    "ppm_error": err_ppm[idx],
                    "abs_ppm_error": abs(err_ppm[idx]),
                    "neutral_mass": row.get("EXACT_MASS", np.nan),
                    "LM_ID": row.get("LM_ID", ""),
                    "COMMON_NAME": row.get("COMMON_NAME", ""),
                    "SYSTEMATIC_NAME": row.get("SYSTEMATIC_NAME", ""),
                    "ABBREVIATION": row.get("ABBREVIATION", ""),
                    "FORMULA": row.get("FORMULA", ""),
                    "CATEGORY": row.get("CATEGORY", ""),
                    "MAIN_CLASS": row.get("MAIN_CLASS", ""),
                    "SUB_CLASS": row.get("SUB_CLASS", ""),
                    "INCHI_KEY": row.get("INCHI_KEY", ""),
                    "HMDB_ID": row.get("HMDB_ID", ""),
                    "KEGG_ID": row.get("KEGG_ID", ""),
                    "PUBCHEM_CID": row.get("PUBCHEM_CID", ""),
                })

        out = pd.DataFrame(results)

        if out.empty:
            return out

        out = out.sort_values(
            ["query_mz", "abs_ppm_error", "adduct"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        if max_results is not None:
            out = out.head(max_results)

        return out

    def search(
        self,
        mz,
        ppm_tol: float = 5.0,
        mode: str = "both",
        max_results_per_mz: int | None = None,
    ) -> pd.DataFrame:
        """
        Search single or multiple m/z values.

        Parameters
        ----------
        mz : float or list-like
            Single m/z value or multiple m/z values.
        ppm_tol : float
            ppm tolerance.
        mode : {"pos", "neg", "both"}
            Ionization mode.
        max_results_per_mz : int or None
            Maximum returned candidates for each query m/z.

        Returns
        -------
        pd.DataFrame
            All annotation candidates within ppm tolerance.
        """

        if _is_iterable_mz(mz):
            mz_values = [float(x) for x in mz]
        else:
            mz_values = [float(mz)]

        all_hits = []

        for mz_value in mz_values:
            hits = self.search_one(
                mz=mz_value,
                ppm_tol=ppm_tol,
                mode=mode,
                max_results=max_results_per_mz,
            )

            if not hits.empty:
                all_hits.append(hits)

        if not all_hits:
            return pd.DataFrame(columns=[
                "query_mz",
                "adduct",
                "theoretical_mz",
                "ppm_error",
                "abs_ppm_error",
                "neutral_mass",
                "LM_ID",
                "COMMON_NAME",
                "SYSTEMATIC_NAME",
                "ABBREVIATION",
                "FORMULA",
                "CATEGORY",
                "MAIN_CLASS",
                "SUB_CLASS",
                "INCHI_KEY",
                "HMDB_ID",
                "KEGG_ID",
                "PUBCHEM_CID",
            ])

        return pd.concat(all_hits, ignore_index=True)