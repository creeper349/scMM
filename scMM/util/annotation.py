"""SDF parsing and the stable mass-annotation search facade."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._adducts import (
    CL,
    DEFAULT_ADDUCTS_NEG,
    DEFAULT_ADDUCTS_POS,
    FA,
    NA,
    NH4,
    PROTON,
    K,
    neutral_mass_from_mz,
    theoretical_mz,
)
from ._annotation_search import (
    empty_search_results,
    resolve_adducts,
    search_database,
    validate_search_options,
)


def parse_sdf_record(record: str) -> dict:
    """Parse one SDF record title and property blocks into a dictionary."""
    lines = record.splitlines()
    result = {"record_title": lines[0].strip()} if lines else {}
    property_header = re.compile(r"^>\s*<([^>]+)>\s*$")
    index = 0
    while index < len(lines):
        match = property_header.match(lines[index].strip())
        if match:
            key, values, index = _read_sdf_property(lines, index, property_header)
            result[key] = "\n".join(values).strip()
        index += 1
    return result


def _read_sdf_property(lines: list[str], header_index: int, pattern):
    key = pattern.match(lines[header_index].strip()).group(1).strip()
    values = []
    index = header_index + 1
    while index < len(lines):
        line = lines[index]
        if line.strip() == "":
            break
        if pattern.match(line.strip()):
            return key, values, index - 1
        values.append(line.rstrip())
        index += 1
    return key, values, index


def load_lipidmaps_sdf(sdf_path: str | Path) -> pd.DataFrame:
    """Load SDF properties and retain records with numeric ``EXACT_MASS``."""
    with Path(sdf_path).expanduser().open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    records = [record.strip() for record in text.split("$$$$") if record.strip()]
    database = pd.DataFrame([parse_sdf_record(record) for record in records])
    if "EXACT_MASS" not in database.columns:
        raise ValueError("SDF does not contain EXACT_MASS field.")
    database["EXACT_MASS"] = pd.to_numeric(database["EXACT_MASS"], errors="coerce")
    return database.dropna(subset=["EXACT_MASS"]).reset_index(drop=True)


def _is_iterable_mz(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, pd.Series, pd.Index))


class SDFMzSearcher:
    """Search an SDF property table under configurable ion adducts."""

    def __init__(
        self,
        sdf_path: str | Path,
        adducts_pos: dict | None = None,
        adducts_neg: dict | None = None,
    ):
        self.db = load_lipidmaps_sdf(sdf_path)
        self.adducts_pos = DEFAULT_ADDUCTS_POS if adducts_pos is None else adducts_pos
        self.adducts_neg = DEFAULT_ADDUCTS_NEG if adducts_neg is None else adducts_neg

    def _get_adducts(self, mode: str) -> dict:
        return resolve_adducts(mode, self.adducts_pos, self.adducts_neg)

    def search_one(
        self,
        mz: float,
        ppm_tol: float = 5.0,
        mode: str = "both",
        max_results: int | None = None,
    ) -> pd.DataFrame:
        """Return candidates for one observed m/z value."""
        validate_search_options(ppm_tol, max_results)
        return search_database(
            self.db,
            float(mz),
            ppm_tol,
            self._get_adducts(mode),
            max_results,
        )

    def search(
        self,
        mz,
        ppm_tol: float = 5.0,
        mode: str = "both",
        max_results_per_mz: int | None = None,
    ) -> pd.DataFrame:
        """Return candidates for one or multiple observed m/z values."""
        mz_values = [float(value) for value in mz] if _is_iterable_mz(mz) else [float(mz)]
        hits = []
        for value in mz_values:
            result = self.search_one(
                value,
                ppm_tol=ppm_tol,
                mode=mode,
                max_results=max_results_per_mz,
            )
            if not result.empty:
                hits.append(result)
        return pd.concat(hits, ignore_index=True) if hits else empty_search_results()


__all__ = [
    "CL",
    "DEFAULT_ADDUCTS_NEG",
    "DEFAULT_ADDUCTS_POS",
    "FA",
    "NA",
    "NH4",
    "PROTON",
    "K",
    "SDFMzSearcher",
    "load_lipidmaps_sdf",
    "neutral_mass_from_mz",
    "parse_sdf_record",
    "theoretical_mz",
]
