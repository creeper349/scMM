"""Candidate generation and ranking for SDF exact-mass records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._adducts import theoretical_mz

RESULT_COLUMNS = [
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
]

DATABASE_RESULT_FIELDS = RESULT_COLUMNS[6:]


def resolve_adducts(mode: str, positive: dict, negative: dict) -> dict:
    """Select positive, negative, or combined adduct definitions."""
    if mode not in {"pos", "neg", "both"}:
        raise ValueError("mode must be 'pos', 'neg', or 'both'.")
    if mode == "pos":
        return positive
    if mode == "neg":
        return negative
    return {**positive, **negative}


def search_database(
    database: pd.DataFrame,
    query_mz: float,
    ppm_tol: float,
    adducts: dict,
    max_results: int | None,
) -> pd.DataFrame:
    """Generate and rank all database candidates for one query mass."""
    validate_search_options(ppm_tol, max_results)
    exact_masses = database["EXACT_MASS"].to_numpy(dtype=float)
    records = []
    for adduct_name, adduct_info in adducts.items():
        theoretical = theoretical_mz(exact_masses, adduct_info)
        ppm_error = (query_mz - theoretical) / theoretical * 1e6
        hit_indices = np.where(np.abs(ppm_error) <= ppm_tol)[0]
        records.extend(
            _candidate_record(
                database.iloc[index],
                query_mz,
                adduct_name,
                theoretical[index],
                ppm_error[index],
            )
            for index in hit_indices
        )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result = result.sort_values(
        ["query_mz", "abs_ppm_error", "adduct"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return result.head(max_results) if max_results is not None else result


def validate_search_options(ppm_tol: float, max_results: int | None) -> None:
    if ppm_tol < 0:
        raise ValueError("ppm_tol must be non-negative")
    if max_results is not None and max_results < 0:
        raise ValueError("max_results must be non-negative or None")


def _candidate_record(row, query_mz, adduct, theoretical, ppm_error) -> dict:
    record = {
        "query_mz": query_mz,
        "adduct": adduct,
        "theoretical_mz": theoretical,
        "ppm_error": ppm_error,
        "abs_ppm_error": abs(ppm_error),
        "neutral_mass": row.get("EXACT_MASS", np.nan),
    }
    record.update(
        {field: row.get(field, "") for field in DATABASE_RESULT_FIELDS if field != "neutral_mass"}
    )
    return record


def empty_search_results() -> pd.DataFrame:
    """Return the stable schema used by a multi-mass search with no hits."""
    return pd.DataFrame(columns=RESULT_COLUMNS)
