import numpy as np
import pytest

from scMM.util._annotation_search import RESULT_COLUMNS
from scMM.util.annotation import (
    PROTON,
    SDFMzSearcher,
    load_lipidmaps_sdf,
    neutral_mass_from_mz,
    parse_sdf_record,
    theoretical_mz,
)

SDF_RECORD = """Example lipid
  scMM

>  <LM_ID>
LMTEST0001

>  <EXACT_MASS>
100.0

>  <COMMON_NAME>
Example

$$$$
"""

MULTI_RECORD_SDF = (
    SDF_RECORD
    + """Near lipid
  scMM

>  <LM_ID>
LMTEST0002

>  <EXACT_MASS>
100.0002

>  <COMMON_NAME>
Near example

$$$$
Invalid lipid
  scMM

>  <LM_ID>
LMINVALID

>  <EXACT_MASS>
not-a-number

$$$$
"""
)


def test_parse_sdf_record_accepts_standard_spacing() -> None:
    parsed = parse_sdf_record(SDF_RECORD)

    assert parsed["record_title"] == "Example lipid"
    assert parsed["LM_ID"] == "LMTEST0001"
    assert parsed["EXACT_MASS"] == "100.0"


def test_load_and_search_sdf(tmp_path) -> None:
    sdf_path = tmp_path / "lipids.sdf"
    sdf_path.write_text(SDF_RECORD, encoding="utf-8")

    database = load_lipidmaps_sdf(sdf_path)
    searcher = SDFMzSearcher(sdf_path)
    result = searcher.search(100.0 + PROTON, ppm_tol=1.0, mode="pos")

    assert database.loc[0, "EXACT_MASS"] == 100.0
    assert result.loc[0, "LM_ID"] == "LMTEST0001"
    assert result.loc[0, "adduct"] == "[M+H]+"
    assert np.isclose(result.loc[0, "ppm_error"], 0.0)


def test_annotation_validates_tolerance_and_charge(tmp_path) -> None:
    sdf_path = tmp_path / "lipids.sdf"
    sdf_path.write_text(SDF_RECORD, encoding="utf-8")
    searcher = SDFMzSearcher(sdf_path)

    with pytest.raises(ValueError, match="non-negative"):
        searcher.search_one(101.0, ppm_tol=-1.0)
    with pytest.raises(ValueError, match="charge"):
        theoretical_mz(100.0, {"mass_shift": 1.0, "charge": 0})


def test_search_ranks_by_error_and_limits_results(tmp_path) -> None:
    sdf_path = tmp_path / "lipids.sdf"
    sdf_path.write_text(MULTI_RECORD_SDF, encoding="utf-8")
    searcher = SDFMzSearcher(sdf_path)

    result = searcher.search_one(
        100.0 + PROTON,
        ppm_tol=5.0,
        mode="pos",
        max_results=1,
    )

    assert result["LM_ID"].tolist() == ["LMTEST0001"]
    assert len(searcher.db) == 2


def test_multi_mass_search_returns_stable_empty_schema(tmp_path) -> None:
    sdf_path = tmp_path / "lipids.sdf"
    sdf_path.write_text(SDF_RECORD, encoding="utf-8")
    searcher = SDFMzSearcher(sdf_path)

    result = searcher.search([500.0, 600.0], ppm_tol=1.0, mode="both")

    assert result.empty
    assert result.columns.tolist() == RESULT_COLUMNS


def test_custom_doubly_charged_adduct_round_trip(tmp_path) -> None:
    sdf_path = tmp_path / "lipids.sdf"
    sdf_path.write_text(SDF_RECORD, encoding="utf-8")
    adduct = {"mass_shift": 2 * PROTON, "charge": 2}
    searcher = SDFMzSearcher(sdf_path, adducts_pos={"[M+2H]2+": adduct})
    query_mz = theoretical_mz(100.0, adduct)

    result = searcher.search(query_mz, ppm_tol=1.0, mode="pos")

    assert result.loc[0, "adduct"] == "[M+2H]2+"
    assert neutral_mass_from_mz(query_mz, adduct) == pytest.approx(100.0)


def test_annotation_validates_mode_and_result_limit(tmp_path) -> None:
    sdf_path = tmp_path / "lipids.sdf"
    sdf_path.write_text(SDF_RECORD, encoding="utf-8")
    searcher = SDFMzSearcher(sdf_path)

    with pytest.raises(ValueError, match="mode"):
        searcher.search_one(101.0, mode="unknown")
    with pytest.raises(ValueError, match="max_results"):
        searcher.search_one(101.0, max_results=-1)


def test_load_sdf_requires_exact_mass_field(tmp_path) -> None:
    sdf_path = tmp_path / "missing-mass.sdf"
    sdf_path.write_text("No mass\n$$$$\n", encoding="utf-8")

    with pytest.raises(ValueError, match="EXACT_MASS"):
        load_lipidmaps_sdf(sdf_path)
