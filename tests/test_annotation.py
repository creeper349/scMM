import numpy as np
import pytest

from scMM.util.annotation import (
    PROTON,
    SDFMzSearcher,
    load_lipidmaps_sdf,
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
