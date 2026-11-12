import numpy as np
import pandas as pd
import pyopenms as oms
import pytest

from scMM.analysis.annotation_tools import (
    align_manual_annotations, orient_annotation_table, read_annotation_table, sum_cell_frames,
    manual_annotation_columns, write_manual_annotation_columns, import_row_positions,
)
from scMM.file.data import CyESIData
from scMM.file.io import sum_spec
from scMM.plot.annotation import cell_spectrum_figure
from scMM.plot.export import render_figure


def test_manual_csv_excel_position_keys_and_row_orientation(tmp_path):
    data = CyESIData(X=np.ones((2, 3)), var=pd.DataFrame(
        {"mz": [100., 200., 300.], "database_id": [101, 102, 103]}, index=["001", "002", "003"]))
    path = tmp_path / "names.csv"
    path.write_text("名称\nGlucose\nPC 34:1\nCer 42:2;O2\n", encoding="utf-8-sig")
    frame = read_annotation_table(path)
    values, matched, _ = align_manual_annotations(data, frame, "名称")
    assert values.tolist() == ["Glucose", "PC 34:1", "Cer 42:2;O2"]
    assert matched.all()

    path.write_text("feature_id,name\n003,C\n001,A\n001,A2\n", encoding="utf8")
    frame = read_annotation_table(path)
    values, matched, description = align_manual_annotations(data, frame, "name")
    assert values.tolist() == ["A/A2", "", "C"]
    assert matched.tolist() == [True, False, True]
    assert description == "feature_id → feature_id"

    workbook = tmp_path / "annotations.xlsx"
    pd.DataFrame({"database_id": [103., 101.], "name": ["C", "A"]}).to_excel(workbook, index=False)
    frame = read_annotation_table(workbook)
    values, matched, _ = align_manual_annotations(data, frame, "name")
    assert values.tolist() == ["A", "", "C"]
    rows = pd.DataFrame({"field": ["name"], "003": ["C"], "001": ["A"], "002": ["B"]})
    frame = orient_annotation_table(rows, transpose=True)
    values, matched, _ = align_manual_annotations(data, frame, "name")
    assert values.tolist() == ["A", "B", "C"]
    assert matched.all()
    with pytest.raises(ValueError, match="3 条记录"):
        align_manual_annotations(data, frame.iloc[:2], "name", mode="position")


def test_manual_mz_matching_respects_ppm_and_auto_alignment():
    data = CyESIData(X=np.ones((2, 3)), var=pd.DataFrame({"mz": [100., 200., 300.]}, index=["a", "b", "c"]))
    frame = pd.DataFrame({"observed_mz": [200.0001, 100.0001, 300.01], "name": ["B", "A", "outside"]})
    for mode in ("auto", "key"):
        values, matched, _ = align_manual_annotations(data, frame, "name", mode=mode,
                                                       source_key="observed_mz", var_key="mz", ppm=5.)
        assert values.tolist() == ["A", "B", ""]
        assert matched.tolist() == [True, True, False]


def _experiment(scale):
    experiment = oms.MSExperiment()
    for index, level in enumerate([1, 2, 1, 1]):
        spectrum = oms.MSSpectrum()
        spectrum.setMSLevel(level)
        spectrum.set_peaks((np.array([99.9, 100., 100.1, 101., 101.00335, 101.1]),
                            np.array([0., 10., 0., 0., 1., 0.]) * scale * (index + 1)))
        experiment.addSpectrum(spectrum)
    return experiment


def test_cell_spectrum_reads_only_user_selected_files_and_original_ms1_frames(monkeypatch):
    data = CyESIData(X=np.ones((6, 2)), obs=pd.DataFrame(
        {"source_file": ["a.raw"] * 3 + ["b.mzML"] * 3, "frame_id": [0, 2, 3] * 2,
         "cell_keep_qc": np.array([1, 1, 1, 1, 0, 1], dtype=np.uint8)}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame({"mz": [100., 101.00335]}, index=["parent", "isotope"]))
    paths = []
    experiments = {"a.raw": _experiment(1.), "b.mzML": _experiment(7.)}

    def load(path, **kwargs):
        from pathlib import Path
        paths.append(Path(path).name)
        return experiments[paths[-1]], {}

    monkeypatch.setattr("scMM.analysis.annotation_tools.load_single_file", load)
    result = sum_cell_frames(data, {"b.mzML": "b.mzML"}, count=100, mz_range=(99., 102.))
    assert paths == ["b.mzML"]
    assert result["cells"] == ["c3", "c5"]
    assert result["sources"][0]["frame_ids"] == [0, 3]
    expected = oms.MSExperiment()
    expected.addSpectrum(experiments["b.mzML"][0])
    expected.addSpectrum(experiments["b.mzML"][3])
    mz, intensity = sum_spec(expected, mz_range=(99., 102.)).get_peaks()
    np.testing.assert_allclose(result["mz"], mz)
    np.testing.assert_allclose(result["intensity"], intensity)
    paths.clear()
    all_cells = sum_cell_frames(data, {"a.raw": "a.raw", "b.mzML": "b.mzML"}, count=100,
                                active_only=False, mz_range=(99., 102.))
    assert paths == ["a.raw", "b.mzML"]
    assert [row["n_cell_frames"] for row in all_cells["sources"]] == [3, 3]
    paths.clear()
    per_file = sum_cell_frames(data, {"a.raw": "a.raw", "b.mzML": "b.mzML"}, count=1,
                               sample_mode="per_file", mz_range=(99., 102.))
    assert paths == ["a.raw", "b.mzML"]
    assert [row["n_cell_frames"] for row in per_file["sources"]] == [1, 1]
    assert sum_cell_frames(data, {"a.raw": "a.raw"}, count=1, seed=3, mz_range=(99., 102.))["cells"] == \
        sum_cell_frames(data, {"a.raw": "a.raw"}, count=1, seed=3, mz_range=(99., 102.))["cells"]
    data.obs.loc["c3", "frame_id"] = 1
    with pytest.raises(ValueError, match="MS2"):
        sum_cell_frames(data, {"b.mzML": "b.mzML"}, count=100, mz_range=(99., 102.))
    with pytest.raises(ValueError, match="请选择"):
        sum_cell_frames(data, {})


def test_cell_spectrum_highlights_feature_and_isotopes_on_curve_and_exports():
    data = CyESIData(X=np.ones((2, 2)), var=pd.DataFrame({"mz": [100., 101.00335],
                    "isotope_parent": ["", "parent"], "annotation_isotope": [0, 1]}, index=["parent", "isotope"]))
    spectrum = sum_spec(_experiment(1.), mz_range=(99., 102.))
    mz, intensity = spectrum.get_peaks()
    payload = dict(mz=mz, intensity=intensity, resolution_200=35000.)
    for selected in data.var_names:
        figure = cell_spectrum_figure(data, payload, selected, half_window=.1, max_points=60)
        points = [trace for trace in figure.data if trace.mode == "markers+text"]
        assert {trace.name for trace in points} == {"parent", "isotope"}
        for point in points:
            assert point.y[0] == np.interp(point.x[0], figure.data[0].x, figure.data[0].y)
        assert len(figure.data[0].x) <= 65
        assert b"<text" in render_figure(figure, file_format="svg")


def test_manual_multi_field_row_selection_and_atomic_writes():
    data = CyESIData(X=np.ones((2, 3)), var=pd.DataFrame({'mz':[100.,200.,300.], 'compound':['old1','old2','old3']},index=['f0','f1','f2']))
    frame = pd.DataFrame({'feature_id':['f2','f0','f1'], 'name':['C','A','B'], 'category':['Lipid','Sugar','Other']})
    values, matched, method = manual_annotation_columns(data, frame, {'name':'compound','category':'class'},
                                                       row_positions=import_row_positions('1-2',len(frame)))
    assert matched.tolist() == [True, False, True]
    write_manual_annotation_columns(data, values, matched)
    assert data.var.compound.tolist() == ['A','old2','C']
    assert data.var['class'].tolist() == ['Sugar','','Lipid']
    write_manual_annotation_columns(data, values, matched, clear_unmatched=True)
    assert data.var.compound.tolist() == ['A','','C']
    before = data.var.copy()
    with pytest.raises(ValueError, match='不同的键名'):
        manual_annotation_columns(data, frame, {'name':'duplicate','category':'duplicate'})
    pd.testing.assert_frame_equal(data.var, before)
    for selection in ('0', '1-9', '3-1'):
        with pytest.raises(ValueError):
            import_row_positions(selection, 3)
    assert import_row_positions('1,3,1',3).tolist() == [0,2]
    values, matched, _ = manual_annotation_columns(data, frame[['name','category']], {'name':'compound','category':'class'},
                                                  mode='position', row_positions=[2])
    assert matched.tolist() == [False,False,True]
    assert values.loc['f2','compound'] == 'B'
    transposed = orient_annotation_table(pd.DataFrame({'field':['name','category'], 'f2':['C','Lipid'],
                                                      'f0':['A','Sugar'],'f1':['B','Other']}), transpose=True)
    values, matched, _ = manual_annotation_columns(data, transposed, {'name':'compound','category':'class'}, row_positions=[0,1])
    assert values.loc['f2','class'] == 'Lipid'
    assert matched.tolist() == [True, False, True]
