import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyopenms as oms
import pytest

from scMM.analysis.annotation import (
    ADDUCTS, accept_annotations, annotate_features, annotation_report, collect_msms,
    deisotope_features, export_annotations, load_annotation_database, match_fragments,
    normalize_lipid_name, theoretical_isotopes, write_annotations, canonical_adduct,
)


def test_library_fixed_isotopes_preserve_mass_and_pattern(tmp_path):
    formula = 'C15[13C]3H34O5'
    native = oms.EmpiricalFormula('C15(13)C3H34O5')
    source = tmp_path / 'labelled.mgf'
    source.write_text(f'BEGIN IONS\nNAME=Labelled standard\nFORMULA={formula}\nADDUCT=[M-H]-\nPEPMASS=332.2515\n100 100\nEND IONS\n', encoding='utf8')
    db, _ = load_annotation_database(source)
    assert db.iloc[0].formula == formula
    assert db.iloc[0].neutral_mass == pytest.approx(native.getMonoWeight())
    assert db.iloc[0].neutral_mass - oms.EmpiricalFormula('C18H34O5').getMonoWeight() == pytest.approx(3 * 1.003355, abs=1e-6)
    np.testing.assert_allclose(theoretical_isotopes(formula, '[M-H]-', 4), theoretical_isotopes('C15(13)C3H34O5', '[M-H]-', 4))
    np.testing.assert_allclose(theoretical_isotopes('C20H21D11O3', '[M-H]-', 4), theoretical_isotopes('C20H21[2H]11O3', '[M-H]-', 4))


@pytest.mark.parametrize('alias,canonical', [('[M+H-H2O]+','[M-H2O+H]+'), ('[M+H -H2O]+','[M-H2O+H]+'), ('[M.Cl]-','[M+Cl]-'), ('[M.]+','[M]+')])
def test_lipidmaps_library_adduct_aliases(alias, canonical):
    assert canonical_adduct(alias) == canonical
from scMM.annotation_ui import apply_table_edit
from scMM.file.data import CyESIData
from scMM.plot.annotation import mirror_spectrum
from scMM.plot.export import render_figure


def sample_data():
    mz = 180.06338810418 + ADDUCTS['[M+H]+']['mass_shift']
    expected = theoretical_isotopes('C6H12O6', '[M+H]+', 4)
    parent = np.arange(1, 21, dtype=float) * 100
    return CyESIData(
        X=np.column_stack([parent, parent * expected[1] / expected[0], np.ones(20)]),
        var=pd.DataFrame({'mz': [mz, mz + 1.003355, 500.]}, index=['f0', 'f1', 'f2']),
    )


def database(tmp_path):
    path = tmp_path / 'library.json'
    path.write_text(json.dumps([
        dict(name='Glucose', formula='C6H12O6', adduct='[M+H]+', category='Sugar', peaks=[[60., 100.], [90., 25.], [120., 9.]]),
        dict(name='Fructose', formula='C6H12O6', adduct='[M+H]+', category='Sugar', peaks=[[60., 10.], [150., 100.], [170., 30.]]),
        dict(name='Different formula', formula='C7H16O5', neutral_mass=180.06338810418, adduct='[M+H]+', category='Other'),
    ]), encoding='utf8')
    return load_annotation_database(path)


def annotated(tmp_path):
    data = sample_data()
    deisotope_features(data, ppm_tol=5.)
    db, spectra = database(tmp_path)
    table = annotate_features(data, db, spectra, adducts=['[M+H]+'], min_matches=3,
                              measured={'f0': dict(peaks=np.array([[60., 100.], [90., 25.], [120., 9.]]), scans=['a.raw:scan=42'])})
    return data, table


def test_write_multiple_annotation_fields_numeric_values_and_atomic_failure(tmp_path):
    data, table = annotated(tmp_path)
    accept_annotations(table, first=True)
    count = write_annotations(data, table, mapping={'name':'compound','formula':'formula','ppm_error':'ppm','accepted':'accepted'})
    assert count == 1
    assert data.var.loc['f0','compound'] == table.loc[table.accepted,'name'].iloc[0]
    assert data.var.loc['f0','formula'] == 'C6H12O6'
    assert pd.api.types.is_numeric_dtype(data.var.ppm)
    assert np.isnan(data.var.loc['f1','ppm'])
    assert data.var.loc['f1','compound'] == ''
    assert data.var.loc['f0','accepted'] == 1
    before = data.var.copy()
    with pytest.raises(ValueError):
        write_annotations(data, table, mapping={'name':'same','formula':'same'})
    with pytest.raises(KeyError):
        write_annotations(data, table, mapping={'name':'pending','nonexistent':'bad'})
    pd.testing.assert_frame_equal(data.var, before)
    path = tmp_path/'fields.h5ad'
    data.write_h5ad(path)
    import anndata as ad
    restored = ad.read_h5ad(path)
    assert restored.var.loc['f0','compound'] == data.var.loc['f0','compound']


def test_adduct_masses_and_lipid_names():
    known = {'[M]+': -.000548579909, '[M+H]+': 1.007276466621, '[M+Na]+': 22.989220702091,
             '[M+K]+': 38.963157906491, '[M+NH4]+': 18.033825553441,
             '[M-H2O+H]+': -17.003288217409, '[M+Li]+': 7.015454856691,
             '[M-H]-': -1.007276466621, '[M+HCOO]-': 44.998202836979,
             '[M+CH3COO]-': 59.013852901439, '[M+Cl]-': 34.969401261909}
    for adduct, mass in known.items():
        assert ADDUCTS[adduct]['mass_shift'] == pytest.approx(mass, abs=1e-9)
    assert normalize_lipid_name('PC 16:0/18:1') == 'PC 34:1'
    assert normalize_lipid_name('PC(16:0/18:1)') == 'PC 34:1'
    assert normalize_lipid_name('Cer t18:0/24:1') == 'Cer 42:1;O3'
    assert normalize_lipid_name('Cer t 18:0') == 'Cer 18:0;O3'
    assert normalize_lipid_name('Glucose') == 'Glucose'
    assert normalize_lipid_name('PC 34:1') == 'PC 34:1'


def test_deisotope_preserves_matrix_and_maps_feature_ids():
    data = sample_data()
    original = data.X.copy()
    pairs = deisotope_features(data, ppm_tol=5.)
    assert len(pairs) == 1
    assert data.var['annotation_isotope'].tolist() == [0, 1, 0]
    assert data.var.loc['f1', 'isotope_parent'] == 'f0'
    np.testing.assert_array_equal(data.X, original)
    assert data.var['annotation_isotope'].dtype == np.uint8


def test_candidate_ranking_acceptance_export_and_h5ad(tmp_path):
    data, table = annotated(tmp_path)
    top = table.iloc[0]
    assert top['name'] == 'Glucose'
    assert top['msms_similarity'] == pytest.approx(1.)
    assert top['isotope_similarity'] > .99
    assert top['match_level'] == 'MS/MS'
    assert top['msms_scan_number'] == 'a.raw:scan=42'
    assert table.iloc[-1]['is_isotope']
    accept_annotations(table, first=True)
    assert table['accepted'].sum() == 1
    write_annotations(data, table, 'compound')
    assert data.var['compound'].tolist() == ['Glucose', '', '']
    accept_annotations(table)
    write_annotations(data, table, 'compound')
    assert set(data.var.loc['f0', 'compound'].split('/')) == {'Glucose', 'Fructose', 'Different formula'}
    exported = export_annotations(table)
    assert len(exported[exported.feature_id.eq('f0')]) == 3
    assert {'Glucose', 'Fructose'}.issubset(set(exported.name))
    assert exported['is_isotope'].sum() == 0
    levels, categories = annotation_report(data, table)
    assert levels == {'MS/MS': 1, 'unmatched': 1}
    assert categories['Sugar'] == 1
    restored = CyESIData.read(data.save(tmp_path / 'annotated.h5ad'))
    assert restored.uns['annotation']['table']['accepted'].sum() == 3
    np.testing.assert_allclose(restored.uns['annotation']['spectra']['observed']['f0'], data.uns['annotation']['spectra']['observed']['f0'])


def test_isotope_restoration_and_hidden_filter_refresh(tmp_path):
    data, table = annotated(tmp_path)
    data.set_feature_filter('annotation_isotope', ~data.var['annotation_isotope'].astype(bool).to_numpy())
    row_id = table[table['feature_id'].eq('f1')].index[0]
    apply_table_edit(data, table, row_id, 'is_isotope', False)
    assert data.var.loc['f1', 'annotation_isotope'] == 0
    assert data.feature_keep_mask().all()
    db, references = database(tmp_path)
    refreshed = annotate_features(data, db, references, adducts=['[M+H]+'])
    assert not refreshed['is_isotope'].any()
    assert refreshed.loc[refreshed.feature_id.eq('f1'), 'match_level'].iloc[0] == 'unmatched'


def test_ms1_and_spectral_duplicate_combines_evidence(tmp_path):
    data = sample_data()
    deisotope_features(data)
    db, refs = database(tmp_path)
    row = db.iloc[[0]].copy()
    row['library_id'] = 'ms1'
    row['category'] = 'From MS1'
    db.loc[0, 'category'] = ''
    table = annotate_features(data, pd.concat([row, db]), refs, adducts=['[M+H]+'], min_matches=1,
                              measured={'f0': dict(peaks=np.array([[60., 100.]]), scans=['scan=1'])})
    glucose = table[table.name.eq('Glucose')]
    assert len(glucose) == 1
    assert glucose.iloc[0]['msms_similarity'] > 0
    assert glucose.iloc[0]['category'] == 'From MS1'


def test_one_to_one_fragments_count_unmatched_in_denominator():
    score, matched = match_fragments([[100., 100.], [100.0001, 100.], [200., 100.]], [[100., 100.]])
    assert len(matched) == 1
    assert score == pytest.approx(1 / np.sqrt(3))


def _spectrum(mz, intensity, precursor, native, level=2, profile=False):
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(level)
    spectrum.setNativeID(native)
    spectrum.setType(oms.SpectrumSettings.SpectrumType.PROFILE if profile else oms.SpectrumSettings.SpectrumType.CENTROID)
    spectrum.set_peaks((np.array(mz, dtype=float), np.array(intensity, dtype=float)))
    p = oms.Precursor()
    p.setMZ(precursor)
    spectrum.setPrecursors([p])
    return spectrum


@pytest.mark.parametrize('suffix', ['mzML', 'mzXML'])
def test_multiple_ms_files_sum_only_targeted_ms2(tmp_path, suffix):
    files = []
    for i in range(2):
        experiment = oms.MSExperiment()
        experiment.addSpectrum(_spectrum([60., 90.], [10., 20.], 181., 'scan=2'))
        for level, precursor in [(1, 181.), (3, 181.), (2, 182.)]:
            experiment.addSpectrum(_spectrum([60., 90.], [1000., 1000.], precursor, f'scan={level+3}', level=level))
        path = tmp_path / f'input{i}.{suffix}'
        writer = oms.MzMLFile() if suffix == 'mzML' else oms.MzXMLFile()
        writer.store(str(path), experiment)
        files.append(path)
    measured = collect_msms(files, {'feature': 181.}, min_relative_intensity=0.)
    np.testing.assert_allclose(measured['feature']['peaks'], [[60., 20.], [90., 40.]])
    assert len(measured['feature']['scans']) == 2


@pytest.mark.parametrize('unknown_type', [False, True])
def test_profile_baseline_centroiding_retains_known_peaks(tmp_path, monkeypatch, unknown_type):
    x = np.arange(50., 80., .002)
    y = 5 + 100 * np.exp(-.5 * ((x-60.)/.015)**2) + 70 * np.exp(-.5 * ((x-70.)/.015)**2)
    experiment = oms.MSExperiment()
    experiment.addSpectrum(_spectrum(x, y, 181., 'scan=20', profile=True))
    path = tmp_path / 'profile.mzML'
    oms.MzMLFile().store(str(path), experiment)
    if unknown_type:
        spectrum = experiment.getSpectrum(0)
        spectrum.setType(oms.SpectrumSettings.SpectrumType.UNKNOWN)
        experiment.setSpectra([spectrum])
        monkeypatch.setattr('scMM.file.io.load_single_file', lambda *a, **k: (experiment, {}))
    measured = collect_msms([path], {'f': 181.}, profile_step=.002, mz_min=50., mz_max=80., baseline_width=.5)
    peaks = measured['f']['peaks']
    np.testing.assert_allclose(peaks[:, 0], [60., 70.], atol=.003)
    np.testing.assert_allclose(peaks[:, 1], [100., 70.], rtol=.02)


def test_msp_mgf_csv_and_mass_formula_loading(tmp_path):
    msp = tmp_path / 'ref.msp'
    msp.write_text('Name: A\nFormula: C6H12O6\nPrecursorMZ: 181.07066\nPrecursor_type: [M+H]+\nIon_mode: Positive\nNum Peaks: 2\n60 100\n90 20\n\nName: B\nFormula: C6H12O6\nPrecursor_type: [M+H]+\nNum Peaks: 1\n70 10\n', encoding='utf8')
    db, spectra = load_annotation_database(msp)
    assert db.name.tolist() == ['A', 'B']
    assert len(spectra['lib0']) == 2
    msp.write_bytes(b'NAME: PC 34:1\r\r\nFORMULA: C42H82NO8P\r\r\nPRECURSORTYPE: [M+H]+\r\r\nNum Peaks: 2\r\r\n60 100\r\r\n90 20\r\r\n')
    db, spectra = load_annotation_database(msp)
    assert len(db) == 1
    np.testing.assert_allclose(spectra['lib0'], [[60., 100.], [90., 20.]])
    mgf = tmp_path / 'ref.mgf'
    mgf.write_text('BEGIN IONS\nTITLE=Glucose\nFORMULA=C6H12O6\nPEPMASS=181.07066 1000\nADDUCT=[M+H]+\n60 100\nEND IONS\n', encoding='utf8')
    db, _ = load_annotation_database(mgf)
    assert db.iloc[0].precursor_mz == pytest.approx(181.07066)
    csv = tmp_path / 'ref.csv'
    pd.DataFrame([dict(name='Glucose', formula='C6H12O6')]).to_csv(csv, index=False)
    db, _ = load_annotation_database(csv)
    assert db.iloc[0].neutral_mass == pytest.approx(180.063388, abs=1e-5)
    sdf = tmp_path / 'ref.sdf'
    sdf.write_text('Glucose\n>  <EXACT_MASS>  (1)\n180.063388\n\n>  <FORMULA>\nC6H12O6\n\n$$$$\n', encoding='utf8')
    db, _ = load_annotation_database(sdf)
    assert db.iloc[0].neutral_mass == pytest.approx(180.063388)


def test_mirror_and_report_matplotlib_svg(tmp_path):
    figure = mirror_spectrum([[60., 100.], [90., 50.]], [[60., 10.], [100., 100.]], 'Mirror')
    assert np.nanmax(figure.data[0].y) == 100
    assert np.nanmin(figure.data[1].y) == -100
    payload = render_figure(figure, title='Mirror')
    assert b'<text' in payload and b'Observed MS/MS' in payload and b'Library MS/MS' in payload
    (tmp_path / 'mirror.svg').write_bytes(payload)
    (tmp_path / 'mirror.png').write_bytes(render_figure(figure, file_format='png', title='Mirror'))
    import plotly.graph_objects as go
    payload = render_figure(go.Figure(go.Pie(labels=['MS1', 'MS/MS', 'Unmatched'], values=[3, 2, 1], hole=.35)))
    assert b'MS/MS' in payload and b'<text' in payload


def test_thermo_ms2_precursor_and_spectrum_type(monkeypatch, tmp_path):
    from scMM.file.thermo_raw import load_thermo_raw
    raw = SimpleNamespace(
        IsOpen=True, IsError=False, InAcquisition=False,
        RunHeaderEx=SimpleNamespace(FirstSpectrum=1, LastSpectrum=2),
        FileHeader=SimpleNamespace(CreationDate=None),
        SelectInstrument=lambda *_: None, GetInstrumentData=lambda: SimpleNamespace(Model='Test', Name='Test'),
        GetFilterForScanNumber=lambda i: SimpleNamespace(MSOrder=i),
        GetScanStatsForScanNumber=lambda _: SimpleNamespace(IsCentroidScan=True, SpectrumPacketType='FtCentroid'),
        GetCentroidStream=lambda *_: SimpleNamespace(Masses=[60.], Intensities=[100.]),
        GetScanEventForScanNumber=lambda _: SimpleNamespace(GetReaction=lambda _: SimpleNamespace(PrecursorMass=181.07066, CollisionEnergy=25.)),
        RetentionTimeFromScanNumber=lambda _: 1., Dispose=lambda: None,
    )
    monkeypatch.setattr('scMM.file.thermo_raw._load_api', lambda: (SimpleNamespace(FileFactory=lambda _: raw), SimpleNamespace(MS=1)))
    experiment, _ = load_thermo_raw(str(tmp_path / 'file.raw'))
    spectrum = experiment.getSpectrum(1)
    assert spectrum.getPrecursors()[0].getMZ() == pytest.approx(181.07066)
    assert spectrum.getPrecursors()[0].getActivationEnergy() == 25.
    assert spectrum.getType() == oms.SpectrumSettings.SpectrumType.CENTROID
