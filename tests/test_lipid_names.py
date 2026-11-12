import json

import numpy as np
import pandas as pd
import pytest

from scMM.analysis.annotation import (ADDUCTS, accept_annotations, annotate_features,
                                      export_annotations, load_annotation_database,
                                      standardize_lipid_annotations, write_annotations)
from scMM.file.data import CyESIData
from scMM.util.lipids import normalize_lipid_name, report_lipid_name


@pytest.mark.parametrize('original,expected', [
    ('PC 16:0/18:1(9Z)', 'PC 34:1'),
    ('PC(18:1(9E)/16:0)', 'PC 34:1'),
    ('PC 18:0_16:1', 'PC 34:1'),
    ('LPC 0:0/18:1(9Z)', 'LPC 18:1'),
    ('TG(16:0/18:1/18:2)', 'TG 52:3'),
    ('CL(16:0/18:1/18:2/20:4)', 'CL 72:7'),
    ('PE P-16:0/20:4(5Z,8Z,11Z,14Z)', 'PE O-36:5'),
    ('PE O-18:1/18:0', 'PE O-36:1'),
    ('PC 18:0/18:2;OOH', 'PC 36:2;O2'),
    ('FA 18:1;9oxo', 'FA 18:1;9oxo'),
    ('FA 18:2(9Z,12Z);13OH', 'FA 18:2(9Z,12Z);13OH'),
    ('FA 18:0;9Y', 'FA 18:0;9Y'),
    ('Cer(t18:0/24:1(15Z))', 'Cer 42:1;O3'),
    ('SM(d18:1/24:0)', 'SM 42:1;O2'),
    ('GalCer(d18:1/24:0)', 'HexCer 42:1;O2'),
    ('GlcCer(d18:1/24:0)', 'HexCer 42:1;O2'),
    ('LacCer(d18:1/16:0)', 'Hex2Cer 34:1;O2'),
    ('GM3(d18:1/16:0)', 'GM3(d18:1/16:0)'),
    ('CE(18:2(9Z,12Z))', 'SE 45:3'),
    ('SE 27:2/18:1', 'SE 45:3'),
    ('Cholesterol', 'ST 27:1;O'),
    ('PIP2(3\',5\')(16:0/18:1)', 'PIP2 34:1'),
    ('15-HETE', '15-HETE'),
    ('Maresin 1 (LMFA04050001)', 'Maresin 1 (LMFA04050001)'),
    ('DGTS 16:0/18:1', 'DGTS 34:1'),
    ('DGTS(16:0/18:1(9Z))', 'DGTS 34:1'),
    ('PC 16:0/18:1[M+H]+', 'PC 34:1'),
    ('PC 16:0/18:1-d7', 'PC 34:1[M[2]H7]'),
])
def test_species_names_and_idempotence(original, expected):
    assert normalize_lipid_name(original) == expected
    assert normalize_lipid_name(expected) == expected


def _run(tmp_path):
    path = tmp_path / 'lipids.json'
    # Same composition, distinct sn/chains/DB positions, one lower cosine despite more fragments.
    mass = 760.585 - ADDUCTS['[M+H]+']['mass_shift']
    records = [
        dict(name='PC 16:0/18:1(9Z)', formula='C42H82NO8P', neutral_mass=mass, category='PC', adduct='[M+H]+',
             peaks=[[100.,100.],[150.,10.],[200.,1.]], id='poor'),
        dict(name='PC 18:1(9E)/16:0', formula='C42H82NO8P', neutral_mass=mass, category='PC', adduct='[M+H]+',
             peaks=[[100.,100.],[150.,20.]], id='best'),
        dict(name='PC 18:0/16:1(7Z)', formula='C42H82NO8P', neutral_mass=mass, category='PC', adduct='[M+H]+',
             peaks=[[100.,10.],[150.,10.]], id='other'),
        # Same molecular formula can also denote PE with another chain composition.
        dict(name='PE 18:0/19:1', formula='C42H82NO8P', neutral_mass=mass, category='PE', adduct='[M+H]+'),
    ]
    path.write_text(json.dumps(records), encoding='utf8')
    db, spectra = load_annotation_database(path)
    data = CyESIData(X=np.ones((4,1)), var=pd.DataFrame({'mz':[760.585]}, index=['feature']))
    table = annotate_features(data, db, spectra, adducts=['[M+H]+'], min_matches=1,
                              measured={'feature':dict(peaks=np.array([[100.,100.],[150.,20.],[200.,.1]]),scans=['scan=8'])})
    return data, table


def test_species_merge_keeps_best_real_spectrum_and_all_evidence(tmp_path):
    data, table = _run(tmp_path)
    assert set(table.name) == {'PC 34:1', 'PE 37:1'}
    pc = table[table.name.eq('PC 34:1')].iloc[0]
    assert pc.original_name == 'PC 18:1(9E)/16:0'
    assert pc.database_id == 'best'
    assert pc.isomer_count == 3
    assert pc.n_matched_fragments == 2
    members = json.loads(pc.candidate_metrics)
    assert len(members) == 3
    assert pc.msms_similarity == pytest.approx(max(r['msms_similarity'] for r in members), abs=1e-14)
    np.testing.assert_array_equal(data.uns['annotation']['spectra']['reference'][pc.library_id], [[100.,100.],[150.,20.]])
    twice = standardize_lipid_annotations(table)
    assert twice.name.tolist() == table.name.tolist()
    assert len(json.loads(twice[twice.name.eq('PC 34:1')].iloc[0].candidate_metrics)) == 3
    accept_annotations(table)
    write_annotations(data, table)
    assert set(data.var.loc['feature','annotation'].split('/')) == {'PC 34:1','PE 37:1'}
    exported = export_annotations(table)
    assert len(exported) == 2  # PC and PE remain separate despite their identical formula.
    assert all(member['accepted'] for value in exported.candidate_metrics for member in json.loads(value))
    restored = CyESIData.read(data.save(tmp_path / 'lipids.h5ad'))
    assert restored.uns['annotation']['table'].name.tolist() == table.name.tolist()


def test_same_similarity_prefers_more_matched_fragments(tmp_path):
    data, table = _run(tmp_path)
    pc = table[table.name.eq('PC 34:1')].iloc[0].to_dict()
    pc.pop('candidate_metrics')
    low, high = pc.copy(), pc.copy()
    low.update(name='PC 16:0/18:1', original_name='few', n_matched_fragments=2)
    high.update(name='PC 18:0/16:1', original_name='many', n_matched_fragments=5)
    merged = standardize_lipid_annotations(pd.DataFrame([low,high], index=['a','b']))
    assert merged.iloc[0].original_name == 'many'


def test_curated_shorthand_and_formula_fallback():
    assert report_lipid_name('full stereochemical library name','PC 34:1','Glycerophospholipids [GP]','C42H82NO8P') == ('PC 34:1','species')
    assert report_lipid_name('Unparsed structure','','Fatty Acyls [FA]','C20H32O4') == ('Unparsed structure','unchanged')
    assert report_lipid_name('Chlorophyll a','PR 55:12','Prenol Lipids [PR]','C55H72MgN4O5') == ('Chlorophyll a','unchanged')
    assert report_lipid_name('Glucose','','Sugar','C6H12O6') == ('Glucose','unchanged')


def test_complex_compounds_remain_separate_and_old_coarse_rows_restore(tmp_path):
    _, table = _run(tmp_path)
    template = table.iloc[0].to_dict()
    names = ['Chlorophyll a','Pheophytin a','Maresin 1','Maresin 2','Glucose','Fructose']
    rows = [{**template,'name':name,'original_name':name,'category':'Prenol Lipids [PR]',
             'abbreviation':'','report_level':'unchanged','candidate_metrics':''} for name in names]
    distinct = standardize_lipid_annotations(pd.DataFrame(rows,index=[f'r{i}' for i in range(len(rows))]))
    assert distinct.name.tolist() == names
    exported = export_annotations(distinct)
    assert exported.name.tolist() == names
    old = {**template, 'name':'PR [C42H82NO8P]', 'report_level':'class_formula',
           'candidate_metrics':json.dumps(rows[:2])}
    restored = standardize_lipid_annotations(pd.DataFrame([old], index=['old']))
    assert restored.name.tolist() == names[:2]


def test_lmsd_id_links_mgf_to_curated_abbreviation(tmp_path):
    data, _ = _run(tmp_path)
    mass = 760.585 - ADDUCTS['[M+H]+']['mass_shift']
    db = pd.DataFrame([
        dict(library_id='ms1',name='unparsed full name',abbreviation='PC 34:1',formula='C42H82NO8P',category='Glycerophospholipids [GP]',database_id='LMGP01010001',neutral_mass=mass,precursor_mz=np.nan,adduct='[M+H]+',polarity='pos'),
        dict(library_id='ms2',name='unparsed isomer name (LMGP01010001)',abbreviation='',formula='C42H82NO8P',category='',database_id='LMGP01010001',neutral_mass=mass,precursor_mz=np.nan,adduct='[M+H]+',polarity='pos'),
    ])
    peaks = np.array([[100.,100.],[150.,20.]])
    table = annotate_features(data,db,{'ms2':peaks},adducts=['[M+H]+'],min_matches=1,measured={'feature':dict(peaks=peaks,scans=['scan=1'])})
    assert len(table)==1
    assert table.iloc[0]['name']=='PC 34:1'
    assert table.iloc[0].library_id=='ms2'
    assert table.iloc[0].isomer_count==2


def test_exact_formula_mass_retains_reference_with_imprecise_library_precursor(tmp_path):
    import pyopenms as oms
    formula='C22H32O4'
    mz=oms.EmpiricalFormula(formula).getMonoWeight()+ADDUCTS['[M-H]-']['mass_shift']
    path=tmp_path/'imprecise.mgf'
    path.write_text('BEGIN IONS\nNAME=Maresin 1 (LMFA04050001)\nFORMULA=C22H32O4\nPEPMASS=359.2000\nADDUCT=[M-H]-\n100 100\n150 20\nEND IONS\n',encoding='utf8')
    db,spectra=load_annotation_database(path)
    data=CyESIData(X=np.ones((3,1)),var=pd.DataFrame({'mz':[mz]},index=['f']))
    table=annotate_features(data,db,spectra,adducts=['[M-H]-'],mode='neg',min_matches=1,
                            measured={'f':dict(peaks=spectra['lib0'],scans=['scan=1'])})
    assert table.iloc[0]['name']=='Maresin 1 (LMFA04050001)'
    assert table.iloc[0].msms_similarity==pytest.approx(1.)
    assert table.iloc[0].library_precursor_mz==359.2
    assert table.iloc[0].ppm_error==pytest.approx(0.)
