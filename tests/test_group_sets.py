import io

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from streamlit.testing.v1 import AppTest

from scMM.analysis.group_sets import feature_membership, compound_records, annotation_membership, intersection_counts
from scMM.file.data import CyESIData
from scMM.plot.group_sets import group_set_figure


def example_data(sparse_raw=False):
    raw = np.array([[5, 0, 1, 100], [0, 8, 0, 100], [5, 0, 0, 100], [0, 8, 0, 100]], dtype=float)
    return CyESIData(X=np.zeros_like(raw), layers={"raw": sparse.csr_matrix(raw) if sparse_raw else raw},
                     obs=pd.DataFrame({"group": ["A", "A", "B", "B"], "cell_keep_qc": np.array([1, 0, 1, 1], np.uint8)}, index=list("abcd")),
                     var=pd.DataFrame({"mz": [100., 200., 300., 400.], "feature_keep_qc": np.array([1, 1, 1, 0], np.uint8),
                                       "annotation": ["C/Z", "C", "D", "E"], "category": ["Lipid", "Lipid", "Sugar", "Natural"]}, index=[f"f{i}" for i in range(4)]))


@pytest.mark.parametrize("sparse_raw", [False, True])
def test_feature_sets_use_raw_active_masks_and_detection_thresholds(sparse_raw):
    data = example_data(sparse_raw)
    features = feature_membership(data, "group", ["A", "B"])
    assert features.to_dict("list") == {"A": [True, False, True], "B": [True, True, False]}
    assert features.index.tolist() == ["f0", "f1", "f2"]
    strict = feature_membership(data, "group", ["A", "B"], detection_rate=.6)
    assert strict.index.tolist() == ["f0", "f2"]
    assert strict.B.sum() == 0
    assert feature_membership(data, "group", ["A", "B"], layer="X").empty
    assert feature_membership(data, "group", ["A", "B"], intensity_threshold=8).empty
    assert intersection_counts(features)["count"].sum() == 3


def test_annotation_sets_deduplicate_compounds_and_count_categories():
    data = example_data()
    features = feature_membership(data, "group", ["A", "B"])
    table = pd.DataFrame({"feature_id": ["f0", "f1", "f2", "f3"], "name": ["C", "C", "D", "E"],
                          "category": ["Lipid", "Lipid", "Sugar", "Natural"], "is_isotope": False,
                          "accepted": [True, True, False, True], "match_level": "MS1"})
    records = compound_records(data, table=table)
    members, counts = annotation_membership(features, records)
    assert members.index.tolist() == ["C"]
    assert counts.Lipid.tolist() == [1, 1]
    records = compound_records(data, table=table, accepted_only=False)
    members, counts = annotation_membership(features, records)
    assert members.loc["D"].tolist() == [True, False]
    assert counts.Sugar.tolist() == [1, 0]
    data.var["annotation_isotope"] = np.array([0, 0, 1, 0], np.uint8)
    records = compound_records(data, table=table, accepted_only=False)
    assert "D" not in records.name.tolist()
    records = compound_records(data, name_key="annotation", category_key="category")
    members, counts = annotation_membership(features, records)
    assert set(members.index) == {"C", "Z"}
    assert members.loc["Z"].tolist() == [True, True]
    assert counts.Lipid.tolist() == [2, 2]


def test_annotation_sets_drop_missing_placeholders_and_current_or_legacy_isotopes():
    names=['PC 34:1',None,'  ','NaN','-', 'Parent B','Iso C','Iso D']
    data=CyESIData(X=np.ones((2,8)),obs=pd.DataFrame({'group':['A','B']},index=['a','b']),
                   var=pd.DataFrame({'mz':np.arange(8)+100.,'names':names,
                                     'annotation_isotope':['0']*6+['1','true'],
                                     'deisotope_role':['unique']*5+['isotope']*3},index=[f'f{i}' for i in range(8)]))
    features=feature_membership(data,'group',['A','B'])
    records=compound_records(data,name_key='names')
    assert records.name.tolist()==['PC 34:1','Parent B']
    members,_=annotation_membership(features,records)
    assert members.sum().tolist()==[2,2]
    table=pd.DataFrame({'feature_id':data.var_names,'name':names,'category':'Lipid',
                        'is_isotope':False,'match_level':'MS1','accepted':True})
    assert compound_records(data,table=table).name.tolist()==['PC 34:1','Parent B']
    data.var=data.var.drop(columns='annotation_isotope')
    assert compound_records(data,name_key='names').name.tolist()==['PC 34:1']
    data.var=data.var.drop(columns='deisotope_role')
    data.var['isotope_order']=[0]*5+[1,2,3]
    assert compound_records(data,name_key='names').name.tolist()==['PC 34:1']


@pytest.mark.parametrize("kind", ["Venn", "UpSet", "composition"])
def test_set_figures_preserve_svg_text_and_category_totals(kind, tmp_path):
    membership = pd.DataFrame({"A": [1, 1, 0], "B": [0, 1, 1], "C": [0, 0, 0]}, dtype=bool)
    counts = pd.DataFrame({"Lipid": [2, 1, 0], "Sugar": [1, 3, 0]}, index=["A", "B", "C"])
    figure = group_set_figure(membership, kind=kind, composition=counts, percentages=True)
    buffer = io.BytesIO()
    with mpl.rc_context({"svg.fonttype": "none"}):
        figure.savefig(buffer, format="svg", bbox_inches="tight")
    assert b"<text" in buffer.getvalue()
    figure.savefig(tmp_path / f"{kind}.png", dpi=100, bbox_inches="tight")
    if kind == "composition":
        heights = np.array([patch.get_height() for patch in figure.axes[0].patches]).reshape(2, 3)
        np.testing.assert_allclose(heights.sum(axis=0), [100., 100., 0.])


def test_upset_matrix_with_pandas_copy_on_write():
    membership = pd.DataFrame({'A':[True,True,False], 'B':[False,True,True]})
    with pd.option_context('mode.copy_on_write', True):
        figure = group_set_figure(membership, kind='UpSet')
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png')
    assert buffer.getvalue().startswith(b'\x89PNG')


@pytest.mark.parametrize('composition', [False, True])
def test_group_sets_choose_custom_var_annotation_keys(composition):
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.file.data import CyESIData
from scMM.group_sets_ui import render_group_sets
if 'data' not in st.session_state:
    st.session_state.data=CyESIData(X=np.array([[1.,1.,0.],[1.,0.,1.]]),obs=pd.DataFrame({'group':['A','B']},index=['a','b']),var=pd.DataFrame({'mz':[100.,200.,300.],'identification':pd.Categorical(['Shared','Only A',None]),'numeric_id':[101.,np.nan,103.]},index=['f0','f1','f2']))
render_group_sets(st,st.session_state.data,key='custom_sets',composition=COMPOSITION)
'''.replace('COMPOSITION',repr(composition))
    app=AppTest.from_string(script,default_timeout=30).run()
    app.selectbox(key='custom_sets_content').set_value('注释物质').run()
    assert app.selectbox(key='custom_sets_name').value == 'identification'
    app.button(key='custom_sets_run').click().run()
    assert not app.exception
    members = app.session_state.custom_sets_result['membership']
    assert set(members.index) == {'Shared','Only A'}
    assert members.loc['Only A'].tolist() == [True,False]
    app.selectbox(key='custom_sets_name').set_value('numeric_id').run()
    assert not app.exception
    assert not app.get('image')
    app.button(key='custom_sets_run').click().run()
    assert not app.exception
    assert set(app.session_state.custom_sets_result['membership'].index) == {'101.0','103.0'}


def test_group_sets_ui_both_figures_annotation_composition_and_export(monkeypatch, tmp_path):
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.file.data import CyESIData
from scMM.group_sets_ui import render_group_sets
if 'data' not in st.session_state:
    st.session_state.data=CyESIData(X=np.array([[1.,0.,2.],[1.,3.,0.],[1.,0.,0.],[0.,3.,2.]]),obs=pd.DataFrame({'group':['A','A','B','B']},index=list('abcd')),var=pd.DataFrame({'mz':[100.,200.,300.],'annotation':['C','D','E'],'category':['Lipid','Lipid','Sugar']},index=['f0','f1','f2']))
render_group_sets(st,st.session_state.data,key='test_sets',composition=True)
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    app.button(key="test_sets_run").click().run()
    assert not app.exception
    assert len(app.get("image")) == 2
    app.selectbox(key="test_sets_kind").set_value("Venn").run()
    assert not app.exception
    assert len(app.get("image")) == 2
    app.selectbox(key="test_sets_content").set_value("注释物质").run()
    app.button(key="test_sets_run").click().run()
    assert not app.exception
    assert set(app.session_state.test_sets_result["membership"].index) == {"C", "D", "E"}
    app.button(key="test_sets_Venn_export").click().run()
    assert not app.exception
    # AppTest runs the full script for dialog controls.
    app.button(key="test_sets_Venn_export").click()
    app.button(key="test_sets_Venn_preview").click().run()
    assert not app.exception
    assert b"<text" in app.session_state["test_sets_Venn_preview_data"]["payload"]
    path = tmp_path / "venn.svg"
    monkeypatch.setattr("scMM.file.dialogs.choose_save_path", lambda *args: str(path))
    app.button(key="test_sets_Venn_export").click()
    app.button(key="test_sets_Venn_download").click().run()
    assert path.read_bytes().startswith(b"<?xml")
    app.multiselect(key="test_sets_groups").set_value(["A"]).run()
    app.selectbox(key="test_sets_kind").set_value("UpSet").run()
    app.button(key="test_sets_run").click().run()
    assert not app.exception
    assert any("至少两个" in item.value for item in app.info)
