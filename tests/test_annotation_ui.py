from streamlit.testing.v1 import AppTest


def _edit_field_mapping(app, key, edits):
    import json
    editor = next(element for element in app.dataframe if element.proto.id.endswith(f'-{key}'))
    states = app._tree.get_widget_states()
    state = next((state for state in states.widgets if state.id == editor.proto.id), None)
    if state is None:
        state = states.widgets.add()
        state.id = editor.proto.id
    state.string_value = json.dumps({'edited_rows': edits, 'added_rows': [], 'deleted_rows': []})
    return app._run(widget_state=states)


def test_annotation_controls_results_acceptance_and_export(tmp_path):
    script = '''
import numpy as np
import pandas as pd
import streamlit as st
from scMM.webapp import _state
from scMM.file.data import CyESIData
from scMM.analysis.annotation import annotate_features
from scMM.annotation_ui import render_annotation_page
_state(st)
if st.session_state.data is None:
    data = CyESIData(X=np.ones((4,2)), var=pd.DataFrame({'mz':[181.070665,500.]}, index=['f0','f1']))
    data.var['annotation_isotope'] = np.zeros(2,dtype=np.uint8)
    db = pd.DataFrame([dict(library_id='ref', name='Glucose', formula='C6H12O6', neutral_mass=180.06338810418, precursor_mz=np.nan, adduct='[M+H]+', category='Sugar', polarity='pos', database_id='id')])
    peaks=np.array([[60.,100.],[90.,25.],[120.,9.]])
    annotate_features(data,db,{'ref':peaks},adducts=['[M+H]+'],measured={'f0':dict(peaks=peaks,scans=['scan=1'])})
    st.session_state.data=data
render_annotation_page()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 5
    uploaders = {w.proto.label: w.proto for w in app.get('file_uploader')}
    assert uploaders['上传MS1 数据库'].multiple_files
    assert uploaders['上传标准 MS/MS 谱库'].multiple_files
    assert app.number_input(key='ann_isotope_diff').proto.format == '%.8f'
    assert app.number_input(key='ann_carbon13').proto.format == '%.8f'
    assert app.number_input(key='ann_isotope_r2').proto.format == '%.6f'
    app.number_input(key='ann_carbon13').set_value(.01090012).run()
    assert app.number_input(key='ann_carbon13').value == .01090012
    app.button(key='ann_deisotope').click().run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Accept First').click().run()
    assert not app.exception
    assert app.session_state.data.uns['annotation']['table']['accepted'].sum() == 1
    next(b for b in app.button if b.label == '写入 var').click().run()
    assert app.session_state.data.var.loc['f0','annotation'] == 'Glucose'
    next(b for b in app.button if b.label == '导出注释表格').click().run()
    assert not app.exception
    assert len(app.get('download_button')) == 1
    app.session_state.ann_mirror = 'a0'
    app.run()
    assert not app.exception
    # AppTest reruns the full script when a dialog widget changes.
    app.session_state.ann_mirror = 'a0'
    app.button(key='ann_mirror_preview').click().run()
    assert not app.exception
    assert b'<text' in app.session_state.ann_mirror_cache['payload']
    legacy = app.session_state.data.uns['annotation']
    legacy['table'] = legacy['table'].drop(columns=['reference_msms_available','msms_status'])
    app.run()
    assert not app.exception
    assert legacy['table']['reference_msms_available'].iloc[0]
    app.session_state[f'ann_table_{id(app.session_state.data)}'] = dict(action='select', id='a1', event_id='test-selection')
    app.run()
    assert not app.exception
    assert app.session_state.ann_spectrum_feature == 'f1'


def test_multiple_ms1_and_msms_libraries_run_together(tmp_path):
    import json
    import numpy as np
    import pandas as pd
    import pyopenms as oms

    database_paths = []
    library_paths = []
    for name in ('A', 'B'):
        db = tmp_path / f'{name}.csv'
        pd.DataFrame([dict(name=f'Compound {name}', neutral_mass=98.992723533379)]).to_csv(db, index=False)
        database_paths.append(str(db))
        lib = tmp_path / f'{name}.msp'
        lib.write_text(f'Name: Compound {name}\nPrecursorMZ: 100\nPrecursor_type: [M+H]+\nNum Peaks: 3\n60 100\n70 50\n80 20\n', encoding='utf8')
        library_paths.append(str(lib))
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(2)
    spectrum.setType(oms.SpectrumSettings.SpectrumType.CENTROID)
    spectrum.set_peaks((np.array([60.,70.,80.]), np.array([100.,50.,20.])))
    precursor = oms.Precursor()
    precursor.setMZ(100.)
    spectrum.setPrecursors([precursor])
    experiment = oms.MSExperiment()
    experiment.addSpectrum(spectrum)
    measured = tmp_path / 'measured.mzML'
    oms.MzMLFile().store(str(measured), experiment)
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.webapp import _state
from scMM.file.data import CyESIData
from scMM.annotation_ui import render_annotation_page
_state(st)
if st.session_state.data is None:
    st.session_state.data=CyESIData(X=np.ones((3,1)),var=pd.DataFrame({'mz':[100.]},index=['feature']))
render_annotation_page()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    app.text_area(key='ann_path_database').set_value('\n'.join(database_paths))
    app.text_area(key='ann_path_library').set_value('\n'.join(library_paths))
    app.text_area(key='ann_path_measured').set_value(str(measured))
    app.button(key='ann_run').click().run()
    assert not app.exception
    assert not app.error
    annotation = app.session_state.data.uns['annotation']
    table = annotation['table']
    assert set(table.name) == {'Compound A', 'Compound B'}
    assert set(table.match_level) == {'MS/MS'}
    assert len(annotation['spectra']['reference']) == 4
    for _, row in table.iterrows():
        assert row.source_library.endswith('.msp')
        sources = {r['source_library'] for r in json.loads(row.candidate_metrics)}
        assert sources == {f'{row["name"][-1]}.csv', f'{row["name"][-1]}.msp'}


def test_manual_import_preview_and_var_write_preserves_unmatched(tmp_path):
    path = tmp_path / 'manual.csv'
    path.write_text('feature_id,name\nf2,PC 34:1\nf0,Glucose\n', encoding='utf-8-sig')
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.webapp import _state
from scMM.file.data import CyESIData
from scMM.annotation_ui import render_annotation_page
_state(st)
if st.session_state.data is None:
    data=CyESIData(X=np.ones((2,3)),var=pd.DataFrame({'mz':[100.,200.,300.], 'manual_annotation':['old0','keep','old2']},index=['f0','f1','f2']))
    st.session_state.data=data
render_annotation_page()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    app.text_input(key='ann_path_manual').set_value(str(path)).run()
    app.button(key='ann_manual_read').click().run()
    assert not app.exception
    assert app.session_state.data.var.manual_annotation.tolist() == ['old0', 'keep', 'old2']
    app.button(key='ann_manual_write').click().run()
    assert not app.exception
    assert app.session_state.data.var.manual_annotation.tolist() == ['Glucose', 'keep', 'PC 34:1']
    app.checkbox(key='ann_manual_clear').set_value(True)
    app.button(key='ann_manual_write').click().run()
    assert app.session_state.data.var.manual_annotation.tolist() == ['Glucose', '', 'PC 34:1']


def test_spectrum_source_selection_generation_feature_focus_and_invalidation(tmp_path):
    import numpy as np
    import pyopenms as oms
    for name in ['a', 'b']:
        experiment = oms.MSExperiment()
        for frame, level in enumerate([1, 2, 1]):
            spectrum = oms.MSSpectrum()
            spectrum.setMSLevel(level)
            spectrum.set_peaks((np.array([99.99,100.,100.01,101.,101.00335,101.01]), np.array([0.,10.,0.,0.,1.,0.])*(frame+1)))
            experiment.addSpectrum(spectrum)
        oms.MzMLFile().store(str(tmp_path / f'{name}.mzML'), experiment)
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.webapp import _state
from scMM.file.data import CyESIData
from scMM.annotation_ui import render_annotation_page
_state(st)
if st.session_state.data is None:
    data=CyESIData(X=np.ones((4,2)),obs=pd.DataFrame({'source_file':['a.mzML','a.mzML','b.mzML','b.mzML'],'frame_id':[0,2,0,2]},index=['a0','a2','b0','b2']),var=pd.DataFrame({'mz':[100.,101.00335], 'isotope_parent':['','parent']},index=['parent','isotope']))
    st.session_state.data=data
render_annotation_page()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    app.session_state.data.uns['files'] = {f'{name}.mzML': dict(path=str(tmp_path/f'{name}.mzML')) for name in ['a','b']}
    assert app.multiselect(key='ann_spectrum_sources').value == []
    assert app.button(key='ann_cell_sum').disabled
    app.multiselect(key='ann_spectrum_sources').set_value(['b.mzML']).run()
    app.button(key='ann_cell_sum').click().run()
    assert not app.exception
    assert not app.error
    cache = app.session_state.ann_cell_spectrum
    assert cache['cells'] == ['b0','b2']
    assert [record['source_file'] for record in cache['sources']] == ['b.mzML']
    assert cache['sources'][0]['frame_ids'] == [0,2]
    assert len(app.get('plotly_chart')) == 1
    app.selectbox(key='ann_spectrum_feature').set_value('isotope').run()
    assert not app.exception
    assert app.session_state.ann_cell_spectrum is cache
    assert 'isotope' in app.get('plotly_chart')[0].proto.spec
    app.session_state.data.set_cell_filter('qc', np.array([True,True,False,True]))
    app.run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0
    app.multiselect(key='ann_spectrum_sources').set_value(['a.mzML']).run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0
    app.button(key='ann_cell_sum').click().run()
    assert app.session_state.ann_cell_spectrum['cells'] == ['a0','a2']
    from scMM.analysis.annotation import annotate_features
    import pandas as pd
    database = pd.DataFrame([dict(library_id='ref', name='Parent', formula='', neutral_mass=98.992723533379,
                                  precursor_mz=np.nan, adduct='[M+H]+', category='', polarity='pos', database_id='test')])
    annotate_features(app.session_state.data, database, {}, adducts=['[M+H]+'])
    app.run()
    app.selectbox(key='ann_spectrum_feature').set_value('parent').run()
    cache = app.session_state.ann_cell_spectrum
    table = app.session_state.data.uns['annotation']['table']
    row = table.index[table.feature_id.eq('isotope')][0]
    app.session_state[f'ann_table_{id(app.session_state.data)}'] = dict(action='select', id=row, event_id='spectrum-selection')
    app.run()
    assert not app.exception
    assert app.selectbox(key='ann_spectrum_feature').value == 'isotope'
    assert app.session_state.ann_cell_spectrum is cache
    import json
    elements = list(app.tabs[3])
    table_position = next(i for i, element in enumerate(elements)
                          if element.type == 'component_instance' and 'scmm_annotation_table' in element.proto.component_name)
    spectrum_position = next(i for i, element in enumerate(elements)
                             if element.type == 'plotly_chart' and 'Cell-frame MS1 spectrum' in element.proto.spec)
    pie_position = next(i for i, element in enumerate(elements)
                        if element.type == 'plotly_chart' and json.loads(element.proto.spec)['data'][0]['type'] == 'pie')
    parameter_keys = {'ann_spectrum_sources', 'ann_cell_count', 'ann_cell_seed', 'ann_cell_active',
                      'ann_cell_mz_min', 'ann_cell_mz_max', 'ann_cell_resolution', 'ann_cell_points',
                      'ann_spectrum_feature', 'ann_cell_zoom'}
    parameter_positions = [i for i, element in enumerate(elements) if getattr(element, 'key', None) in parameter_keys]
    assert len(parameter_positions) == len(parameter_keys)
    assert table_position < spectrum_position < min(parameter_positions)
    assert max(parameter_positions) < pie_position
    app.multiselect(key='ann_spectrum_sources').set_value(['a.mzML','b.mzML']).run()
    app.number_input(key='ann_cell_count').set_value(1)
    app.button(key='ann_cell_sum').click().run()
    assert not app.exception
    assert [item['n_cell_frames'] for item in app.session_state.ann_cell_spectrum['sources']] == [1,1]


def test_manual_import_mapping_editor_multiple_columns_and_selected_rows(tmp_path):
    source = tmp_path/'manual.csv'
    source.write_text('feature_id,name,category\nf2,PC 34:1,Lipid\nf0,Glucose,Sugar\n',encoding='utf8')
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.webapp import _state
from scMM.file.data import CyESIData
from scMM.annotation_ui import render_annotation_page
_state(st)
if st.session_state.data is None:
    st.session_state.data=CyESIData(X=np.ones((2,3)),var=pd.DataFrame({'mz':[100.,200.,300.]},index=['f0','f1','f2']))
render_annotation_page()
'''
    app=AppTest.from_string(script,default_timeout=30).run()
    app.text_input(key='ann_path_manual').set_value(str(source)).run()
    app.button(key='ann_manual_read').click().run()
    app.text_input(key='ann_manual_rows').set_value('2').run()
    editor=next(element for element in app.dataframe if element.proto.id.endswith('-ann_manual_mapping'))
    fields=editor.value.source.tolist()
    app.button(key='ann_manual_write').click()
    _edit_field_mapping(app,'ann_manual_mapping',{fields.index('name'):{'write':True,'target':'compound_name'},
                                                  fields.index('category'):{'write':True,'target':'compound_class'}})
    assert not app.exception
    assert app.session_state.data.var.compound_name.tolist() == ['Glucose','','']
    assert app.session_state.data.var.compound_class.tolist() == ['Sugar','','']
    _edit_field_mapping(app,'ann_manual_mapping',{fields.index('name'):{'write':True,'target':'duplicate'},
                                                  fields.index('category'):{'write':True,'target':'duplicate'}})
    assert not app.exception
    assert any('不同的键名' in message.value for message in app.info)
