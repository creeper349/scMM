import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from scMM.plot.volcano import volcano_figure, draw_volcano, labelled_features, selected_features
from scMM.plot.export import render_figure


def test_volcano_fdr_coordinates_sizes_continuous_colour_and_export(tmp_path):
    result = pd.DataFrame({'feature_id':list('abcde'), 'mz':np.arange(5)+100., 'mean_a':1., 'mean_b':1.,
                           'p_value':[1e-10]*5, 'fdr':[1.,.1,.001,0.,np.nan], 'log2_fold_change':[-2.,-1.,1.,2.,0.]})
    labels = ['A','','C','','']
    figure = volcano_figure(result, labels, group_a='treated', group_b='control')
    points = figure.data[0]
    np.testing.assert_allclose(points.y[:3], [0.,1.,3.])
    assert len(points.x) == 4
    assert np.isfinite(points.y).all()
    assert np.all(np.diff(points.marker.size) > 0)
    assert points.marker.cmin == -points.marker.cmax == -2.
    assert figure.layout.yaxis.title.text == '−log10(FDR)'
    assert figure.layout.legend.title.text == '−log10(FDR)'
    assert not figure.layout.shapes
    assert {item.text for item in figure.layout.annotations} == {'A','C'}
    native = draw_volcano(figure)
    scatter = native.axes[0].collections[0]
    np.testing.assert_allclose(scatter.get_offsets()[:,1], points.y)
    np.testing.assert_allclose(scatter.get_sizes(), points.marker.size)
    assert isinstance(scatter.norm, TwoSlopeNorm)
    assert scatter.cmap.name == 'coolwarm'
    svg = render_figure(figure, file_format='svg', cmap='coolwarm')
    assert b'<text' in svg and b'FDR' in svg
    assert b'p-value' not in svg
    (tmp_path/'volcano.svg').write_bytes(svg)
    (tmp_path/'volcano.png').write_bytes(render_figure(figure,file_format='png',cmap='coolwarm',dpi=110))


def test_volcano_constant_or_empty_values():
    result = pd.DataFrame({'feature_id':['a'], 'mz':[100.], 'mean_a':[1.], 'mean_b':[1.],
                           'p_value':[.01], 'fdr':[.05], 'log2_fold_change':[0.]})
    figure = volcano_figure(result, [''], group_a='A',group_b='B')
    assert figure.data[0].marker.size[0] == 10.
    assert b'<svg' in render_figure(figure)
    result['fdr'] = np.nan
    figure = volcano_figure(result, [''], group_a='A',group_b='B')
    assert len(figure.data[0].x) == 0
    assert b'<svg' in render_figure(figure)


def test_volcano_label_layout_is_repeatable_for_export_preview():
    result = pd.DataFrame({'feature_id':list('abc'), 'mz':[100.,200.,300.], 'mean_a':1.,'mean_b':1.,
                           'p_value':.01,'fdr':.05,'log2_fold_change':0.})
    first = volcano_figure(result,['same']*3,group_a='A',group_b='B')
    second = volcano_figure(result,['same']*3,group_a='A',group_b='B')
    assert [item.to_plotly_json() for item in first.layout.annotations] == [item.to_plotly_json() for item in second.layout.annotations]


def test_volcano_leaders_join_each_point_to_its_adjusted_label(monkeypatch, tmp_path):
    import scMM.plot.volcano as module
    result = pd.DataFrame({'feature_id':['a','b','c','d'], 'mz':[100.,200.,300.,400.], 'mean_a':1.,'mean_b':1.,
                           'p_value':.001,'fdr':[.01,np.nan,.011,.012],'log2_fold_change':[1.,-3.,1.01,1.02]})
    captured = []
    original = module.draw_volcano

    def record(*args, **kwargs):
        native = original(*args, **kwargs)
        captured.append(native)
        return native

    monkeypatch.setattr(module, 'draw_volcano', record)
    figure = module.volcano_figure(result, ['Label A','Missing','Label C','Label D'],group_a='A',group_b='B')
    native = captured[0]
    points = np.asarray(native.axes[0].collections[0].get_offsets())
    assert len(figure.layout.annotations) == 3
    for annotation, point, text in zip(figure.layout.annotations, points, native.axes[0].texts):
        np.testing.assert_allclose([annotation.x, annotation.y], point)
        np.testing.assert_allclose([annotation.ax, annotation.ay], text.get_position())
        assert annotation.xref == annotation.axref == 'x'
        assert annotation.yref == annotation.ayref == 'y'
        assert annotation.text == text.get_text()
    # Matplotlib arrow endpoints use the same data coordinates after layout changes.
    from matplotlib.patches import FancyArrowPatch
    for width, height, font in [(10.,7.,10.),(6.,5.,16.)]:
        rendered = original(figure,width=width,height=height,annotation_size=font)
        rendered.canvas.draw()
        arrows = [patch for patch in rendered.axes[0].patches if isinstance(patch,FancyArrowPatch)]
        assert arrows
        for arrow in arrows:
            tail, head = arrow._posA_posB
            assert any(np.allclose(head, point) for point in points)
            assert any(np.allclose(tail, text.get_position()) for text in rendered.axes[0].texts)
        rendered.savefig(tmp_path/f'volcano_leaders_{int(width)}.png',dpi=110,bbox_inches='tight')
    assert b'<text' in render_figure(figure,file_format='svg')


def test_manual_labels_priority_caps_modes_and_font_in_export():
    assert labelled_features(['a','b','c'], {'a':False,'z':True}, 2) == ['z','b']
    assert labelled_features(['a','b'], {'z':True,'x':True}, 1, '仅手动') == ['x']
    assert labelled_features(['a','b'], {'a':False,'x':True}, 1, '仅自动') == ['a']
    assert labelled_features(['a'], {'z':True}, 0) == []
    assert selected_features({'selection':{'points':[{'curve_number':0,'point_index':1},
                                                     {'curve_number':1,'point_index':0},
                                                     {'point_index':-1}]}}, ['a','b']) == ['b']
    result = pd.DataFrame({'feature_id':['a'], 'mz':[100.], 'mean_a':1.,'mean_b':1.,'p_value':.01,'fdr':.05,'log2_fold_change':1.})
    figure = volcano_figure(result,['Manual'],group_a='A',group_b='B',annotation_size=17.)
    assert figure.layout.annotations[0].font.size == 17. * 4/3
    assert draw_volcano(figure).axes[0].texts[0].get_fontsize() == 17.
    assert draw_volcano(figure,annotation_size=21.).axes[0].texts[0].get_fontsize() == 21.
    assert b'Manual' in render_figure(figure,file_format='svg')


def test_volcano_click_toggle_controls_and_export_metadata():
    import json
    from streamlit.testing.v1 import AppTest
    script = '''
import streamlit as st
import numpy as np
import pandas as pd
from scMM.webapp import _state, _render_differential_plots
from scMM.file.data import CyESIData
_state(st)
if st.session_state.data is None:
    st.session_state.data=CyESIData(X=np.ones((4,3)),obs=pd.DataFrame({'group':['A','A','B','B']},index=list('abcd')),var=pd.DataFrame({'mz':[100.,200.,300.],'names':['Alpha','Beta','Gamma']},index=['a','b','c']))
    st.session_state.differential_result=pd.DataFrame({'feature_id':['a','b','c'],'mz':[100.,200.,300.],'mean_a':1.,'mean_b':1.,'p_value':[.001,.002,.5],'fdr':[.01,.02,.8],'log2_fold_change':[2.,-2.,.2]})
    st.session_state.differential_context={'group_key':'group','group_a':'A','group_b':'B'}
_render_differential_plots(st,st.session_state.data)
'''
    app=AppTest.from_string(script,default_timeout=30).run()
    app.selectbox(key='volcano_label_mode').set_value('仅手动')
    app.selectbox(key='volcano_var_label').set_value('names')
    app.number_input(key='volcano_label_count').set_value(1)
    app.number_input(key='volcano_label_size').set_value(17.).run()

    def click(points):
        plot = next(p for p in app.get('plotly_chart') if json.loads(p.proto.spec)['layout'].get('meta',{}).get('scmm_plot')=='volcano')
        states=app._tree.get_widget_states()
        state=next((state for state in states.widgets if state.id==plot.proto.id),None)
        if state is None:
            state=states.widgets.add()
            state.id=plot.proto.id
        state.string_value=json.dumps({'selection':{'points':[{'curve_number':0,'point_index':i} for i in points],
                                                     'point_indices':points,'box':[],'lasso':[]}})
        app._run(widget_state=states)
        assert not app.exception
        plot=next(p for p in app.get('plotly_chart') if json.loads(p.proto.spec)['layout'].get('meta',{}).get('scmm_plot')=='volcano')
        return json.loads(plot.proto.spec)['layout']

    first=click([0])
    assert first['meta']['scmm_volcano_labels']==['Alpha','','']
    assert first['meta']['scmm_volcano_annotation_size']==17.
    assert click([0])['meta']['scmm_volcano_labels']==['','','']
    click([0])
    assert click([1])['meta']['scmm_volcano_labels']==['','Beta','']
    assert click([2])['meta']['scmm_volcano_labels']==['','','Gamma']
    app.button(key='volcano_reset_labels').click().run()
    assert not app.exception
    assert app.session_state.volcano_annotation_state['overrides']=={}
