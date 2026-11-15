import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from scMM.i18n import languages, translate, localize_figure


def test_language_packs_keep_template_placeholders_and_can_be_extended(tmp_path, monkeypatch):
    packs = languages()
    assert {'zh_CN', 'en'} <= set(packs)
    assert translate('物质注释', 'en') == 'Compound annotation'
    assert translate('○ **数据**', 'en') == '○ **Data**'
    assert translate('当前数据层：raw。切换后可重新运行降维。', 'en').startswith('Current data layer: raw')
    for source, target in packs['en']['messages'].items():
        assert sorted(re.findall(r'\{\d+\}', source)) == sorted(re.findall(r'\{\d+\}', target))
    (tmp_path/'fr.json').write_text(json.dumps({'code':'fr','name':'Français','messages':{'图片导出':'Exporter la figure'}}),encoding='utf8')
    monkeypatch.setenv('SCMM_LANGUAGE_DIR',str(tmp_path))
    assert languages()['fr']['name'] == 'Français'
    assert translate('图片导出', 'fr') == 'Exporter la figure'


def test_switching_language_preserves_selected_options_and_parameters():
    script = '''
from scMM.webapp import render_app
render_app()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    mode = next(w for w in app.radio if w.label == '输入模式')
    mode.set_value(mode.options[1]).run()
    ppm = next(w for w in app.number_input if w.label == '特征匹配 ppm')
    key = ppm.key
    ppm.set_value(18.).run()
    app.selectbox(key='ui_language').set_value('en').run()
    assert not app.exception
    assert any('Data import' in w.value for w in app.header)
    assert app.number_input(key=key).value == 18.
    assert next(w for w in app.radio if w.key == mode.key).value == '处理单个 mzML / mzXML / RAW 文件'
    assert not any(re.search('[\u4e00-\u9fff]', w.label) for w in app.number_input)
    app.selectbox(key='ui_language').set_value('zh_CN').run()
    assert not app.exception
    assert app.number_input(key=key).value == 18.


def test_english_analysis_retains_original_dataset_values():
    app = AppTest.from_string('''
import streamlit as st
import numpy as np, pandas as pd
from scMM.file.data import CyESIData
from scMM.webapp import _state, page_analysis
_state(st)
st.session_state.ui_language='en'
if st.session_state.data is None:
 st.session_state.data=CyESIData(X=np.random.default_rng(3).random((20,4)),obs=pd.DataFrame({'group':['对照']*10+['处理']*10},index=[str(i) for i in range(20)]))
page_analysis()
''', default_timeout=30).run()
    assert not app.exception
    assert app.session_state.data.obs.group.tolist() == ['对照']*10+['处理']*10
    for kind in ['header','subheader','button','number_input']:
        for element in app.get(kind):
            text = getattr(element,'label',None) or getattr(element,'value','')
            assert not re.search('[\u4e00-\u9fff]', text), text
