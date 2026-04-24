import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

@st.cache_resource
def load_models():
    from src.inference import bert_predict, baseline_predict
    return baseline_predict, bert_predict

baseline_predict, bert_predict = load_models()

    


st.set_page_config(page_title='Spam Classifier', page_icon='📧')
st.title('SMS Spam Classifier')
st.write('Demo of different ML models to detect spam')
model_name = st.selectbox('Chose model:',
                          ['Baseline (TF-IDF + LogReg)', 'BERT'])

txt = st.text_area('Enter your message', height=100)

if st.button('predict'):
    if txt.strip() == "":
        st.warning('You need to enter something')
    else:
        if model_name == 'BERT':
            pred = bert_predict(txt)
        else:
            pred = baseline_predict(txt)
        if pred == 'spam':
            st.warning('Spam detected!')
        else:
            st.success('Seems ok.')