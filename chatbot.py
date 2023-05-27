import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache_data()
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_data()
def get_dataset(file_path):
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
wellness_df = get_dataset('C:\\Users\\21813903\\Desktop\\chatbot2\\mental-health-chatbot\\wellness_dataset.csv')
symptom_df = get_dataset('C:\\Users\\21813903\\Desktop\\chatbot2\\mental-health-chatbot\\sysmptom_data.csv')

st.header('심리상담 챗봇')
st.markdown("[❤️빵형의 개발도상국](https://www.youtube.com/c/빵형의개발도상국)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    image_input = st.file_uploader('이미지 업로드:', type=['jpg'], accept_multiple_files=False)
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    wellness_df['distance'] = wellness_df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    symptom_df['distance'] = symptom_df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

    wellness_answer = wellness_df.loc[wellness_df['distance'].idxmax()]
    symptom_answer = symptom_df.loc[symptom_df['distance'].idxmax()]

    if wellness_answer['distance'] > symptom_answer['distance']:
        answer = wellness_answer
    else:
        answer = symptom_answer

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
