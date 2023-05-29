import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_data()
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_data()
def get_dataset(file_path):
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))  # 모델의 입력 크기에 맞게 이미지 크기 조정
    img = np.array(img)
    img = img / 255.0  # 이미지 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    return img

def classify_image(image):
    # 분류 모델 불러오기
    model = tf.keras.models.load_model('C:\\Users\\21813903\\Desktop\\chatbot2\\mental-health-chatbot\\my_new_model.h5')
    # 이미지 분류
    prediction = model.predict(image)
    print(prediction)
    if 0.1 >= prediction[0][1]:
        return "Burns"
    else:
        return "frostbite"

model = cached_model()
wellness_df = get_dataset('C:\\Users\\21813903\\Desktop\\chatbot2\\mental-health-chatbot\\wellness_dataset.csv')
symptom_df = get_dataset('C:\\Users\\21813903\\Desktop\\chatbot2\\mental-health-chatbot\\sysmptom_data.csv')

st.header('건강관리 챗봇')
st.markdown("[서울 xx병원](https://www.amc.seoul.kr/asan/healthinfo/symptom/symptomSubmain.do)")

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

if image_input is not None:
    img = load_image(image_input)
    classification = classify_image(img)
    if classification == "Burns":
        st.session_state.past.append("화상 사진")
        st.session_state.generated.append("화상입니다! 흐르는 찬물에 20분이상 담가 유지하세요! 화상 부위에 바세린이나 화상 거즈(깨끗한 거즈)로 덮어주고 붕대로 감아주세요! ")
    else:
        st.session_state.past.append("동상 사진")
        st.session_state.generated.append("동상입니다! 따듯한 곳으로 이동하고 동상부위를 따뜻한 물(39~42도)에 30분간 담그세요!")

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')