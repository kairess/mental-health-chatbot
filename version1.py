import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
import json
from datetime import datetime
import os

@st.cache_data()
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_data()
def get_dataset(file_path):
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df
import tensorflow as tf
load_model = tf.keras.models.load_model("C:\\Users\\21813903\\Desktop\\chatbot2\\mental-health-chatbot\\my_model.h5")
image_height =224
image_width=224

#오류확인
#model = keras.models.load_model(model.path)
import tempfile
from PIL import Image
import io

def classify_image(image):
    # 이미지 전처리
    #image = tf.image.resize(image, (image_height, image_width))
    #image = image / 255.0
    #image = np.expand_dims(image, axis=0)

    #with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
    #   temp_image.write(image)
    #    temp_image_path = temp_image.name

   # img = tf.keras.preprocessing.image.load_img(temp_image_path, target_size=(image_height, image_width))
    img = Image.open(io.BytesIO(image))
    try:
        img.verify()
    except Exception as e:
        # 유효하지 않은 이미지
        return "유효하지 않은 이미지"
    img = img.resize((image_width, image_height))

    path='C:\\Users\\21813903\\Desktop\\KakaoTalk_20230525_144443034.jpg'
    x = tf.keras.preprocessing.image.img_to_array(img)
    #img=tf.keras.preprocessing.image.load_img(temp_image, target_size=(image_height,image_width))
    #x=tf.keras.preprocessing.image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images=np.vstack([x])
    classes=load_model.predict(images, batch_size=10)

    # 이미지 분류
    #prediction = load_model.predict(image)
    #print(prediction)
    class_names = ['고양이', '강아지']
    classes = classes.tolist()
    probabilities = classes[0]

    threshold = 0.5

    if probabilities[0] > threshold:
        return class_names[0]
    elif probabilities[1] > threshold:
        return class_names[1]
    else:
        return "분류 결과 없음"

    #if classes[0][0] > classes[0][1]:
    #    return class_names[0]
    #    #return classes
    #else:
    #   return class_names[1]
        #return classes
    
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


if image_input is not None:
    image = mpimg.imread(image_input)
    classification = classify_image(image)
    if classification == "강아지":
        st.session_state.past.append("강아지 사진")
        #st.session_state.generated.append("입력한 이미지는 강아지입니다."+str(classification[0][0])+"/"+str(classification[0][1]))
        st.session_state.generated.append("입력한 이미지는 강아지입니다.")

    else:
        st.session_state.past.append("고양이 사진")
        #st.session_state.generated.append("입력한 이미지는 고양이입니다."+str(classification[0][0])+"/"+str(classification[0][1]))
        st.session_state.generated.append("입력한 이미지는 고양이입니다.")

""" if image_input is not None:
    image = mpimg.imread(image_input)
    classification = image_chatbot(image)
    if classification == "dog":
        st.session_state.past.append("강아지 사진")
        st.session_state.generated.append("강아지 사진입니다")
    else:
        st.session_state.past.append("고양이 사진")
        st.session_state.generated.append("고양이 사진입니다")
 """

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')