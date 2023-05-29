# 건강 특화 챗봇
현대인들의 지친 마음을 상담할 수 있는 심리 상담 기능과 사용자의 증상을 인지하고 이비인후과 질환을 예측하는 기능을 가지고 있다. 또한 사용자들에게 화상과 동상 이미지를 입력 받아 화상인지 동상인지 알려주는 기능을 가지고 있다.

## 1. 심리 상담 기능
SentenceBERT 모델을 사용하였음
-> 한국어 문장 처리 모델 [SentenceBERT](https://huggingface.co/jhgan/ko-sroberta-multitask)
![챗봇 심리상담 기능](https://github.com/younga13/mental-health-chatbot/assets/129020528/b7b052f2-b03c-49de-b6f2-9ba3ed9b56ac)

[사용 데이터]
* [웰니스 대화 스크립트 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&dataSetSn=267&aihubDataSe=extrldata)

## 2. 이비인후과 질환 예측
심리 상담 기능과 같은 모델로 구현하였음. 데이터는 직접 만들어서 사용하였음.
![이비인후과 질환 챗봇](https://github.com/younga13/mental-health-chatbot/assets/129020528/4ed84754-a22a-4d3f-b935-ff3a208fd2ec)

[참고 데이터]
* [서울 아산병원](http://ent.amc.seoul.kr/asan/depts/ent/K/disease.do?menuId=4076)

## 3. 화상 동상 이미지 분류 기능
인공지능 모델을 이용하여 화상, 동상 이미지를 학습 시킴. 
-> ResNet50으로 하였으나 정확도가 너무 낮아 모델을 변경함

![화상 동상 기능 챗봇](https://github.com/younga13/mental-health-chatbot/assets/129020528/6600f897-80cf-44c5-b46f-8252ccbbb084)

[참고 모델]
* [ResNet50](https://github.com/younga13/080263/tree/master/chap5)

## 4. 개발자 매뉴얼

* 1. [텍스트 쳇봇 데이터 추가 사용](https://github.com/younga13/mental-health-chatbot/wiki/%EA%B0%9C%EB%B0%9C%EC%9E%90-%EB%A7%A4%EB%89%B4%EC%96%BC%7C-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EA%B8%B0%EB%8A%A5-%EC%B6%94%EA%B0%80)
* 2. [이미지 챗봇 데이터 추가 사용](https://github.com/younga13/mental-health-chatbot/wiki/%EA%B0%9C%EB%B0%9C%EC%9E%90-%EB%A7%A4%EB%89%B4%EC%96%BC%7C-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EA%B8%B0%EB%8A%A5-%EC%B6%94%EA%B0%80)

## 5. 협력자
*  @leegangryong1  gangryong.hci.du@gmail.cpm

## Dependency

- streamlit
- streamlit-chat
- pandas
- sentence-transformers
- scikit-learn
