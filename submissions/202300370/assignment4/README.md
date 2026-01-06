# **4th Assignment: Data Collection and Analysis**

## **1. 데이터수집**
Assignment 3에서 제안한 디베이트 보조 프로그램을 학습시키기 위한 예시 sentences와 각각의 예시에 해당하는 stance와 topic을 함께 묶어둔 csv 파일을 준비. 
실제로 과거 디베이트 수업 당시 사용했던 자료들을 이용해 엑셀 파일을 통해 정리한 것으로, 실용성을 높일 수 있도록 하는 방향으로 데이터 품질을 고려.
sentence, stance, topic이라는 레이블로 묶여 있어 모델이 서로 다른 topic의 예시들을 헷갈리거나 각각의 stance에 혼란이 오지 않도록 함.
처음에는 이 셋을 한 군데에 묶어서 분류해 뒀었으나, 코딩의 과정에서 모델이 이를 구별해 내는데에 어려움을 겪어 별도의 레이블을 묶어두는 형식으로 정리.

**예시 수:** 269개

**header:** sentence, stance, topic

## **2. 분석 요약**
+ **데이터 크기:** 총 269개 문장, 3개 컬럼 (sentence, stance, topic)
+ **stance 분포:** pro와 con이 전체적으로 균형 있게 분포
+ **topic 별 특징:** 
  -Climate change is the greatest threat to humanity: pro 문장이 더 많음 -> 환경 위기 논의에 대해 찬성 논거가 풍부
  -Physical Shops vs Online shops: pro/con이 비슷 ->실제 사회적 논쟁의 균형 반영

### **추가 EDA 결과:**
+문장 길이 분포: 대부분 10-25 단어 사이, 평균 길이는 약 18단어
+상위 단어 빈도: exploration, space, ai, climate, social, media 등이 자주 등장 -> 주제별 핵심 키워드 반영

## **3. 부족한 점이나 개선할 점**
미리 간단하게 모델의 stance clarification을 학습시켜 본 결과 아직은 정확도가 조금 떨어짐. 미리 학습한 데이터와 비슷한 주제 내지는 빈번히 등장하는 stance들은 쉽게 분류해낼 수 있는 반면 여전히 오류도 많이 나타남.
광범위한 기사 자료들을 스크랩 해 와서 학습시킴으로써 향후 개선할 수 있는 여지가 보임. 



