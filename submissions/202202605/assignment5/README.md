본 프로젝트에서는 KLUE-BERT-base 모델을 기반으로 한 이진 분류 모델을 구축하여, 한국어 단어가 쉬운 단어(0) 또는 어려운 단어(1) 중 어디에 속하는지를 예측하도록 설계했다.


### 1. 아키텍쳐 구성

모델: KLUE-BERT-base (klue/bert-base)

선정 이유:
- 한국어 데이터에 특화된 KLUE 언어모델

모델 구조 요약:
- 입력 단어 → Tokenizer(max_length=32)
- BERT Encoder
- [CLS] 토큰 임베딩 → Dropout → Linear Layer (2-class)
- Loss: CrossEntropy
- Optimizer: AdamW

### 2. 학습 데이터

- 총 데이터 개수: 3,715개
- 라벨 구성:
  - 0 = 쉬운 단어
  - 1 = 어려운 단어
- Train/Validation/Test 비율:
  - Train 80%
  - Validation 10%
  - Test 10%

### 3. 학습 (Training)

- Epoch: 5
- Batch Size: Train=16 / Eval=64
- Learning Rate: 2e-5
- Optimizer: AdamW
- load_best_model_at_end=True로 설정하여 최적 모델을 저장


### 4. 최종 Test Set 성능

- Accuracy: 77.69%

| 라벨       | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| 0(어려움) |   0.58    |  0.59  |  0.59    |   100   |
| 1(앎)     |   0.85    |  0.85  |  0.85    |   272   |


Confusion Matrix 요약
실제 0 → 예측 1: 41
실제 1 → 예측 0: 42

https://drive.google.com/drive/folders/1oJOY1WUbx0Jd8e8v8EmIAh6uiFexDEut?usp=sharing
