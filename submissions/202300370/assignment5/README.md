# 영어 디베이트 보조 프로그램

## 1. 프로젝트 개요
영어 디베이트 수업 준비를 돕기 위한 보조 도구

입력된 문장을 **찬성 (pro) / 반대 (con) 로 stance를 분류**하는 모델을 학습하여,  
자료 분석 및 토론 준비 시간을 단축하고 반박 포인트를 빠르게 파악할 수 있도록 지원

  + 코드는 copilot을 이용해 생성한 것을 바탕으로 세부사항 및 오류를 직접 수정함

---
## 2. Training
### 데이터셋: 
- 저장 위치: `202300370/assignment4/data-analysis.ipynb`  
- 파일명: `stance_examples1.csv`  
- 구성: `sentence`, `stance`, `topic`  
- 총 문장 수: 269개  
- stance는 pro/con으로 균형 있게 분포되어 있으며, 토픽별로 구분되어 있음

  + 트레이닝 과정에서 사용된 이름은 `stance_examples1.csv`로, 코드상에는 해당 이름으로 지칭함
  + 모델을 학습시킬 때 사용할 여러 가지 예문들을 pro인지 con인지 분류해 둔 상태로, 엑셀 파일 형태로 저장됨

### 학습 과정
- 사전학습 언어모델(예: DistilBERT)을 기반으로 미세조정
- 입력 포맷: `[TOPIC] <topic> [SEP] <sentence>`
- Optimizer: AdamW, Loss: CrossEntropy
- Epochs: 6, Early stopping 적용

### 가중치 저장
- 학습 후 가중치는 구글 드라이브에 저장  
- 위치: [Google Drive 링크](https://drive.google.com/drive/folders/1lrVAAxNIcdFhe7BdV6GU-QR12Et_PiKb?usp=sharing)

---
## 3. Evaluation
- 평가 지표: Accuracy, Macro F1-score
- Confusion Matrix 및 Classification Report를 통해 성능 분석
### 평가 결과 요약
- Accuracy: 0.483
- Macro F1-score: 0.400
- Confusion Matrix:
[[115 15]
[124 15]]

- 실제 con(130개) 중 115개를 맞추고 15개를 pro로 잘못 예측
- 실제 pro(139개) 중 124개를 con으로 잘못 예측하고 15개만 맞춤

### 세부 지표
- **con 클래스**
  - Precision: 0.48
  - Recall: 0.88
  - F1-score: 0.62
- **pro 클래스**
  - Precision: 0.50
  - Recall: 0.11
  - F1-score: 0.18

### 분석
- 모델이 **con 클래스에 치우친 예측**을 하고 있음
- pro 클래스 문장을 거의 잡아내지 못해 Recall이 매우 낮음

---
## 4. Inference
- 저장된 모델 가중치를 불러와 새로운 문장에 대해 stance를 예측
- 입력: 토픽 + 문장
- 출력: stance (pro/con) 및 확률 값

- We should ban gene editing.
   + Gene editing can bring ethical problems.
   + {'stance': 'pro', 'proba_con': 0.468, 'proba_pro': 0.532}

- We should ban gene editing.
   + Gene editing can be exploited by the government.
   + {'stance': 'pro', 'proba_con': 0.492, 'proba_pro': 0.508}

- We should ban gene editing." 
   + Gene editing can help enhancement of human health.
   + {'stance': 'pro', 'proba_con': 0.474, 'proba_pro': 0.526}

- Space exploration is more important than sea exploration.
  + We can find useful resources such as minerals or bacteria while exploring the sea.
  + {'stance': 'con', 'proba_con': 0.509, 'proba_pro': 0.491}

- Animal testing shouldn't be banned.
  + We should appreciate animal rights.
  + {'stance': 'pro', 'proba_con': 0.484, 'proba_pro': 0.516}

---
## 5. 추가 분석

모델이 거의 절반 수준(Accuracy ≈ 0.48)밖에 맞추지 못하고 있고, 특히 pro 클래스에 대한 Recall이 0.11로 매우 낮음.
즉, pro 문장을 거의 잡아내지 못하고 대부분 con으로 예측. 
Confusion Matrix를 보면:
실제 con(130개) 중 115개는 맞췄고, 15개는 pro로 잘못 예측
실제 pro(139개) 중 124개를 con으로 잘못 예측했고, 15개만 맞춤
→ 모델이 **con 쪽으로 치우친 예측**을 하고 있음

+ 그러나 inference에서의 실험으로는 막상 **pro가 많이 등장하는 현상**이 등장

### 원인 분석

- 데이터 불균형/토픽 편향: 특정 토픽에서 pro 문장이 더 많고 con은 부족해서, 모델이 한쪽 클래스에 치우쳐 학습했을 수 있음.
- 데이터셋 크기 부족: 269개 문장은 Transformer 기반 모델을 학습시키기엔 너무 적음 → 과적합/일반화 실패.
- 입력 포맷 문제: 토픽을 [TOPIC] ... [SEP] ... 형태로 넣었지만, 토픽 정보가 모델에 충분히 반영되지 않았을 수 있음.
- 학습 파라미터: Epoch 수, Learning rate, Batch size 등이 최적화되지 않아 underfitting 발생 가능.

### 개선 방법

- 데이터 확장
- 더 많은 문장을 수집 (뉴스 기사, 논문, 토론 자료 등)
- 특히 con 클래스 문장을 늘려 균형 맞추기
- 데이터 증강(paraphrasing, back-translation) 활용

#### 참고 
  + 가중치 저장 드라이브 위치: 
    https://drive.google.com/drive/folders/1lrVAAxNIcdFhe7BdV6GU-QR12Et_PiKb?usp=sharing
