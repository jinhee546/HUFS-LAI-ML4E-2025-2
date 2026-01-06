## 1. 프로젝트 개요
대규모 다국어 모델인 **mBART-large-cc25**를 특정 도메인의 한국어-영어 번역 태스크에 맞게 파인튜닝하여 모델의 성능을 최적화하고, 다양한 데이터셋(Validation/Test)에서의 일반화 성능을 분석했다. 

---

## 2. Training (학습) 상세 내용

### 2.1. 모델 설계의 적절성

* **문제 유형:** 영어-한국어 신경망 기계 번역 (NMT)
* **모델 선택:** **mBART-large-cc25** 모델을 채택했다. 이는 다양한 언어 쌍에 대해 사전 학습된 **트랜스포머 기반의 Seq2Seq** 모델로, NMT 태스크, 특히 한국어와 같은 저자원 언어 쌍의 학습 효율을 높이는 데 적합하다.

### 2.2. 데이터셋 분할 및 전처리

AI HUB 데이터셋을 활용하여 모델 학습에 필요한 데이터셋을 구성했다.

| 데이터셋 | 샘플 개수 | 용도 |
| :--- | :--- | :--- |
| **Training Set** | 20,000쌍 | 모델 파인튜닝 |
| **Validation Set** | 2,000쌍 | 학습 중 성능 모니터링 |
| **Test Set** | 435쌍 (강의 자료) | 최종 성능 평가 |

* **전처리:** `AutoTokenizer`를 사용하여 입력과 타겟 시퀀스를 최대 길이 50으로 제한했고, 소스 언어는 `"en_XX"`, 타겟 언어는 `"ko_KR"`로 설정했다.

### 2.3. 학습 환경 및 재현성

* **주요 하이퍼파라미터:** 총 3 Epoch 동안 학습하며, 배치 크기는 8, `save_steps`는 1500으로 설정했다.
* **학습 추이:** Training Loss는 초기 3.7506에서 최종 0.6726까지 지속적으로 감소하는 경향을 확인했다.
* **모델 저장 위치:** 최종 모델 가중치와 토크나이저는 Google Drive의 **`/content/drive/MyDrive/AIHUB_DATA/final_nmt_model_1500`** 경로에 저장했다.
- **가중치 저장 위치 (Google Drive)** : https://drive.google.com/drive/folders/1h7E7BVUraKPLw69ltUiSjvMssa8oU6Fw?usp=sharing
---

## 3. Evaluation (평가) 상세 내용

### 3.1. 평가 지표 및 계산

* **평가 지표:** 기계 번역의 표준 성능 지표인 **BLEU Score (sacreBLEU)**로 생성된 번역과 참조 번역 간의 n-gram 일치도를 측정했다.
* **디코딩 전략:** 최종 평가 시 **탐욕적 디코딩(Greedy Decoding, `num_beams=1`)** 및 최대 생성 길이 50을 적용했다.

### 3.2. 주요 성능 수치 요약

| 지표 | Validation Data (2,000쌍) | Test Data (435쌍) |
| :--- | :--- | :--- |
| **Loss (손실)** | 1.1770 | **2.4105** |
| **BLEU Score** | 24.14 | **7.61** |

---

## 4. Inference (추론) 평가

### 4.1. 모델 가중치 및 추론 환경

* **모델 로드:** 저장된 모델을 지정된 Google Drive 경로에서 로드하여 추론을 수행했다.
* **추론 대상:** 강의 자료 PDF 3개 파일(`lec11_oop.pdf`, `ml4e-lecture-week13.pdf`, `NLP_11.pdf`)의 1~5 페이지 텍스트를 추출하여 사용했다.
* **재현성:** `inference.ipynb`를 통해 모델 로드, PDF 텍스트 전처리, 추론, 결과 출력이 모두 독립적으로 실행 가능하다.

### 4.2. 예시 추론 결과

| Source (EN) | Translation (KO) |
| :--- | :--- |
| Sequence Modeling Sequence modeling is the task of predicting what comes next E. g. | 퀀스 모델링 퀀스 모델링은 다음(E)에 어떤 일이 발생할지를 예측하는 작업이다. g. |
| Lecture 11-2 Neural Nets-Feedforward nets can only handle inputs and outputs that have a fixed size-Recurrent Neural Nets (RNNs) handle variable length sequences (as | 11-2 신경망-Feed 엔진은 고정된 크기를 갖는 입력 및 출력만 처리할 수 있고 고정된 크기를 갖는 반복 신경망(RNN)은 변수 길이 시퀀스 |

---

## 5. 특이사항 및 한계

* **도메인 불일치(Domain Mismatch) 문제:** Test Set의 BLEU Score(7.61)가 Validation Set(24.14) 대비 크게 낮게 측정되었다. 이는 Test Set으로 사용된 강의 자료 데이터가 훈련 데이터의 일반적인 도메인과 달라 발생하는 **도메인 일반화 성능의 부족**을 시사한다.
* **데이터 제약 및 과적합:** 학습 데이터의 규모(20,000쌍)가 상대적으로 제한되어 있으며, Validation Loss(1.1770) 대비 Test Loss(2.4105)가 높아 Test Set에서 모델이 기대만큼 일반화되지 못한 것으로 판단된다.
* **PDF 텍스트 추출/정제 오류:** PDF 레이아웃 특성상 텍스트 추출 시 문장이 불완전하게 연결되거나 불필요한 잔여 기호가 포함되어 번역 품질에 부정적인 영향을 미쳤다.
