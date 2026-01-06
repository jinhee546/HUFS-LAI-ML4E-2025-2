# Assignment 5 — KoBART 기반 한국어 뉴스 요약 모델
201903774 언어인지과학과 한형준

본 과제는 Assignment 4에서 수집·분석한 **네이버 뉴스 한국어 요약 데이터셋**을 기반으로,  
KoBART 모델을 미세조정(fine-tuning)하여 **한국어 뉴스 기사 → 요약문**을 생성하는 모델을 학습하고 평가하는 것을 목표로 한다.

- 과제 목표  
  - 프로젝트 주제(강의 슬라이드 요약 서비스 “BonCahier AI”)에 맞는 **요약 모델 아키텍처 선택**
  - Assignment 4에서 정리한 **`document` → `summary` 구조 데이터를 사용한 학습**
  - Train / Validation / Test 분할 및 **ROUGE 기반 평가 프로토콜 정의**
  - 학습된 모델 가중치를 Google Drive에 저장하고,  
    `evaluation.ipynb` / `inference.ipynb`에서 재사용 가능한 구조 구현

---

## 1. 전체 파일 구조

이 과제 디렉토리는 다음과 같이 구성되어 있다.

```text
assignment5/
├── training.ipynb      # 모델 학습 및 가중치 저장
├── evaluation.ipynb    # Test set 평가 (ROUGE 계산 + 예시 출력)
├── inference.ipynb     # 저장된 모델 로드 + 새로운 입력에 대한 요약 예시
└── README.md           # 본 문서: 프로젝트 요약 및 결과 정리
```

세 개의 Notebook은 모두 동일한 모델 디렉토리를 공유하도록 설계했다.

- 공통 모델 경로(예시):  
  `/content/drive/MyDrive/boncahier/models/kobart_ko_news`

`training.ipynb`에서 이 디렉토리에 모델을 저장하고,  
`evaluation.ipynb`와 `inference.ipynb`에서 같은 경로를 지정해 모델을 로드한다.

---

## 2. 데이터셋

### 2.1 데이터 출처 및 구조

Assignment 4에서 사용한 **네이버 뉴스 한국어 요약 데이터셋**을 그대로 활용하였다.

- 원본: HuggingFace `daekeun-ml/naver-news-summarization-ko`
- Assignment 4에서 저장한 CSV:  
  `data/naver_news_summarization_ko.csv`
- 주요 컬럼:
  - `document`: 뉴스 기사 본문 (입력 텍스트)
  - `summary`: 사람이 작성한 요약문 (타깃 텍스트)
  - 그 외 `date`, `category`, `press`, `title`, `link` 등은 이번 학습에서는 사용하지 않음

Assignment 4에서 EDA를 통해 확인한 주요 통계는 다음과 같다(Train 전체 기준):

- 샘플 수: 약 22,000개
- 본문 평균 길이: 약 1,000자
- 요약 평균 길이: 약 180자
- 평균 압축 비율: 약 0.42 (요약이 본문 길이의 약 40% 정도)

이 구조는 **“긴 텍스트 → 짧은 요약”**이라는 BonCahier AI의 요구사항과 잘 맞는다.

### 2.2 Train / Validation / Test 분할

`training.ipynb`에서 CSV를 읽어온 뒤, HuggingFace `Dataset`으로 변환하고 다음 기준으로 분할하였다.

- 전체 데이터를 셔플(shuffle)한 뒤:
  - Train: 80%
  - Validation: 10%
  - Test: 10%
- 랜덤 시드: `seed = 42` (재현 가능성 보장)

실험 시간 및 Colab 자원 제한을 고려하여, 학습에는 전체 데이터의 일부만 사용하는 옵션을 두었다.

- 학습에 사용한 샘플 수(예시):
  - Train: 최대 약 5,000개
  - Validation: 최대 약 1,000개
  - Test: 최대 약 1,000개

(정확한 수치는 `training.ipynb` / `evaluation.ipynb` 실행 로그에서 확인 가능)

---

## 3. 모델 및 학습 설정

### 3.1 모델 아키텍처

- 베이스 모델: **`gogamza/kobart-base-v2`**
  - 한국어 BART 아키텍처(KoBART)
  - Transformer 기반 encoder–decoder 구조
  - 사전학습(pretrained)된 한국어 생성 모델로, 요약 / 번역 등의 sequence-to-sequence 태스크에 적합

- 입력/출력 정의:
  - Encoder 입력: `document` (뉴스 기사 본문)
  - Decoder 타깃: `summary` (요약문)

### 3.2 토크나이저 및 전처리

- 토크나이저: `AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")`
- 최대 길이 설정:
  - `MAX_SOURCE_LENGTH = 512`
  - `MAX_TARGET_LENGTH = 128`
- Padding & Truncation:
  - 입력/타깃 모두 `max_length` 기준으로 `padding="max_length"`, `truncation=True`
- 레이블 처리:
  - 요약 토큰 시퀀스에서 `pad_token_id`를 `-100`으로 치환하여  
    loss 계산 시 padding이 무시되도록 설정

### 3.3 학습 하이퍼파라미터

(Assignment 제출 시점에 사용한 **최종 설정**)

- Optimizer / Trainer: HuggingFace `Seq2SeqTrainer`
- Batch size:
  - `per_device_train_batch_size = 4`
  - `per_device_eval_batch_size = 4`
  - `gradient_accumulation_steps = 4`  → 실효 batch size ≈ 16
- Epoch:
  - `NUM_TRAIN_EPOCHS = 4`
- Learning rate:
  - `LEARNING_RATE = 3e-5`
- 기타:
  - weight decay: 0.01
  - fp16: Colab T4 GPU 사용 시 자동 활성화
  - `seed = 42`

초기에는 2 epoch, 5e-5 설정으로 학습을 시도했으나,  
eval loss 및 ROUGE 개선을 위해 epoch 수를 늘리고, learning rate를 낮추고,  
gradient accumulation을 도입하는 방향으로 하이퍼파라미터를 조정하였다.  
또한 Assignment 4의 EDA 결과를 바탕으로, 극단적인 길이/압축비를 가진 샘플을 제거하는  
간단한 전처리도 함께 적용하였다.

### 3.4 학습 환경

- 실행 환경: Google Colab
- GPU: Tesla T4
- 프레임워크:
  - `transformers`
  - `datasets`
  - `evaluate`
  - `sentencepiece`

---

## 4. 평가 프로토콜

### 4.1 평가 데이터 및 절차

- 데이터: Train/Validation/Test 분할 시 생성된 **Test set** (약 1,000개 샘플)
- 과정:
  1. `evaluation.ipynb`에서 `MODEL_DIR`에 저장된 KoBART 모델 로드
  2. Test set을 동일한 토크나이징 규칙으로 변환
  3. `Seq2SeqTrainer.evaluate()`를 이용하여
     - `eval_loss`
     - ROUGE 스코어(ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
     를 계산

### 4.2 평가 지표

텍스트 요약 태스크의 특성상 **Accuracy/F1**보다  
**n-gram 기반 유사도를 측정하는 ROUGE 지표가 더 적절하다고 판단하였다.

- ROUGE-1: unigram 기반 정밀도/재현율의 조화 평균
- ROUGE-2: bigram 기반 지표
- ROUGE-L: 최장 공통 부분수열(LCS)에 기반한 문장 레벨 유사도
- ROUGE-Lsum: 전체 요약에 대한 LCS 기반 유사도

모든 ROUGE 값은 0~100 스케일로 변환하여 보고하였다.

---

## 5. 실험 결과

`evaluation.ipynb`에서 얻은 대표적인 평가 결과는 다음과 같다.  
(약 1,000개 Test 샘플 기준)

| 지표          | 값        |
|--------------|-----------|
| eval_loss    | **0.505** |
| ROUGE-1      | **35.62** |
| ROUGE-2      | **14.46** |
| ROUGE-L      | **34.99** |
| ROUGE-Lsum   | **34.90** |

- `eval_runtime ≈ 81.45초`, `eval_samples_per_second ≈ 12.28`으로,  
  T4 GPU 환경에서 실용적인 속도로 평가가 이루어짐을 확인했다.
- ROUGE-1이 약 35~36 수준으로, KoBART를 활용한 한국어 뉴스 요약 태스크에서  
  **베이스라인으로 보기 충분한 성능**이라고 판단된다.

### 5.1 결과 해석

- ROUGE-1 / ROUGE-L이 35 전후로 형성되어,  
  원문에서 중요한 키워드와 문장 구조를 어느 정도 잘 유지하는 것으로 보인다.
- ROUGE-2가 14 수준으로 다소 낮은 편인데,  
  이는 문장 내부의 세부 bigram 패턴(자연스러운 표현, 문맥 흐름)에 대한 개선 여지가 있음을 의미한다.
- epoch 수, learning rate, gradient accumulation, 이상치 제거 등 기본적인 튜닝을 적용한 뒤에도  
  추가적인 성능 향상을 위해서는 beam search 설정이나 도메인 적응 등 더 고급 기법을  
  시도해 볼 여지가 남아 있다.

---

## 6. Inference: 실제 요약 예시

`inference.ipynb`에서는 학습된 모델을 로드한 뒤,  
직접 정의한 `summarize(text)` 함수를 이용해 새로운 입력 텍스트를 요약한다.

### 6.1 요약 함수 구조

```python
def summarize(text, max_source_length=MAX_SOURCE_LENGTH, max_target_length=MAX_TARGET_LENGTH):
    inputs = tokenizer(
        text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )

    # KoBART/BART 계열은 token_type_ids를 사용하지 않으므로 제거
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary
```

### 6.2 예시 입력 유형

`inference.ipynb`에서는 다음과 같은 세 가지 유형의 입력에 대해 요약 결과를 확인했다.

1. **네이버 뉴스 스타일 기사 문단**  
   - Assignment 4에서 사용했던 실제 뉴스 기사와 유사한 구조
2. **“지도학습 vs 비지도학습”을 설명하는 강의 슬라이드 스타일 텍스트**  
   - 향후 BonCahier AI의 실제 타깃 도메인(강의 자료 요약)에 대한 예시
3. **인공지능 기반 요약 시스템에 대한 설명**  
   - 프로젝트 자체를 설명하는 텍스트를 다시 모델이 요약하도록 하는 메타적인 예시

각 예시에서 모델은 전체 문단의 핵심 주제를 유지하면서도  
원문보다 짧은 길이의 요약을 생성하는 모습을 보였다.

---

## 7. 모델 가중치 저장 위치

- 학습된 KoBART 요약 모델 가중치는 Google Drive에 저장하였다.
- 경로 예시:
  - `/content/drive/MyDrive/boncahier/models/kobart_ko_news`
- `training.ipynb`에서 `trainer.save_model(OUTPUT_DIR)` 및  
  `tokenizer.save_pretrained(OUTPUT_DIR)`를 통해 저장하고,
- `evaluation.ipynb`와 `inference.ipynb`에서는 동일한 `MODEL_DIR` 경로를 사용하여 모델을 로드한다.

> PR Description에는 이 경로 또는 공유 가능한 Google Drive 링크를 함께 기재하였다.

---

## 8. 한계점 및 향후 개선 방향

- **학습 자원 제한**으로 인해:
  - 전체 데이터와 에폭 수를 무제한으로 늘리지는 못하였다.
  - 그럼에도 불구하고, 본 과제에서는 다음과 같은 튜닝을 실제로 적용하였다.
    1. epoch 수를 4로 증가시키고, validation 성능을 모니터링하며 학습 안정성 확인
    2. learning rate를 3e-5 수준으로 낮춰 더 안정적인 수렴 유도
    3. gradient accumulation을 활용하여 실효 batch size를 키워 loss 곡선을 더 매끄럽게 유지
    4. Assignment 4에서 수행한 EDA 결과를 바탕으로  
       극단적인 길이/압축비를 가진 outlier 샘플을 제거하는 전처리 강화

- **향후 추가적으로 시도할 수 있는 개선 방향**은 다음과 같다.
  1. beam search 설정(`num_beams`, `length_penalty`)을 보다 체계적으로 튜닝하여  
     과도하게 긴/짧은 요약을 줄이고, ROUGE-2 및 ROUGE-L을 추가로 개선
  2. BERTScore 등 의미 기반 평가 지표를 함께 도입하여,  
     표면적 n-gram 일치율뿐만 아니라 의미적 유사도까지 함께 평가
  3. 실제 강의 슬라이드(PPT/PDF) 텍스트를 본격적으로 수집하여,  
     BonCahier AI의 최종 타깃 도메인(강의 자료 요약)에 대한 도메인 적응(fine-tuning) 수행
  4. 뉴스 도메인 외의 다양한 한국어 텍스트(블로그, 리포트 등)를 추가로 학습하여  
     보다 다양한 문체와 주제에 대응할 수 있는 요약 모델로 확장

그럼에도 불구하고, 본 Assignment 5에서는

- **데이터 수집/EDA (Assignment 4)** →  
- **KoBART 요약 모델 학습/평가/추론 (Assignment 5)**

까지의 파이프라인을 end-to-end로 구현하고,  
기본적인 하이퍼파라미터 튜닝과 전처리를 포함한 모델 개선 과정을 수행했으며,  
결과를 정량적인 지표(ROUGE)와 예시 출력으로 확인했다는 점에서  
과제의 요구사항을 충실히 달성했다고 판단한다.
