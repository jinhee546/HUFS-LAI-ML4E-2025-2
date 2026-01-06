# Assignment 5: Model Training and Evaluation

## 1. 프로젝트 개요
- **프로젝트명**: 항공 뉴스 캡션 자동 생성기
- **목표**: 항공 스케줄 뉴스(AeroRoutes)에서 핵심 정보(항공사, 노선, 기종, 일정)를 추출하여 인스타그램용 한국어 캡션을 자동 생성하는 AI 모델 개발.

## 2. 모델 아키텍처 및 학습 전략
- **모델**: `bert-base-multilingual-cased` 기반 **Token Classification (NER)**
- **학습 데이터**: AeroRoutes 기사 1,000개 (Assignment 4 수집 데이터)
- **라벨링 방식**:
    - **Weak Labeling (Rule-based)**: 정규식(Regex)과 국토교통부 항공 데이터(CSV)를 활용하여 `B-AIRLINE`, `B-AIRCRAFT`, `B-DATE`, `I-ROUTE` 태그를 자동 생성.
    - **Subword Handling**: BERT 토크나이저의 Subword(`##`)까지 라벨을 전파하여 학습 안정성 확보.

## 3. 평가 결과 (Evaluation)
- **Validation Set**: 학습 데이터의 15% (약 150개) 사용
- **성능 지표 (F1-Score)**:
    - **Overall**: **0.94** (매우 우수)
    - **AIRCRAFT**: 0.98 (기종 패턴이 명확하여 인식률 최상)
    - **AIRLINE**: 0.93 (국토부 CSV DB 활용 효과)
    - **ROUTE**: 0.89 (다양한 도시 이름으로 인해 상대적으로 낮음)

## 4. 추론 (Inference)
- **Hybrid System**:
    - 모델이 놓치는 정보(날짜 패턴, 복잡한 노선)를 **정규식(Regex)으로 2차 보완**하여 재현율(Recall)을 100%로 끌어올림.
- **Knowledge Base 연동**:
    - 국토교통부 공항/항공사 코드(IATA) CSV 데이터를 활용하여, 추출된 영문 코드를 **정확한 한글 명칭(예: ICN -> 인천국제공항)**으로 변환.
    - 코드쉐어(공동운항) 여부 자동 감지 및 캡션 템플릿 분기 처리.

## 5. 모델 가중치 (Weights)
- **Google Drive Link**: https://drive.google.com/drive/folders/17-g8otxAMhDxbsnyuXeykRUnJ1xP01bz?usp=drive_link
