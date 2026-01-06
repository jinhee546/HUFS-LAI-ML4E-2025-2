*주제 변경

ML 기반 이민 국가 추천 시스템 (ML-Powered Immigration Destination Recommender)

1. 프로젝트 개요 (Project Overview)

1.1 프로젝트 주제

머신러닝 기반 이민 국가 추천 시스템 (ML-Powered Immigration Destination Recommender)

1.2 목표

사용자(잠재적 이민자)가 중요하게 생각하는 경제적 안정성 및 사회적 포용성(소수자 인식) 지표를 종합적으로 분석합니다.

이민자나 인권에 관한 정책이 성소수자에 대한 정책의 포용성과 관련이 있다는 전제 하에 주제를 선정하였습니다.

머신러닝 모델을 활용하여 데이터 기반의 최적 이민 대상 국가를 정량적으로 추천하는 지능형 시스템을 개발합니다.

1.3 핵심 특징 (Machine Learning Emphasis)

본 프로젝트는 단순 지표 필터링이 아닌, 다차원적인 국가 지표 간의 복잡한 상관관계를 학습하고 사용자의 선호도(가중치)에 가장 적합한 국가 순위를 산출하는 지도/비지도 학습 기법을 핵심적으로 사용합니다.


2. 데이터 수집 및 통합 (Data Collection & Integration)

이민 결정에 결정적인 영향을 미치는 국가 지표를 수집하고 통합하여 단일 데이터셋을 구축했습니다.

2.1 데이터 소스 및 지표

- worldbank_data_2013_2022.csv(코드 내에서 다운로드) - 국가 경제 및 거버넌스 지표 (World Bank) | 2013년 ~ 2022년 | 1인당 GDP, 군사비 지출 비중, 정부 효과성(Government Effectiveness)
- lgbt-military-equaldex.csv - 사회적 포용성 지표 (Equaldex, LGBT 권리) | ~ 2022년 | 군 복무 허용 여부 (소수자 포용성의 간접 지표)  
- 한국 현황 정리.csv - 기준 국가(한국)의 상세 지표 | 2013년 ~ 2022년 | 벤치마킹 및 비교 분석을 위한 지표
- 코드 내 뉴스기사 스크레이핑 | 감성분석을 위한 실제적 언어 데이터

- worldbank_data_2013_2022.csv: 701개 데이터 존재
- lgbt-military-equaldex.csv: 총 데이터는 2566개, 그러나 코드 내에서 범위를 2013 ~ 2022년으로 한정하여 659개 데이터 활용
- 코드 내 뉴스기사 스크레이핑: 총 데이터는 566개, 각 카테고리별로 수집함, Ambiguous 94개 | "Don't ask, don't tell" 107개 | Legal 89개 | LGB permitted, transgender people banned 109개 | Illegal 111개 | South Korea(별도 분석용) 56개 


2.2 데이터 전처리 및 통합

데이터 통합: 국가 코드(Code)와 연도(Year)를 기준으로 모든 CSV 파일을 단일 데이터프레임으로 병합했습니다.

결측치 처리 (Imputation): 누락된 데이터는 주변 연도 값의 평균 또는 중앙값을 활용하여 대체 처리했습니다.

범주형 인코딩: 'Serving openly in the military'와 같은 범주형 지표는 머신러닝 모델 학습을 위해 One-Hot Encoding을 적용하여 수치화했습니다.


3. 탐색적 데이터 분석 (EDA Findings)

주요 분석 결과 및 문제점 

국가 안정성: 1인당 GDP와 정부 효과성 지표는 높은 양의 상관관계를 보여, 두 지표를 묶어 '국가 안정성' 요소로 모델에 활용할 수 있음을 확인했습니다.

독립 변수 확인: 군사비 지출 비중은 경제 규모(GDP)와 독립적인 특성을 보여, 국가의 재정 건전성을 평가하는 개별적인 변수로 활용할 계획입니다.

범주형 인권 지표의 수치화 필요: Serving openly in the military 값은 머신러닝에 직접 입력 불가이므로 → Label Encoding / One-hot Encoding을 반드시 적용해야 합니다.

국가 군집 그룹 가능성 확인: K-Means 초기 테스트 결과에 따르면 군 복무 인권이 Legal인 국가들은 정부 효율성·GDP가 높은 그룹에 분포하므로 Legal 여부가 ‘경제·행정 지표와 함께 클러스터를 잘 형성’함을 확인하였고 → 추천 모델의 유의미한 feature로 활용 가능함을 파악했습니다.

국가별 데이터 병합의 복잡성: 세 파일 모두 Entity(국가명) 또는 Code(ISO 코드) 기준으로 매칭 가능하지만, 연도가 서로 다를 수 있으므로 가장 최신 연도 기준으로 정렬/병합이 필요합니다.

한국 관련 데이터의 자의성: 한국에 관한 데이터는 기존 CSV 파일에 존재하지 않아서 따로 정리하였습니다. 또한 한국이 들어가 있는 카테고리는 변희수 하사 사건을 통해 본 자의적 판단이기에 실제 법적 정책과는 차이가 있을 수 있습니다.

데이터의 몇년간 차이: 2022년까지의 데이터를 기반으로 하므로 그 기간동안 정책이나 지표가 변경된 국가들에 케이스에는 대응하기 어렵습니다.


4. 머신러닝 모델 구축 및 향후 계획 (ML Model & Future Plan)

4.1 적용 머신러닝 기법

비지도 학습 (Unsupervised) - K-Means Clustering 또는 DBSCAN : 전 세계 국가들을 경제 및 포용성 지표의 유사성에 따라 잠재적 이민 유형별 그룹(클러스터)으로 자동 분류하고, 사용자의 선호 유형을 식별하는 기반을 마련합니다. 
 
지도 학습 (Supervised) - Regression (회귀 분석) (Linear, Random Forest) : 사용자가 설정한 지표별 가중치를 입력받아, 이를 기반으로 계산된 '종합 이민 매력 점수(Target Score)'를 예측하여 국가 순위를 정량적으로 산출합니다. 

4.2 시스템 구축 계획: 국가 추천 ML 모델 개발

추천 목적:
사용자 값(경제 안정성 중시, 인권 중시 등)을 고려하여 국가 점수를 산출

사용 모델 후보: 

- RandomForestRegressor

- GradientBoosting (XGBoost/LightGBM)

- Multi-class Classification 기반 국가 그룹 추천

- Unsupervised Clustering 기반 유사 국가 탐색


Feature Engineering 계획:

- 인권 지표(범주형) → Label Encoding

- GDP_per_capita → Log 변환

- Military_Expenditure → GDP 대비 안정성 score로 정규화


모델–서비스 통합:

- Python 기반 추천 모델을 API 형태로 제공

- Streamlit 또는 Flask UI 구성

- 국가별 Radar Chart 시각화

- 사용자 가중치 기반 실시간 재계산 기능 구현
