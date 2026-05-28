# ML Feature Engineering Pipeline Project

## 1. 과제 개요
본 프로젝트는 Titanic Dataset을 활용하여 머신러닝 성능 향상을 위한 특성 공학(Feature Engineering) 파이프라인을 구현하고, 전처리 및 변수 선택 전략에 따라 성능 차이를 비교·분석합니다.

### 목표
- 데이터 탐색(EDA)부터 결측치 처리, 인코딩, 스케일링, 파생 변수 생성, 변수 선택, 모델 학습 및 평가까지 전체 ML Pipeline 구현
- 서로 다른 전처리 전략에 따른 모델 성능 비교
- 결과를 바탕으로 최적의 Feature Engineering 전략 도출

## 2. 데이터셋 소개
- Dataset: Titanic Dataset
- 과제 유형: Classification
- 목적: 생존 여부 예측 (`Survived`)
- 특성: 수치형 + 범주형 혼합, 결측치 포함, 500개 이상 샘플
- 데이터 로드: `data/titanic.csv`가 없으면 공개 GitHub 미러에서 자동 다운로드

## 3. 과제 구성
### STEP 01. 데이터 준비
- 데이터 로드 및 기본 구조 확인
- 타겟 변수 정의: `Survived`
- 데이터셋 소개 및 컬럼 설명

### STEP 02. 탐색적 데이터 분석 (EDA)
- 결측치 비율 분석
- 이상치 탐색
- 변수 분포 시각화
- 상관관계 분석
- 타겟 분포 확인

### STEP 03. 특성 공학 파이프라인 구현
- 결측치 처리 비교: Mean / Median / Most Frequent / 없음
- 범주형 인코딩 비교: One-Hot / Label
- 스케일링 비교: StandardScaler / MinMaxScaler / RobustScaler
- 파생 변수 생성: `FamilySize`, `IsAlone`, `FarePerPerson`, `AgeGroup`, `Title`

### STEP 04. 변수 선택 (Feature Selection)
- SelectKBest 적용
- 선택 전/후 성능 비교

### STEP 05. 모델 학습 및 평가
- 사용 모델: Logistic Regression, Random Forest
- 평가 지표: Accuracy, Precision, Recall, F1-score, ROC-AUC

## 4. 프로젝트 구조
```text
.
├── data/                     # 데이터 저장 위치
├── notebooks/                # Jupyter Notebook
├── report/                   # 보고서 템플릿 및 목차
├── results/
│   ├── figures/              # EDA 및 성능 시각화
│   └── metrics/              # 결과 CSV
├── src/                      # 분석 코드
├── main.py                   # 전체 실행 파일
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. 실행 방법
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 6. 출력 결과
- `results/metrics/missing_ratio.csv` : 결측치 비율 분석
- `results/metrics/experiment_results.csv` : 실험별 모델 성능 비교
- `results/metrics/gridsearch_result.csv` : GridSearchCV 최적 하이퍼파라미터
- `results/metrics/shap_lime_summary.csv` : SHAP 및 LIME 설명 가능성 분석 요약
- `results/metrics/automl_result.csv` : AutoML(TPOT) 비교 실험 성능
- `results/metrics/tpot_exported_pipeline.py` : TPOT AutoML에서 생성한 최적 파이프라인 코드
- `results/figures/` : EDA, 모델 성능, SHAP/LIME 및 Feature Importance 시각화 이미지

## 7. 실험 설계
본 프로젝트는 다음의 조합별 실험을 수행하여 전처리 전략의 효과를 비교합니다.

### 기본 실험
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Base | 없음 | 없음 | 없음 | 없음 |

### Mean Imputation 조합
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Exp-1 | Mean | One-Hot | Standard | X |
| Exp-2 | Mean | One-Hot | Standard | O |
| Exp-3 | Mean | Label | Standard | X |
| Exp-4 | Mean | Label | Standard | O |
| Exp-5 | Mean | One-Hot | MinMax | X |
| Exp-6 | Mean | Label | MinMax | O |

### Median Imputation 조합
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Exp-7 | Median | One-Hot | MinMax | X |
| Exp-8 | Median | One-Hot | MinMax | O |
| Exp-9 | Median | Label | MinMax | X |
| Exp-10 | Median | Label | MinMax | O |
| Exp-11 | Median | One-Hot | Robust | X |
| Exp-12 | Median | Label | Robust | O |

### Most Frequent Imputation 조합
| 실험 | 결측치 처리 | 인코딩 | 스케일링 | Feature Selection |
|---|---|---|---|---|
| Exp-13 | Most Frequent | One-Hot | Robust | X |
| Exp-14 | Most Frequent | One-Hot | Robust | O |
| Exp-15 | Most Frequent | Label | Robust | X |
| Exp-16 | Most Frequent | Label | Robust | O |
| Exp-17 | Most Frequent | One-Hot | Standard | X |
| Exp-18 | Most Frequent | Label | MinMax | O |

각 실험은 **Logistic Regression**과 **Random Forest** 2개 모델로 평가되어 총 38개의 결과를 도출합니다.

## 8. 생성된 파생 변수
- `FamilySize` : `SibSp + Parch + 1`
- `IsAlone` : 가족 없이 혼자인 경우
- `FarePerPerson` : `Fare / FamilySize`
- `AgeGroup` : 연령대 범주화
- `Title` : 이름에서 추출한 호칭

## 9. 평가 지표
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## 10. 가산점 요소
- `sklearn.Pipeline` 객체 활용
- `ColumnTransformer` 기반 전처리
- `GridSearchCV` 적용
- `SHAP` 및 `LIME` 기반 설명 가능성 분석
- `TPOT` AutoML 비교 실험
- Feature Importance 시각화 고도화

## 11. 보고서 작성 안내
- 데이터셋 소개
- EDA 결과
- Feature Engineering 과정
- 모델 학습 및 평가
- 실험 비교 표
- 최종 결론

## 12. GitHub 제출 방법
```bash
git init
git add .
git commit -m "Initial ML feature engineering project"
git branch -M main
git remote add origin 본인깃허브주소
git push -u origin main
```
