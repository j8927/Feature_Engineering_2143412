from pathlib import Path
from src.data_loader import load_titanic
from src.feature_engineering import add_titanic_features
from src.eda import run_eda
from src.modeling import run_experiments, run_grid_search_for_best
from src.visualize_results import plot_metrics, plot_feature_importance
from src.explainability import run_shap_lime_analysis, run_automl_comparison


def main():
    """
    ML Feature Engineering Pipeline 실행
    
    실험 구성:
    - Base (1개): 전처리 없음 + Logistic/Random Forest
    - Mean Imputation (6가지): 다양한 인코딩, 스케일링, Feature Selection 조합
    - Median Imputation (6가지): 다양한 인코딩, 스케일링, Feature Selection 조합
    - Most Frequent (6가지): 다양한 인코딩, 스케일링, Feature Selection 조합
    
    총: 1 + 18개 조합 × 2 모델 = 38개 실험
    
    출력:
    - results/metrics/missing_ratio.csv: 결측치 비율 분석
    - results/metrics/experiment_results.csv: 38개 실험 결과 (F1, ROC_AUC 기준 정렬)
    - results/metrics/gridsearch_result.csv: GridSearchCV 최적 파라미터
    - results/figures/*.png: EDA 및 성능 비교 시각화
    """
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    df = load_titanic("data/titanic.csv")
    print("Dataset shape:", df.shape)
    print(df.info())

    # EDA before feature engineering
    missing = run_eda(df)
    print("\nMissing ratio top 10")
    print(missing.head(10))

    # Feature engineering
    df_fe = add_titanic_features(df)
    df_fe.to_csv("results/metrics/titanic_feature_engineered.csv", index=False)

    # Model comparison
    results = run_experiments(df_fe)
    results.to_csv("results/metrics/experiment_results.csv", index=False)
    print("\nExperiment Results")
    print(results)
    plot_metrics(results)

    # Additional point: GridSearchCV
    grid_result = run_grid_search_for_best(df_fe)
    grid_result.to_csv("results/metrics/gridsearch_result.csv", index=False)
    print("\nGridSearchCV Result")
    print(grid_result)

    # Explainability analysis using SHAP and LIME
    best_pipe, shap_lime_summary = run_shap_lime_analysis(df_fe)
    shap_lime_summary.to_csv("results/metrics/shap_lime_summary.csv", index=False)
    plot_feature_importance(best_pipe, ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson"], ["Sex", "Embarked", "AgeGroup", "Title"])
    print("\nSHAP/LIME explainability summary")
    print(shap_lime_summary)

    # AutoML comparison experiment
    auto_ml_result = run_automl_comparison(df_fe)
    print("\nAutoML comparison result")
    print(auto_ml_result)


if __name__ == "__main__":
    main()
