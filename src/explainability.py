from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tpot import TPOTClassifier
from src.modeling import ExperimentConfig, build_pipeline, get_transformed_feature_names


def _get_automl_pipeline(numeric_features, categorical_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop")

    return TPOTClassifier(
        search_space="linear",
        preprocessing=preprocessor,
        max_time_mins=3,
        max_eval_time_mins=10,
        n_jobs=1,
        client=None,
        verbose=0,
        random_state=42,
    )


def _save_shap_visuals(pipe, X_train: pd.DataFrame, feature_names, fig_dir: Path):
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fig = shap.summary_plot(
        shap_values,
        X_train_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=150)
    plt.close()

    fig = shap.summary_plot(
        shap_values,
        X_train_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_feature_importance.png", dpi=150)
    plt.close()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_abs_mean": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_abs_mean", ascending=False)
    shap_df.to_csv(fig_dir / "shap_feature_importance.csv", index=False)
    return shap_df


def _save_lime_explanation(pipe, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names, fig_dir: Path):
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    explainer = LimeTabularExplainer(
        X_train_transformed,
        feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        discretize_continuous=True,
        random_state=42,
    )

    sample_idx = 0
    explanation = explainer.explain_instance(
        X_test_transformed[sample_idx],
        model.predict_proba,
        num_features=min(10, X_test_transformed.shape[1]),
    )

    html_path = fig_dir / "lime_explanation_sample0.html"
    explanation.save_to_file(str(html_path))

    text_path = fig_dir / "lime_explanation_sample0.txt"
    with open(text_path, "w", encoding="utf8") as f:
        for name, weight in explanation.as_list():
            f.write(f"{name}: {weight:.4f}\n")

    return {"html_path": str(html_path), "text_path": str(text_path)}


def run_shap_lime_analysis(df: pd.DataFrame, fig_dir: str = "results/figures") -> pd.DataFrame:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson"]
    categorical_features = ["Sex", "Embarked", "AgeGroup", "Title"]
    X = df[numeric_features + categorical_features]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    config = ExperimentConfig("Explain", "median", "onehot", "standard", False, True)
    pipe = build_pipeline(config, numeric_features, categorical_features, "random_forest")
    pipe.fit(X_train, y_train)

    feature_names = get_transformed_feature_names(pipe.named_steps["preprocessor"], numeric_features, categorical_features)
    shap_df = _save_shap_visuals(pipe, X_train, feature_names, fig_dir)
    lime_info = _save_lime_explanation(pipe, X_train, X_test, feature_names, fig_dir)

    metrics = {
        "Accuracy": accuracy_score(y_test, pipe.predict(X_test)),
        "Precision": precision_score(y_test, pipe.predict(X_test), zero_division=0),
        "Recall": recall_score(y_test, pipe.predict(X_test), zero_division=0),
        "F1": f1_score(y_test, pipe.predict(X_test), zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
    }

    summary = pd.DataFrame([
        {
            "Method": "SHAP",
            "TopFeatures": ", ".join(shap_df.head(10)["feature"].tolist()),
            "LIME_HTML": lime_info["html_path"],
            "LIME_TXT": lime_info["text_path"],
            **metrics,
        }
    ])
    summary.to_csv("results/metrics/shap_lime_summary.csv", index=False)
    return pipe, summary


def run_automl_comparison(df: pd.DataFrame, fig_dir: str = "results/figures") -> pd.DataFrame:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson"]
    categorical_features = ["Sex", "Embarked", "AgeGroup", "Title"]
    X = df[numeric_features + categorical_features]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    automl = _get_automl_pipeline(numeric_features, categorical_features)
    automl.fit(X_train, y_train)

    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test)[:, 1]
    auto_metrics = {
        "Model": "TPOT AutoML",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    result = pd.DataFrame([auto_metrics])
    result.to_csv("results/metrics/automl_result.csv", index=False)

    pipeline_path = Path("results/metrics/tpot_exported_pipeline.joblib")
    dump(automl.fitted_pipeline_, pipeline_path)

    export_code_path = Path("results/metrics/tpot_exported_pipeline.py")
    with open(export_code_path, "w", encoding="utf8") as f:
        f.write(
            "from joblib import load\n"
            "\n"
            "pipeline = load(\"results/metrics/tpot_exported_pipeline.joblib\")\n"
            "\n"
            "def predict(X):\n"
            "    return pipeline.predict(X)\n"
            "\n"
            "def predict_proba(X):\n"
            "    return pipeline.predict_proba(X)\n"
        )

    return result
