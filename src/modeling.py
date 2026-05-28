from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@dataclass
class ExperimentConfig:
    name: str
    imputer: Optional[str]
    encoding: Optional[str]
    scaler: Optional[str]
    feature_selection: bool
    use_engineered: bool = True


def _get_scaler(name: Optional[str]):
    if name is None:
        return "passthrough"
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }[name.lower()]


def _get_encoder(name: Optional[str]):
    if name is None:
        return "passthrough"
    if name.lower() == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name.lower() == "label":
        # OrdinalEncoder is pipeline-safe and is used here as a feature-level label encoding alternative.
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unknown encoder: {name}")


def build_pipeline(config: ExperimentConfig, numeric_features: List[str], categorical_features: List[str], model_name: str):
    if config.imputer is None:
        numeric_pipe = Pipeline([("scaler", _get_scaler(config.scaler))])
        categorical_pipe = Pipeline([("encoder", _get_encoder(config.encoding))])
    else:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=config.imputer)),
            ("scaler", _get_scaler(config.scaler)),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _get_encoder(config.encoding)),
        ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ], remainder="drop")

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    else:
        raise ValueError("model_name must be logistic_regression or random_forest")

    steps = [("preprocessor", preprocessor)]
    if config.feature_selection:
        steps.append(("select", SelectKBest(score_func=f_classif, k=10)))
    steps.append(("model", model))
    return Pipeline(steps)


def get_transformed_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    transformed_features = []
    transformed_features.extend(numeric_features)

    cat_transformer = preprocessor.named_transformers_.get("cat")
    if cat_transformer is not None:
        # If the categorical branch is itself a Pipeline, pull its encoder
        if hasattr(cat_transformer, "named_steps"):
            encoder = cat_transformer.named_steps.get("encoder")
        else:
            encoder = cat_transformer

        if encoder is None or encoder == "passthrough":
            transformed_features.extend(categorical_features)
        elif hasattr(encoder, "get_feature_names_out"):
            transformed_features.extend(encoder.get_feature_names_out(categorical_features).tolist())
        else:
            transformed_features.extend(categorical_features)
    else:
        transformed_features.extend(categorical_features)

    return transformed_features


def evaluate_model(pipe, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        proba = pred
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
    }


def run_experiments(df: pd.DataFrame) -> pd.DataFrame:
    base_numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    engineered_numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson"]
    engineered_categorical = ["Sex", "Embarked", "AgeGroup", "Title"]

    configs = [
        # Base 실험 (전처리 없음)
        ExperimentConfig("Base", None, None, None, False, False),
        
        # Mean Imputation 조합
        ExperimentConfig("Exp-1", "mean", "onehot", "standard", False, True),
        ExperimentConfig("Exp-2", "mean", "onehot", "standard", True, True),
        ExperimentConfig("Exp-3", "mean", "label", "standard", False, True),
        ExperimentConfig("Exp-4", "mean", "label", "standard", True, True),
        ExperimentConfig("Exp-5", "mean", "onehot", "minmax", False, True),
        ExperimentConfig("Exp-6", "mean", "label", "minmax", True, True),
        
        # Median Imputation 조합
        ExperimentConfig("Exp-7", "median", "onehot", "minmax", False, True),
        ExperimentConfig("Exp-8", "median", "onehot", "minmax", True, True),
        ExperimentConfig("Exp-9", "median", "label", "minmax", False, True),
        ExperimentConfig("Exp-10", "median", "label", "minmax", True, True),
        ExperimentConfig("Exp-11", "median", "onehot", "robust", False, True),
        ExperimentConfig("Exp-12", "median", "label", "robust", True, True),
        
        # Most Frequent Imputation 조합
        ExperimentConfig("Exp-13", "most_frequent", "onehot", "robust", False, True),
        ExperimentConfig("Exp-14", "most_frequent", "onehot", "robust", True, True),
        ExperimentConfig("Exp-15", "most_frequent", "label", "robust", False, True),
        ExperimentConfig("Exp-16", "most_frequent", "label", "robust", True, True),
        ExperimentConfig("Exp-17", "most_frequent", "onehot", "standard", False, True),
        ExperimentConfig("Exp-18", "most_frequent", "label", "minmax", True, True),
    ]

    results = []
    y = df["Survived"]

    for config in configs:
        if config.name == "Base":
            X = df[base_numeric].dropna()
            y_use = y.loc[X.index]
            numeric_features = base_numeric
            categorical_features = []
        else:
            X = df[engineered_numeric + engineered_categorical]
            y_use = y
            numeric_features = engineered_numeric
            categorical_features = engineered_categorical

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_use, test_size=0.2, random_state=42, stratify=y_use
        )

        for model_name in ["logistic_regression", "random_forest"]:
            pipe = build_pipeline(config, numeric_features, categorical_features, model_name)
            metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)
            results.append({
                "Experiment": config.name,
                "Missing": "None" if config.imputer is None else config.imputer,
                "Encoding": "None" if config.encoding is None else config.encoding,
                "Scaling": "None" if config.scaler is None else config.scaler,
                "FeatureSelection": "O" if config.feature_selection else "X",
                "Model": model_name,
                **metrics,
            })

    return pd.DataFrame(results).sort_values(["F1", "ROC_AUC"], ascending=False)


def run_grid_search_for_best(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson"]
    categorical_features = ["Sex", "Embarked", "AgeGroup", "Title"]
    X = df[numeric_features + categorical_features]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    config = ExperimentConfig("GridSearch", "median", "onehot", "standard", False, True)
    pipe = build_pipeline(config, numeric_features, categorical_features, "random_forest")
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [4, 6, None],
        "model__min_samples_split": [2, 5],
    }
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    metrics = evaluate_model(grid.best_estimator_, X_train, X_test, y_train, y_test)
    return pd.DataFrame([{ "BestParams": str(grid.best_params_), **metrics }])
