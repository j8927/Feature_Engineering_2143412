from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from src.modeling import get_transformed_feature_names


def plot_metrics(results: pd.DataFrame, fig_dir: str = "results/figures"):
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results, x="Experiment", y="F1", hue="Model")
    plt.title("F1-score by Experiment and Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "comparison_f1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results, x="Experiment", y="ROC_AUC", hue="Model")
    plt.title("ROC-AUC by Experiment and Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "comparison_roc_auc.png", dpi=150)
    plt.close()


def plot_feature_importance(pipe, numeric_features, categorical_features, fig_dir: str = "results/figures"):
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    model = pipe.named_steps.get("model")
    preprocessor = pipe.named_steps.get("preprocessor")
    if model is None or preprocessor is None:
        raise ValueError("Pipeline must include both a preprocessor and a model step for feature importance plotting.")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Feature Importances"
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
        title = "Feature Coefficients (abs)"
    else:
        raise ValueError("Model does not expose feature_importances_ or coef_.")

    feature_names = get_transformed_feature_names(preprocessor, numeric_features, categorical_features)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(Path(fig_dir) / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(20), x="importance", y="feature", color="steelblue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "feature_importance.png", dpi=150)
    plt.close()
