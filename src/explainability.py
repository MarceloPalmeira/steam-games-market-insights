from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import FIGURES_DIR, RANDOM_STATE, TABLES_DIR
from src.modeling import ModelRun

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _final_estimator(model_run: ModelRun):
    return model_run.pipeline.named_steps["model"]


def _save_importance_plot(importance_df: pd.DataFrame, filename: str, title: str) -> None:
    top = importance_df.head(20).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 7))
    sns.barplot(data=top, x="importance", y="feature", color="#3a7ca5")
    plt.title(title)
    plt.xlabel("Importancia")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def save_model_importance(model_run: ModelRun, feature_names: list[str]) -> pd.DataFrame:
    estimator = _final_estimator(model_run)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    _save_importance_plot(importance_df, "feature_importance.png", "Importancia de features do melhor modelo")
    return importance_df


def save_shap_or_permutation(
    model_run: ModelRun,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    sample_size = min(3000, len(x_test))
    x_sample = x_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[x_sample.index]

    if importlib.util.find_spec("shap"):
        import shap

        transformed = model_run.pipeline.named_steps["imputer"].transform(x_sample)
        estimator = _final_estimator(model_run)
        explainer = shap.TreeExplainer(estimator) if hasattr(estimator, "feature_importances_") else shap.Explainer(estimator)
        shap_values = explainer(transformed)
        plt.figure()
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        result = (
            pd.DataFrame({"feature": feature_names, "importance": shap_importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        result.to_csv(TABLES_DIR / "shap_importance.csv", index=False)
        return result

    print("SHAP nao instalado; usando permutation importance como alternativa.")
    permutation = permutation_importance(
        model_run.pipeline,
        x_sample,
        y_sample,
        scoring="f1",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    result = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    result.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)
    _save_importance_plot(result, "permutation_importance.png", "Permutation importance do melhor modelo")
    return result


def run_explainability(
    model_run: ModelRun,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_importance = save_model_importance(model_run, feature_names)
    fallback_importance = save_shap_or_permutation(model_run, x_test, y_test, feature_names)

    if feature_importance["importance"].sum() == 0 and not fallback_importance.empty:
        feature_importance = fallback_importance[["feature", "importance"]].copy()
        feature_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
        _save_importance_plot(
            feature_importance,
            "feature_importance.png",
            "Importancia de features do melhor modelo",
        )

    return feature_importance, fallback_importance
