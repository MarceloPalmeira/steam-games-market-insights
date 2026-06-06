from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, MODELS_DIR, OUTPUT_DIR, RANDOM_STATE, TABLES_DIR, TEST_SIZE


@dataclass
class ModelRun:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float | str]


def _predict_scores(model: Pipeline, x_test: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    y_pred = pd.Series(model.predict(x_test), index=x_test.index)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        y_proba = pd.Series(proba[:, 1], index=x_test.index)
    return y_pred, y_proba


def _evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    y_pred, y_proba = _predict_scores(model, x_test)
    metrics: dict[str, float | str] = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else float("nan"),
    }
    matrix = confusion_matrix(y_test, y_pred)
    metrics["tn"] = int(matrix[0, 0])
    metrics["fp"] = int(matrix[0, 1])
    metrics["fn"] = int(matrix[1, 0])
    metrics["tp"] = int(matrix[1, 1])
    return metrics


def _cross_validation_metrics(name: str, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float | str]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}
    scores = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "model": name,
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_f1_std": float(scores["test_f1"].std()),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
    }


def _build_models() -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "logistic_regression_baseline": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree_baseline": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=50,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=14,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=100,
                        max_depth=14,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=160,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        models["xgboost_optional"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=160,
                        max_depth=5,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    else:
        print("XGBoost nao instalado; modelo opcional pulado.")

    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier

        models["lightgbm_optional"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.06,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    else:
        print("LightGBM nao instalado; modelo opcional pulado.")

    return models


def train_and_evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[ModelRun, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    model_runs: list[ModelRun] = []
    metrics_rows: list[dict[str, float | str]] = []
    cv_rows: list[dict[str, float | str]] = []

    for name, model in _build_models().items():
        cv_rows.append(_cross_validation_metrics(name, model, x_train, y_train))
        model.fit(x_train, y_train)
        metrics = _evaluate_model(name, model, x_test, y_test)
        metrics_rows.append(metrics)
        model_runs.append(ModelRun(name=name, pipeline=model, metrics=metrics))

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["f1", "roc_auc"], ascending=False)
    cv_df = pd.DataFrame(cv_rows)
    metrics_df.to_csv(TABLES_DIR / "model_metrics.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    cv_df.to_csv(TABLES_DIR / "model_cv_metrics.csv", index=False)

    best_name = str(metrics_df.iloc[0]["model"])
    best_run = next(run for run in model_runs if run.name == best_name)
    joblib.dump(best_run.pipeline, MODELS_DIR / "best_model.joblib")

    y_pred, y_proba = _predict_scores(best_run.pipeline, x_test)
    predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba if y_proba is not None else pd.NA,
        }
    )
    predictions.to_csv(TABLES_DIR / "best_model_test_predictions.csv", index=False)

    return best_run, metrics_df, cv_df, x_test, y_test
