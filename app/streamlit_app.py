from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CSV_PATH, RANDOM_STATE, SUCCESS_COMPONENTS, SUCCESS_TOP_SHARE, TARGET_COLUMN, TEST_SIZE

import matplotlib.pyplot as plt

from app.gemini_report import generate_gemini_report, gemini_api_key_available
from src.features import LEAKAGE_COLUMNS, add_success_target, get_model_matrix
from src.load_data import _expand_known_header_issues, load_games_csv
from src.preprocessing import clean_and_engineer_base


APP_REPORTS_DIR = PROJECT_ROOT / "outputs" / "app_reports"
APP_FIGURES_DIR = APP_REPORTS_DIR / "figures"
GEMINI_REPORT_PATH = APP_REPORTS_DIR / "relatorio_automatizado_gemini.md"

MODEL_LABELS = {
    "LogisticRegression": "logistic_regression_baseline",
    "DecisionTreeClassifier": "decision_tree_baseline",
    "RandomForestClassifier": "random_forest",
    "ExtraTreesClassifier": "extra_trees",
    "HistGradientBoostingClassifier": "hist_gradient_boosting",
}


def main() -> None:
    st.set_page_config(page_title="Steam Games Market Insights", layout="wide")
    st.title("Steam Games Market Insights")

    uploaded_file = st.sidebar.file_uploader("CSV opcional", type=["csv"])
    selected_labels = st.sidebar.multiselect(
        "Modelos",
        options=list(MODEL_LABELS),
        default=["HistGradientBoostingClassifier"],
    )
    train_clicked = st.sidebar.button("Treinar modelos", type="primary")

    raw_df = _load_data_section(uploaded_file)
    if raw_df is None:
        return

    with st.spinner("Preparando dados e mantendo a definicao atual de success_commercial..."):
        prepared = prepare_dataset(raw_df)

    _show_dataset_summary(prepared)
    _show_leakage_check(prepared["feature_names"])

    if not selected_labels:
        st.warning("Selecione pelo menos um modelo para treinar.")
        return

    selected_keys = [MODEL_LABELS[label] for label in selected_labels]
    run_key = {
        "dataset": _dataset_signature(uploaded_file),
        "models": selected_keys,
    }

    if train_clicked:
        with st.spinner("Treinando modelos selecionados..."):
            results = train_selected_models(prepared["features"], prepared["target"], selected_keys)
        st.session_state["latest_training_run"] = {"key": run_key, "results": results}

    cached_run = st.session_state.get("latest_training_run")
    if not cached_run or cached_run["key"] != run_key:
        st.info("Selecione os modelos na barra lateral e clique em Treinar modelos.")
        return

    _show_results(cached_run["results"], prepared)


def _dataset_signature(uploaded_file) -> tuple[object, ...]:
    if uploaded_file is not None:
        size = getattr(uploaded_file, "size", None)
        if size is None:
            size = len(uploaded_file.getvalue())
        return ("uploaded", uploaded_file.name, size)
    if CSV_PATH.exists():
        return ("default", str(CSV_PATH), CSV_PATH.stat().st_mtime)
    return ("missing", str(CSV_PATH))


def _load_data_section(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is not None:
        try:
            return load_uploaded_csv(uploaded_file.getvalue())
        except Exception as exc:
            st.error(f"Falha ao ler o CSV enviado: {exc}")
            return None

    if not CSV_PATH.exists():
        st.warning("Arquivo dataset/games.csv nao encontrado.")
        st.markdown(
            "Baixe a base no Kaggle e coloque `games.csv` dentro de `dataset/` antes de rodar o app."
        )
        st.markdown("<https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?resource=download>")
        return None

    try:
        return load_default_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Falha ao carregar dataset/games.csv: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_default_csv(path: str, mtime: float) -> pd.DataFrame:
    return load_games_csv(Path(path))


@st.cache_data(show_spinner=False)
def load_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    header = next(csv.reader([raw_bytes.splitlines()[0].decode("utf-8-sig")]))
    columns = _expand_known_header_issues(header)
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        header=0,
        names=columns,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="warn",
    )


@st.cache_data(show_spinner=False)
def prepare_dataset(raw_df: pd.DataFrame) -> dict[str, object]:
    engineered_df = clean_and_engineer_base(raw_df)
    modeled_df, target_info = add_success_target(engineered_df)
    features, target, feature_names = get_model_matrix(modeled_df)
    return {
        "features": features,
        "target": target,
        "feature_names": feature_names,
        "target_info": target_info,
        "rows": len(modeled_df),
        "positive_rate": float(target.mean()),
        "leakage_features": sorted(set(feature_names).intersection(LEAKAGE_COLUMNS)),
    }


def train_selected_models(features: pd.DataFrame, target: pd.Series, selected_keys: list[str]) -> dict[str, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    model_runs = []
    metrics_rows = []
    for key in selected_keys:
        model = build_model(key)
        model.fit(x_train, y_train)
        y_pred = pd.Series(model.predict(x_test), index=y_test.index)
        y_proba = pd.Series(model.predict_proba(x_test)[:, 1], index=y_test.index)
        matrix = confusion_matrix(y_test, y_pred)
        metrics = {
            "model": key,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        }
        metrics_rows.append(metrics)
        model_runs.append({"name": key, "pipeline": model, "metrics": metrics})

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["roc_auc", "f1"], ascending=False)
    best_name = str(metrics_df.iloc[0]["model"])
    best_run = next(run for run in model_runs if run["name"] == best_name)

    APP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(APP_REPORTS_DIR / "model_metrics.csv", index=False)

    return {
        "metrics_df": metrics_df,
        "best_run": best_run,
        "x_test": x_test,
        "y_test": y_test,
    }


def build_model(name: str) -> Pipeline:
    models = {
        "logistic_regression_baseline": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
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
    return models[name]


def _show_dataset_summary(prepared: dict[str, object]) -> None:
    target_info = prepared["target_info"]
    columns = st.columns(4)
    columns[0].metric("Jogos", f"{prepared['rows']:,}".replace(",", "."))
    columns[1].metric("Features de treino", len(prepared["feature_names"]))
    columns[2].metric("Classe positiva", f"{prepared['positive_rate']:.1%}")
    columns[3].metric("Corte do score", f"{target_info['cutoff']:.3f}")

    with st.expander("Definicao da variavel-alvo"):
        st.markdown(
            f"""
`success_commercial` e uma proxy binaria, nao receita real. O alvo marca jogos no top
{int(SUCCESS_TOP_SHARE * 100)}% de um score composto por `{', '.join(SUCCESS_COMPONENTS)}`.
As colunas usadas diretamente no score e derivadas diretas sao removidas das features.
"""
        )


def _show_leakage_check(feature_names: list[str]) -> None:
    blocked = sorted(set(feature_names).intersection(LEAKAGE_COLUMNS))
    if blocked:
        st.error("Features com risco de vazamento encontradas: " + ", ".join(blocked))
    else:
        st.success("Checagem de vazamento: nenhuma coluna do alvo entrou nas features.")


def _show_results(results: dict[str, object], prepared: dict[str, object]) -> None:
    metrics_df = results["metrics_df"]
    best_run = results["best_run"]
    best_name = best_run["name"]

    st.subheader("Metricas dos modelos")
    st.caption("Melhor modelo escolhido por ROC-AUC; F1 usado como desempate.")
    st.dataframe(_style_metrics_table(metrics_df, best_name), use_container_width=True)

    metrics_fig = plot_metrics(metrics_df)
    confusion_fig = plot_confusion_matrix(best_run, results["x_test"], results["y_test"])
    top_features = extract_feature_importance(best_run["pipeline"], prepared["feature_names"])
    feature_fig = plot_feature_importance(top_features, best_name) if top_features else None

    columns = st.columns(2)
    with columns[0]:
        st.pyplot(metrics_fig)
    with columns[1]:
        st.pyplot(confusion_fig)

    if feature_fig is not None:
        st.pyplot(feature_fig)
    else:
        st.info("O melhor modelo atual nao expoe importancia de features nativa.")

    _show_gemini_section(metrics_df, best_name, top_features, prepared)


def _style_metrics_table(metrics_df: pd.DataFrame, best_name: str):
    def highlight_best(row: pd.Series) -> list[str]:
        if row["model"] == best_name:
            return ["background-color: #e8f5e9"] * len(row)
        return [""] * len(row)

    return metrics_df.style.apply(highlight_best, axis=1).format(
        {
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "roc_auc": "{:.4f}",
        }
    )


def plot_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
    APP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = metrics_df.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("Modelo")
    ax.set_title("Comparacao de metricas")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(APP_FIGURES_DIR / "metrics_bar_chart.png", dpi=160)
    return fig


def plot_confusion_matrix(best_run: dict[str, object], x_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    APP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = best_run["pipeline"].predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Nao sucesso", "Sucesso"]).plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"Matriz de confusao - {best_run['name']}")
    fig.tight_layout()
    fig.savefig(APP_FIGURES_DIR / "best_model_confusion_matrix.png", dpi=160)
    return fig


def extract_feature_importance(pipeline: Pipeline, feature_names: list[str], limit: int = 12) -> list[dict[str, float]]:
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    else:
        return []

    order = np.argsort(values)[::-1][:limit]
    return [{"feature": feature_names[index], "importance": float(values[index])} for index in order]


def plot_feature_importance(top_features: list[dict[str, float]], best_name: str) -> plt.Figure:
    APP_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = pd.DataFrame(top_features).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_xlabel("Importancia")
    ax.set_title(f"Principais features - {best_name}")
    fig.tight_layout()
    fig.savefig(APP_FIGURES_DIR / "feature_importance.png", dpi=160)
    return fig


def _show_gemini_section(
    metrics_df: pd.DataFrame,
    best_name: str,
    top_features: list[dict[str, float]],
    prepared: dict[str, object],
) -> None:
    st.subheader("Relatorio automatizado com Gemini")
    if not gemini_api_key_available():
        st.info("GEMINI_API_KEY nao configurada. O app funciona normalmente; apenas o relatorio Gemini fica desativado.")
        return

    if st.button("Gerar relatorio com Gemini"):
        payload = build_gemini_payload(metrics_df, best_name, top_features, prepared)
        try:
            with st.spinner("Gerando relatorio com Gemini..."):
                report = generate_gemini_report(payload, GEMINI_REPORT_PATH)
            st.success(f"Relatorio salvo em {GEMINI_REPORT_PATH.relative_to(PROJECT_ROOT)}")
            st.markdown(report)
        except Exception as exc:
            st.error(f"Falha ao gerar relatorio com Gemini: {exc}")


def build_gemini_payload(
    metrics_df: pd.DataFrame,
    best_name: str,
    top_features: list[dict[str, float]],
    prepared: dict[str, object],
) -> dict[str, object]:
    target_info = prepared["target_info"]
    metric_columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    return {
        "project": "Steam Games Market Insights",
        "business_problem": "Apoiar decisoes de produto, preco, posicionamento e marketing para jogos na Steam.",
        "target": {
            "name": TARGET_COLUMN,
            "type": "proxy binaria de sucesso comercial, nao receita real",
            "top_share": SUCCESS_TOP_SHARE,
            "components": list(SUCCESS_COMPONENTS),
            "cutoff": float(target_info["cutoff"]),
            "positive_rate": float(target_info["positive_rate"]),
            "leakage_columns_removed": list(target_info["leakage_columns_removed"]),
        },
        "dataset_summary": {
            "rows": int(prepared["rows"]),
            "features_used": len(prepared["feature_names"]),
        },
        "selected_model_metrics": metrics_df[metric_columns].round(4).to_dict(orient="records"),
        "best_model_by_roc_auc": best_name,
        "top_features": top_features,
        "notes": [
            "Dados brutos nao foram enviados ao Gemini.",
            "As interpretacoes devem evitar causalidade indevida.",
            "positive_ratio, Positive, Negative e derivadas diretas nao entram nas features de treino.",
        ],
    }


if __name__ == "__main__":
    main()
