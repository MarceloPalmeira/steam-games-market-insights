from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from scipy import stats

from src.config import FIGURES_DIR, SCORE_COLUMN, TABLES_DIR, TARGET_COLUMN

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def save_dataset_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="count")
        .assign(share=lambda table: table["count"] / table["count"].sum())
    )

    numeric_columns = [
        "owners_midpoint",
        "peak_ccu",
        "price",
        "total_reviews",
        "positive_ratio",
        "recommendations",
        "average_playtime_forever",
        "metacritic_score",
        "achievements",
        SCORE_COLUMN,
    ]
    numeric_summary = df[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).T
    null_summary = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
        .assign(missing_share=lambda table: table["missing_count"] / len(df))
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "columns_after_engineering", "value": df.shape[1]},
            {"metric": "success_rate", "value": df[TARGET_COLUMN].mean()},
            {"metric": "median_price", "value": df["price"].median()},
            {"metric": "median_total_reviews", "value": df["total_reviews"].median()},
            {"metric": "median_owners_midpoint", "value": df["owners_midpoint"].median()},
        ]
    )

    target_distribution.to_csv(TABLES_DIR / "target_distribution.csv", index=False)
    numeric_summary.to_csv(TABLES_DIR / "numeric_descriptive_stats.csv")
    null_summary.to_csv(TABLES_DIR / "missing_values.csv", index=False)
    summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    return {
        "target_distribution": target_distribution,
        "numeric_summary": numeric_summary,
        "null_summary": null_summary,
        "summary": summary,
    }


def save_group_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "free_to_play": "Gratuito",
        "has_multiplayer": "Multiplayer",
        "is_indie": "Indie",
        "has_achievements": "Com achievements",
        "has_metacritic": "Com Metacritic",
        "has_singleplayer": "Single-player",
    }
    for column, label in groups.items():
        grouped = df.groupby(column).agg(
            games=(TARGET_COLUMN, "size"),
            success_rate=(TARGET_COLUMN, "mean"),
            median_score=(SCORE_COLUMN, "median"),
            median_price=("price", "median"),
            median_reviews=("total_reviews", "median"),
        )
        for value, metrics in grouped.iterrows():
            rows.append(
                {
                    "group": label,
                    "value": int(value),
                    "games": int(metrics["games"]),
                    "success_rate": float(metrics["success_rate"]),
                    "median_score": float(metrics["median_score"]),
                    "median_price": float(metrics["median_price"]),
                    "median_reviews": float(metrics["median_reviews"]),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "group_comparisons.csv", index=False)
    return result


def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    tests = []
    for column, description in [
        ("has_multiplayer", "Score comercial: jogos multiplayer vs nao multiplayer"),
        ("free_to_play", "Score comercial: jogos gratuitos vs pagos"),
        ("is_indie", "Score comercial: jogos indie vs nao indie"),
    ]:
        group_0 = df.loc[df[column] == 0, SCORE_COLUMN].dropna()
        group_1 = df.loc[df[column] == 1, SCORE_COLUMN].dropna()
        if len(group_0) > 20 and len(group_1) > 20:
            statistic, p_value = stats.mannwhitneyu(group_1, group_0, alternative="two-sided")
            tests.append(
                {
                    "test": "Mann-Whitney U",
                    "description": description,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "group_0_median": float(group_0.median()),
                    "group_1_median": float(group_1.median()),
                }
            )

    for column, description in [
        ("has_multiplayer", "Sucesso comercial vs multiplayer"),
        ("free_to_play", "Sucesso comercial vs gratuito"),
    ]:
        contingency = pd.crosstab(df[column], df[TARGET_COLUMN])
        if contingency.shape == (2, 2):
            statistic, p_value, _, _ = stats.chi2_contingency(contingency)
            tests.append(
                {
                    "test": "Qui-quadrado",
                    "description": description,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "group_0_median": np.nan,
                    "group_1_median": np.nan,
                }
            )

    result = pd.DataFrame(tests)
    result.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)
    return result


def save_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=target_counts.index.map({0: "Nao sucesso", 1: "Sucesso"}), y=target_counts.values)
    plt.title("Distribuicao da variavel-alvo")
    plt.xlabel("")
    plt.ylabel("Numero de jogos")
    _save_current_figure("target_distribution.png")

    paid_prices = df.loc[df["price"] > 0, "price"].clip(upper=df["price"].quantile(0.99))
    plt.figure(figsize=(8, 4))
    sns.histplot(paid_prices, bins=40, color="#2f6f73")
    plt.title("Distribuicao de precos pagos (limitada ao p99)")
    plt.xlabel("Preco")
    plt.ylabel("Jogos")
    _save_current_figure("price_distribution_paid_p99.png")

    plt.figure(figsize=(8, 4))
    plot_data = df[[TARGET_COLUMN, "total_reviews"]].copy()
    plot_data["log_total_reviews"] = np.log1p(plot_data["total_reviews"])
    sns.boxplot(data=plot_data, x=TARGET_COLUMN, y="log_total_reviews")
    plt.title("Volume de reviews por classe de sucesso")
    plt.xlabel("Sucesso comercial")
    plt.ylabel("log(1 + total de reviews)")
    _save_current_figure("reviews_by_success.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df.groupby("has_multiplayer", as_index=False)[TARGET_COLUMN].mean(),
        x="has_multiplayer",
        y=TARGET_COLUMN,
        color="#3a7ca5",
    )
    plt.title("Taxa de sucesso por presenca de multiplayer")
    plt.xlabel("Tem multiplayer")
    plt.ylabel("Taxa de sucesso")
    _save_current_figure("success_rate_multiplayer.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df.groupby("free_to_play", as_index=False)[TARGET_COLUMN].mean(),
        x="free_to_play",
        y=TARGET_COLUMN,
        color="#8b5e34",
    )
    plt.title("Taxa de sucesso por modelo gratuito/pago")
    plt.xlabel("Gratuito")
    plt.ylabel("Taxa de sucesso")
    _save_current_figure("success_rate_free_to_play.png")

    corr_columns = [
        "price",
        "achievements",
        "metacritic_score",
        "category_count",
        "genre_count",
        "tag_count",
        "positive_ratio",
        SCORE_COLUMN,
    ]
    plt.figure(figsize=(8, 6))
    corr = df[corr_columns].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlacoes entre variaveis numericas selecionadas")
    _save_current_figure("numeric_correlations.png")


def run_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = save_dataset_tables(df)
    tables["group_comparisons"] = save_group_comparisons(df)
    tables["statistical_tests"] = run_statistical_tests(df)
    save_figures(df)
    return tables
